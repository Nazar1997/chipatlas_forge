"""Stage 1 -- cut one allPeaks_light.<org>.05.bed.gz into ordered shards.

This is the only serial stage, and it is serial for a reason worth writing down:
the ChIP-Atlas archives are **plain gzip, not BGZF**. A plain DEFLATE stream has
no independently addressable blocks, so it cannot be seeked into, split by byte
range, or decompressed by more than one thread. pigz parallelises compression,
not decompression. There is no arrangement of SLURM tasks that makes a single
10 GB .gz decompress faster than one core can inflate it.

So the pass happens exactly once, and its only job is to cut the stream into
pieces that *are* independently readable. Everything expensive -- parsing,
accession lookup, grouping -- happens in stage 2, which is 100-way parallel over
those pieces. Shards also survive across reruns: change the grouping and rerun
stages 2-3 without paying for the inflate again.

Two properties the later stages depend on:

* **Shards break only on newlines**, so every shard is independently parseable.
* **Shard order is genomic order.** The input is sorted by (chrom, start) --
  verified over the leading 3M peaks -- and nothing here reorders. Reading shards
  in index order therefore yields sorted output with no sort step anywhere.
  Sorting ~1.5 billion rows would otherwise dominate the entire pipeline.

The splitting is done here rather than by ``split --filter`` because that flag,
and ``-C``, are GNU extensions absent on BSD/macOS -- which would make the one
stage with a subtle invariant the one stage that could not be tested off-cluster.
This process only shuttles bytes between two subprocesses that run concurrently,
so it is not the bottleneck: the inflate is.

Usage:
    python -m chipatlas_forge.shard --root . --org hg38 --chunk-size 512M
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

READ_BLOCK = 8 << 20

_SIZE = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMGT]?)B?$", re.I)
_SCALE = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def parse_size(text: str) -> int:
    match = _SIZE.match(str(text).strip())
    if not match:
        raise ValueError("cannot read %r as a byte size (try 512M)" % text)
    return int(float(match.group(1)) * _SCALE[match.group(2).upper()])


def _pick(*candidates):
    for name in candidates:
        if shutil.which(name):
            return name
    return None


class ShardWriter:
    """Writes one shard, through a compressor subprocess when one is available."""

    def __init__(self, path: Path, tool: str, level: int):
        self.path = path
        self.bytes = 0
        if tool == "zstd":
            self._proc = subprocess.Popen(
                ["zstd", "-q", "-%d" % level, "-o", str(path)],
                stdin=subprocess.PIPE)
            self._sink = self._proc.stdin
        elif tool == "gzip":
            target = open(path, "wb")
            self._proc = subprocess.Popen([_pick("pigz", "gzip"), "-c", "-%d" % level],
                                          stdin=subprocess.PIPE, stdout=target)
            target.close()
            self._sink = self._proc.stdin
        else:
            self._proc = None
            self._sink = open(path, "wb")

    def write(self, blob: bytes):
        self._sink.write(blob)
        self.bytes += len(blob)

    def close(self):
        self._sink.close()
        if self._proc is not None and self._proc.wait() != 0:
            raise SystemExit("compressor failed writing %s" % self.path)


def split_stream(stream, chunk_bytes: int, open_writer) -> list:
    """Cut a byte stream into newline-aligned shards of at least ``chunk_bytes``.

    A shard is closed at the first newline *at or after* the threshold, so
    shards are slightly over the target rather than under and no record is ever
    divided. ``open_writer(index)`` supplies each shard's sink.

    The cut is searched for *inside* the block rather than only at block
    boundaries. Checking only at boundaries silently degrades to one enormous
    shard whenever the read size exceeds the chunk size -- which is invisible at
    the 512M default and immediate at any small value.
    """
    writers, writer, written = [], None, 0
    read_size = max(1 << 16, min(READ_BLOCK, chunk_bytes))
    while True:
        block = stream.read(read_size)
        if not block:
            break
        pos = 0
        while pos < len(block):
            if writer is None:
                writer, written = open_writer(len(writers)), 0
            if written >= chunk_bytes:
                cut = block.find(b"\n", pos)
                if cut >= 0:
                    writer.write(block[pos:cut + 1])
                    writer.close()
                    writers.append(writer)
                    writer, pos = None, cut + 1
                    continue
            # Either under the threshold, or over it with no newline left in
            # this block -- take the remainder and cut in a later one.
            writer.write(block[pos:])
            written += len(block) - pos
            pos = len(block)
    if writer is not None:
        writer.close()
        writers.append(writer)
    return writers


def shard_org(root: Path, org: str, chunk_size: str, level: int) -> dict:
    source = root / "raw" / ("allPeaks_light.%s.05.bed.gz" % org)
    if not source.exists():
        raise SystemExit("missing %s" % source)

    out_dir = root / "work" / "shards" / org
    if out_dir.exists():
        stale = sorted(out_dir.glob("shard_*"))
        if stale:
            print("  clearing %d shards from a previous run" % len(stale))
            for path in stale:
                path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    if _pick("zstd"):
        # zstd -1 compresses at ~400 MB/s per core, well ahead of the ~150-200
        # MB/s the inflate delivers, so it never becomes the limiting stage.
        tool, suffix = "zstd", ".zst"
    elif _pick("pigz", "gzip"):
        tool, suffix = "gzip", ".gz"
    else:
        print("no zstd or gzip; writing uncompressed shards (~6x the disk)",
              file=sys.stderr)
        tool, suffix = "none", ""

    # pigz cannot split the inflate across cores, but it does move reading and
    # writing onto their own threads, which is worth roughly a third.
    reader = _pick("pigz", "gzip")
    if reader is None:
        raise SystemExit("need pigz or gzip on PATH to read %s" % source)

    chunk_bytes = parse_size(chunk_size)
    started = time.time()
    proc = subprocess.Popen([reader, "-dc", str(source)], stdout=subprocess.PIPE)
    try:
        writers = split_stream(
            proc.stdout, chunk_bytes,
            lambda i: ShardWriter(out_dir / ("shard_%04d%s" % (i, suffix)), tool, level),
        )
    finally:
        proc.stdout.close()
        if proc.wait() != 0:
            raise SystemExit("%s failed inflating %s" % (reader, source))
    elapsed = time.time() - started

    if not writers:
        raise SystemExit("sharding %s produced nothing" % org)
    on_disk = sum(w.path.stat().st_size for w in writers)
    raw_bytes = sum(w.bytes for w in writers)

    index = {
        "org": org, "source": str(source), "chunk_size": chunk_size,
        "suffix": suffix, "compressor": tool, "n_shards": len(writers),
        "uncompressed_bytes": raw_bytes, "bytes_on_disk": on_disk,
        "seconds": round(elapsed, 1),
        "shards": [w.path.name for w in writers],
    }
    (out_dir / "shards.json").write_text(json.dumps(index, indent=2))
    print("  %d shards, %.1f GB text -> %.1f GB on disk, %.1f min (%.0f MB/s)"
          % (len(writers), raw_bytes / 1e9, on_disk / 1e9, elapsed / 60,
             raw_bytes / 1e6 / max(elapsed, 1e-9)), flush=True)
    return index


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--org", nargs="+", default=["hg38"])
    parser.add_argument("--chunk-size", default="512M",
                        help="uncompressed bytes per shard; sets the stage-2 "
                             "array width (hg38 is ~55 GB, so 512M ~ 110 tasks)")
    parser.add_argument("--level", type=int, default=1,
                        help="compressor level; 1 keeps it ahead of the inflate")
    args = parser.parse_args(argv)

    for org in args.org:
        print("sharding %s" % org, flush=True)
        shard_org(args.root, org, args.chunk_size, args.level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
