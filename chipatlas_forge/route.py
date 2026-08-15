"""Stage 2 -- map. One SLURM array task per shard; the parallel half of the job.

Each task reads one shard, resolves every peak's accession to a group id, and
writes the rows out partitioned by bucket. Tasks never talk to each other and
never touch the same output file, so the array width is free to be whatever the
queue will give you.

Three choices carry essentially all of the performance here:

**Dictionary-encoded accessions.** A shard holds ~15M peaks drawn from only
~100k distinct accessions -- the file is sorted by position, so every accession
recurs throughout. Asking pyarrow for the SRX column as
``dictionary<int32, string>`` means the string->group lookup runs once per
*distinct* accession instead of once per row: ~100k Python dict hits (about
20 ms) rather than 15M. The per-row step is then ``lut[indices]``, one numpy
take.

**pyarrow's CSV reader, not pandas or split().** It is multithreaded C++ and
parses a 512 MB shard in a couple of seconds. Python-level line splitting on
1.5 billion rows is the difference between minutes and a day.

**Bucketing, not per-group files.** Writing one file per group per shard would
be 110 x 3,172 = 349k files for hg38 alone. Hashing groups into a fixed number
of buckets keeps it at 110 x n_buckets, and because a group's rows always land
in the same bucket, stage 3 can finish a group by reading one bucket.

Row order within each bucket is the shard's own order, which is genomic order --
see shard.py. That is preserved through a *stable* sort here and is what makes
the final output sorted for free.

Usage:
    python -m chipatlas_forge.route --root . --org hg38 --shard 7
    python -m chipatlas_forge.route --root . --org hg38 --shard all
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from . import arrow_compat as compat

BED_COLUMNS = ["chrom", "start", "end", "srx", "score"]


def load_lookup(root: Path, org: str) -> dict:
    base = root / "work" / "manifest" / org
    srx = np.load(base / "srx.npy")
    gid = np.load(base / "group_id.npy")
    return dict(zip(srx.tolist(), gid.tolist()))


def open_shard(path: Path):
    """A binary stream of the shard's text, and the process to reap.

    Whichever compressor stage 1 had available; it records the choice in
    shards.json but the suffix is enough to reverse it.
    """
    if path.suffix == ".zst":
        cmd = ["zstd", "-dc", str(path)]
    elif path.suffix == ".gz":
        cmd = [shutil.which("pigz") or "gzip", "-dc", str(path)]
    else:
        return open(path, "rb"), None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    return proc.stdout, proc


def read_shard(path: Path, block_size: int) -> tuple:
    """Parse one shard into an Arrow table. Returns (table, n_bad_rows)."""
    bad = []

    def on_invalid(row):
        # A truncated or over-wide line. Count it, keep going, report at the end
        # -- one malformed row in 1.5 billion should not kill a 100-task array,
        # but it must never pass unnoticed either.
        bad.append(row)
        return "skip"

    stream, proc = open_shard(path)
    try:
        table = pacsv.read_csv(
            stream,
            read_options=pacsv.ReadOptions(
                column_names=BED_COLUMNS, use_threads=True, block_size=block_size
            ),
            parse_options=pacsv.ParseOptions(
                delimiter="\t", invalid_row_handler=on_invalid
            ),
            convert_options=pacsv.ConvertOptions(
                column_types={
                    # Both string columns are low-cardinality relative to the row
                    # count, so dictionary encoding shrinks them and, for srx,
                    # collapses the lookup. See the module docstring.
                    "chrom": pa.dictionary(pa.int32(), pa.string()),
                    "srx": pa.dictionary(pa.int32(), pa.string()),
                    "start": pa.int32(),
                    "end": pa.int32(),
                    "score": pa.int32(),
                },
            ),
        )
    finally:
        stream.close()
        if proc is not None and proc.wait() != 0:
            raise SystemExit("decompressor failed on %s" % path)
    return table, len(bad)


def group_ids_for(table: pa.Table, lookup: dict) -> pa.Array:
    """Per-row group id as an Arrow Int32Array, -1 where the accession is unknown.

    This is where dictionary encoding pays off. Each chunk's ``dictionary`` holds
    only the *distinct* accessions in that chunk -- tens of thousands, against
    millions of rows -- so the Python-level dict lookup runs once per distinct
    accession, and the per-row expansion is a single Arrow ``take``.

    Chunks are handled one at a time because each carries its own dictionary;
    unifying them first would cost more than the per-chunk lookup does.
    """
    out = []
    for chunk in table.column("srx").chunks:
        values = chunk.dictionary.to_pylist()
        lut = pa.array([lookup.get(v, -1) for v in values], type=pa.int32())
        indices = chunk.indices
        if indices.null_count:
            # Route nulls at a sentinel slot that maps to -1, i.e. unmapped.
            lut = pa.concat_arrays([lut, pa.array([-1], type=pa.int32())])
            indices = indices.fill_null(len(values))
        out.append(lut.take(indices))
    if not out:
        return pa.array([], type=pa.int32())
    return pa.concat_arrays(out)


def route_shard(root: Path, org: str, shard: Path, n_buckets: int,
                block_size: int, level: int) -> dict:
    started = time.time()
    lookup = load_lookup(root, org)
    table, n_bad = read_shard(shard, block_size)
    n_rows = table.num_rows

    gid_arrow = group_ids_for(table, lookup)
    mask = pc.greater_equal(gid_arrow, 0)
    # pc.sum over an empty array is null, not 0.
    n_mapped = pc.sum(pc.cast(mask, pa.int64())).as_py() or 0
    n_unmapped = n_rows - n_mapped
    if n_unmapped:
        # An accession the manifest never saw -- a peak file newer than the meta
        # dump, usually. Dropped, but counted and reported, never silently.
        table = table.filter(mask)
        gid_arrow = gid_arrow.filter(mask)

    table = table.append_column("group_id", gid_arrow)
    gid = compat.to_numpy(gid_arrow, np.int32)
    bucket = (gid % n_buckets).astype(np.int32)

    # Stable, so rows keep the genomic order they arrived in -- the property the
    # whole no-sort design rests on. numpy uses a radix sort for integer keys,
    # so this is linear rather than n log n.
    order = np.argsort(bucket, kind="stable").astype(np.int64)
    table = table.take(compat.to_arrow(order, pa.int64()))
    edges = np.searchsorted(bucket[order], np.arange(n_buckets + 1))

    tag = shard.name.split(".")[0]           # shard_0007
    written = 0
    for b in range(n_buckets):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        out_dir = root / "work" / "parts" / org / ("bucket_%04d" % b)
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table.slice(lo, hi - lo), out_dir / ("%s.parquet" % tag),
            compression="zstd", compression_level=level,
        )
        written += 1

    stats = {
        "shard": shard.name, "rows": n_rows, "unmapped": n_unmapped,
        "bad_rows": n_bad, "buckets_written": written,
        "seconds": round(time.time() - started, 1),
    }
    stats_dir = root / "work" / "stats" / org
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / ("%s.json" % tag)).write_text(json.dumps(stats))
    print("%s  %d rows  %d unmapped  %d bad  %d buckets  %.1fs"
          % (tag, n_rows, n_unmapped, n_bad, written, stats["seconds"]), flush=True)
    return stats


def resolve_shards(root: Path, org: str, selector: str) -> list:
    shard_dir = root / "work" / "shards" / org
    shards = sorted(p for p in shard_dir.glob("shard_*") if p.suffix != ".json")
    if not shards:
        raise SystemExit("no shards in %s -- run stage 1 first" % shard_dir)
    if selector == "all":
        return shards
    if selector == "env":
        index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if index is None:
            raise SystemExit("--shard env needs SLURM_ARRAY_TASK_ID")
        selector = index
    index = int(selector)
    if index >= len(shards):
        # A no-op, not an error. The shard count is only known after stage 1, so
        # run_all.sh submits a generous array and the tail lands here; failing
        # would paint the whole array red for a scheduling detail.
        print("shard %d: nothing to do (%d shards exist)" % (index, len(shards)))
        return []
    if index < 0:
        raise SystemExit("negative shard index %d" % index)
    return [shards[index]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--org", required=True)
    parser.add_argument("--shard", default="env",
                        help="shard index, 'all', or 'env' for SLURM_ARRAY_TASK_ID")
    parser.add_argument("--buckets", type=int, default=128,
                        help="stage-3 array width; a group lands wholly in one")
    parser.add_argument("--block-size", type=int, default=64 << 20,
                        help="pyarrow CSV block size, bytes")
    parser.add_argument("--level", type=int, default=3, help="parquet zstd level")
    args = parser.parse_args(argv)

    for shard in resolve_shards(args.root, args.org, args.shard):
        route_shard(args.root, args.org, shard, args.buckets,
                    args.block_size, args.level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
