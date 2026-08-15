"""Stage 3 -- reduce. One SLURM array task per bucket; writes the final BEDs.

A bucket holds every peak of every group that hashed to it, spread across one
parquet part per shard. Because ``group_id % n_buckets`` sent a group's rows all
to the same bucket, this task can finish those groups completely: no other task
writes to the files it writes, so there is no merge afterwards.

**No sort anywhere.** The input .gz is sorted by (chrom, start); shard.py cut it
without reordering; route.py's stable sort preserved order within each bucket.
So reading parts in shard-index order and appending gives a correctly sorted
BED. Sorting ~1.5 billion rows would otherwise dominate the entire pipeline.

**Bounded memory under heavy skew.** Group sizes are wildly uneven -- ATAC-Seq
in Blood is orders of magnitude larger than most transcription factors. Parts
are therefore streamed one at a time and appended to open handles, never
concatenated, so peak memory tracks the largest *part* (tens of MB) rather than
the largest group (which can be gigabytes).

Usage:
    python -m chipatlas_forge.collect --root . --org hg38 --bucket 12
    python -m chipatlas_forge.collect --root . --org hg38 --bucket all
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from . import arrow_compat as compat

BED_COLUMNS = ["chrom", "start", "end", "srx", "score"]

WRITE_OPTIONS = pacsv.WriteOptions(
    include_header=False, delimiter="\t",
    # The five BED fields never contain a tab or a quote, and downstream readers
    # (bedtools, awk, pandas) treat a quote as literal -- so quoting would
    # corrupt the file rather than protect it.
    quoting_style="none",
)


class GroupWriter:
    """One output BED, opened on first write, optionally piped through gzip."""

    def __init__(self, path: Path, compress: str):
        self.path = path
        self.compress = compress
        self.rows = 0
        self._sink = None
        self._proc = None

    def _ensure(self):
        if self._sink is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.compress == "none":
            self._sink = open(self.path, "wb")
        else:
            target = open(self.path, "wb")
            tool = "pigz" if _has("pigz") else "gzip"
            self._proc = subprocess.Popen([tool, "-c", "-3"],
                                          stdin=subprocess.PIPE, stdout=target)
            target.close()
            self._sink = self._proc.stdin

    def write(self, table):
        self._ensure()
        pacsv.write_csv(table, self._sink, WRITE_OPTIONS)
        self.rows += table.num_rows

    def close(self):
        if self._sink is None:
            return
        self._sink.close()
        if self._proc is not None and self._proc.wait() != 0:
            raise SystemExit("compressor failed for %s" % self.path)


def _has(tool: str) -> bool:
    from shutil import which
    return which(tool) is not None


def collect_bucket(root: Path, org: str, bucket: int, n_buckets: int,
                   compress: str) -> dict:
    started = time.time()
    groups = pd.read_csv(root / "work" / "manifest" / org / "groups.tsv", sep="\t")
    mine = groups[groups["group_id"] % n_buckets == bucket]

    # groups.tsv already carries a path ending in .bed; gzip only adds a suffix.
    extra = "" if compress == "none" else ".gz"
    out_root = root / "out"
    writers = {
        int(row.group_id): GroupWriter(out_root / (row.path + extra), compress)
        for row in mine.itertuples()
    }

    part_dir = root / "work" / "parts" / org / ("bucket_%04d" % bucket)
    parts = sorted(part_dir.glob("shard_*.parquet")) if part_dir.exists() else []

    n_rows = 0
    for part in parts:                        # shard order == genomic order
        table = pq.read_table(part)
        if table.num_rows == 0:
            continue
        gid = compat.to_numpy(table.column("group_id"), np.int32)
        # Stable, so the genomic order inside each group survives the regrouping.
        order = np.argsort(gid, kind="stable").astype(np.int64)
        table = (table.take(compat.to_arrow(order, pa.int64()))
                      .drop_columns(["group_id"]).select(BED_COLUMNS))
        sorted_gid = gid[order]

        # Contiguous run per group: one slice and one C++ CSV write each, rather
        # than a boolean filter per group over the whole part.
        edges = np.flatnonzero(np.diff(sorted_gid)) + 1
        for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(sorted_gid)]):
            writer = writers.get(int(sorted_gid[lo]))
            if writer is None:
                raise SystemExit(
                    "part %s carries group %d, which does not hash to bucket %d "
                    "-- stage 2 and stage 3 disagree on --buckets"
                    % (part.name, int(sorted_gid[lo]), bucket)
                )
            writer.write(table.slice(int(lo), int(hi - lo)))
        n_rows += len(sorted_gid)

    for writer in writers.values():
        writer.close()

    written = {gid: w.rows for gid, w in writers.items() if w.rows}
    stats = {
        "bucket": bucket, "groups_assigned": len(writers),
        "groups_written": len(written), "parts": len(parts), "rows": n_rows,
        "seconds": round(time.time() - started, 1),
    }
    stats_dir = root / "work" / "collect_stats" / org
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / ("bucket_%04d.json" % bucket)).write_text(json.dumps(stats))
    print("bucket %4d  %d/%d groups  %d parts  %d rows  %.1fs"
          % (bucket, len(written), len(writers), len(parts), n_rows,
             stats["seconds"]), flush=True)
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--org", required=True)
    parser.add_argument("--bucket", default="env",
                        help="bucket index, 'all', or 'env' for SLURM_ARRAY_TASK_ID")
    parser.add_argument("--buckets", type=int, default=128,
                        help="MUST match the --buckets stage 2 ran with")
    parser.add_argument("--compress", choices=["none", "gzip"], default="none")
    args = parser.parse_args(argv)

    if args.bucket == "all":
        targets = range(args.buckets)
    elif args.bucket == "env":
        index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if index is None:
            raise SystemExit("--bucket env needs SLURM_ARRAY_TASK_ID")
        targets = [int(index)]
    else:
        targets = [int(args.bucket)]

    targets = [b for b in targets if b < args.buckets]
    if not targets:
        print("nothing to do: bucket index is at or past --buckets %d" % args.buckets)
        return 0

    for bucket in targets:
        collect_bucket(args.root, args.org, bucket, args.buckets, args.compress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
