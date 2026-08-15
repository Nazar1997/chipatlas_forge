"""Stage 5 -- collapse overlapping peaks into a max-score signal track.

The BEDs out of `collect` are raw ChIP-Atlas peaks, so they overlap heavily --
every experiment that called a peak at a locus contributes its own row:

    chr1  9915  10410  SRX24935670  644
    chr1  9919  10288  SRX15914382  377
    chr1  9921  10441  SRX24935673  643

That is a pile of intervals, not a signal. This stage turns it into one: for
every bin, the maximum score over all peaks covering that bin, written as a
4-column bedGraph of constant-value runs.

**Compressed breakpoint space, not a dense array.** The obvious implementation
allocates one slot per base (249M for chr1) and paints each peak across its
~570 bases; that is ~2.2 *trillion* writes over the full dataset. But the max is
piecewise constant and can only change where some peak starts or ends, so the
whole function is determined by at most 2n breakpoints. Working in that
compressed space is exact at 1bp -- nothing is approximated -- while memory
becomes O(peaks) instead of O(genome length), and each peak's write spans its
local overlap *depth* (tens) rather than its length in bases (hundreds). It also
makes the output run-length encoded for free, since compressed segments are
exactly the runs.

**Ascending score, plain assignment.** Painting in increasing score order means
a later write always carries a higher score, so the last writer wins and plain
slice assignment computes the maximum -- no np.maximum, no read-modify-write.

Runs of equal value are merged and zero-coverage gaps dropped, so the output is
the minimal exact representation.

Usage:
    python -m chipatlas_forge.binmax --root . --org hg38 --bin-size 1
    python -m chipatlas_forge.binmax --root . --org hg38 --task 3 --tasks 100
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from . import arrow_compat as compat

BED_COLUMNS = ["chrom", "start", "end", "srx", "score"]

WRITE_OPTIONS = pacsv.WriteOptions(
    include_header=False, delimiter="\t", quoting_style="none")


def max_runs(starts, ends, scores, bin_size=1):
    """Constant-value runs of ``max score over covering peaks``.

    Returns ``(run_start, run_end, run_value)`` in bin units, covering only
    positions where at least one peak lies. Inputs need not be sorted.

    With ``bin_size > 1`` a bin is claimed by any peak overlapping it at all, so
    starts floor and ends ceil -- a peak never loses a bin it partly covers.
    """
    if len(starts) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.int32)

    if bin_size > 1:
        starts = starts // bin_size
        ends = -(-ends // bin_size)          # ceil division
        # A peak shorter than a bin can land start == end; keep it one bin wide.
        ends = np.maximum(ends, starts + 1)

    # Every position where the maximum can change. At most 2n of them, against
    # ~3.1e9 positions in the genome.
    breaks = np.unique(np.concatenate([starts, ends]))
    lo = np.searchsorted(breaks, starts)
    hi = np.searchsorted(breaks, ends)

    value = np.zeros(len(breaks), dtype=np.int32)

    # Ascending score, so the last write into any segment is the largest.
    order = np.argsort(scores, kind="stable")
    # .tolist() first: iterating a numpy array yields numpy scalars and indexing
    # with them is several times slower than with Python ints, and this loop runs
    # once per peak over billions of peaks.
    for a, b, v in zip(lo[order].tolist(), hi[order].tolist(),
                       scores[order].tolist()):
        value[a:b] = v

    # Segment j spans [breaks[j], breaks[j+1]) and holds value[j].
    seg_value = value[:-1]
    if len(seg_value) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.int32)

    changed = np.empty(len(seg_value), dtype=bool)
    changed[0] = True
    np.not_equal(seg_value[1:], seg_value[:-1], out=changed[1:])
    edges = np.flatnonzero(changed)

    run_value = seg_value[edges]
    run_start = breaks[edges]
    run_end = np.append(breaks[edges[1:]], breaks[-1])

    keep = run_value > 0                     # gaps between peaks are not signal
    return run_start[keep], run_end[keep], run_value[keep]


def _chrom_column(chrom: str, n: int) -> pa.Array:
    """An n-long string column of one repeated value, built without Python."""
    return pa.DictionaryArray.from_arrays(
        compat.to_arrow(np.zeros(n, dtype=np.int32), pa.int32()),
        pa.array([chrom]),
    ).cast(pa.string())


def convert_file(source: Path, target: Path, bin_size: int,
                 block_size: int) -> dict:
    """One peak BED -> one bedGraph. Streams, one chromosome resident at a time."""
    target.parent.mkdir(parents=True, exist_ok=True)
    reader = pacsv.open_csv(
        source,
        read_options=pacsv.ReadOptions(column_names=BED_COLUMNS, use_threads=True,
                                       block_size=block_size),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(column_types={
            "chrom": pa.dictionary(pa.int32(), pa.string()),
            "start": pa.int64(), "end": pa.int64(), "score": pa.int32(),
        }),
    )

    n_peaks = n_runs = 0
    pending_chrom, buf_start, buf_end, buf_score = None, [], [], []
    sink = open(target, "wb")

    def flush():
        nonlocal n_runs, buf_start, buf_end, buf_score
        if pending_chrom is None:
            return
        starts = np.concatenate(buf_start)
        ends = np.concatenate(buf_end)
        scores = np.concatenate(buf_score)
        rs, re, rv = max_runs(starts, ends, scores, bin_size)
        if len(rs):
            table = pa.table({
                "chrom": _chrom_column(pending_chrom, len(rs)),
                "start": compat.to_arrow((rs * bin_size).astype(np.int64), pa.int64()),
                "end": compat.to_arrow((re * bin_size).astype(np.int64), pa.int64()),
                "value": compat.to_arrow(rv, pa.int32()),
            })
            pacsv.write_csv(table, sink, WRITE_OPTIONS)
            n_runs += len(rs)
        buf_start, buf_end, buf_score = [], [], []

    try:
        for batch in reader:
            # The file is grouped by chromosome (collect preserves the sorted
            # input), so a chromosome is complete the moment a different one
            # appears and only one need ever be resident.
            chrom_col = batch.column(0)
            names = chrom_col.dictionary.to_pylist()
            codes = compat.to_numpy(chrom_col.indices, np.int32)
            starts = compat.to_numpy(batch.column(1), np.int64)
            ends = compat.to_numpy(batch.column(2), np.int64)
            scores = compat.to_numpy(batch.column(4), np.int32)
            n_peaks += len(codes)

            # Contiguous runs of one chromosome inside this batch.
            bounds = np.flatnonzero(np.diff(codes)) + 1
            for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(codes)]):
                name = names[codes[a]]
                if name != pending_chrom:
                    flush()
                    pending_chrom = name
                buf_start.append(starts[a:b])
                buf_end.append(ends[a:b])
                buf_score.append(scores[a:b])
        flush()
    finally:
        sink.close()
        reader.close()

    return {"peaks": n_peaks, "runs": n_runs,
            "bytes": target.stat().st_size if target.exists() else 0}


def plan(root: Path, org: str, out_name: str, suffix: str) -> list:
    """(source, target) pairs, largest first so array tasks balance."""
    src_root = root / "out" / org
    dst_root = root / out_name / org
    beds = sorted(src_root.rglob("*.bed"))
    if not beds:
        raise SystemExit("no BEDs under %s -- run collect first" % src_root)
    beds.sort(key=lambda p: p.stat().st_size, reverse=True)
    return [(p, dst_root / p.relative_to(src_root).with_suffix(suffix)) for p in beds]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--org", required=True)
    parser.add_argument("--bin-size", type=int, default=1,
                        help="1 keeps full base-pair resolution; the algorithm "
                             "cost is the same either way")
    parser.add_argument("--out-name", default=None,
                        help="output root; defaults to out_binmax<bin size>")
    parser.add_argument("--suffix", default=".bedgraph")
    parser.add_argument("--task", default="env",
                        help="task index, 'all', or 'env' for SLURM_ARRAY_TASK_ID")
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=64 << 20)
    args = parser.parse_args(argv)

    if args.bin_size < 1:
        raise SystemExit("--bin-size must be >= 1")
    out_name = args.out_name or ("out_binmax%d" % args.bin_size)

    pairs = plan(args.root, args.org, out_name, args.suffix)

    if args.task == "all":
        index, stride = 0, 1
    else:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID") if args.task == "env" else args.task
        if raw is None:
            raise SystemExit("--task env needs SLURM_ARRAY_TASK_ID")
        index, stride = int(raw), args.tasks
        if index >= stride:
            print("task %d: nothing to do (%d tasks)" % (index, stride))
            return 0

    # Strided over a size-sorted list, so the handful of multi-gigabyte files are
    # dealt to different tasks instead of piling into one.
    mine = pairs[index::stride]
    print("task %d/%d: %d files, %.1f GB in"
          % (index, stride, len(mine), sum(s.stat().st_size for s, _ in mine) / 1e9),
          flush=True)

    started = time.time()
    totals = {"files": 0, "peaks": 0, "runs": 0, "bytes": 0}
    for source, target in mine:
        stats = convert_file(source, target, args.bin_size, args.block_size)
        totals["files"] += 1
        for key in ("peaks", "runs", "bytes"):
            totals[key] += stats[key]

    totals["seconds"] = round(time.time() - started, 1)
    stats_dir = args.root / "work" / "binmax_stats" / args.org
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / ("task_%04d.json" % index)).write_text(json.dumps(totals))
    print("done: %(files)d files  %(peaks)d peaks -> %(runs)d runs  %(seconds)ss"
          % totals, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
