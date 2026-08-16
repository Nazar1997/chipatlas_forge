"""Stage 5c -- the pan-tissue track: every antigen, maximum over all cell types.

ChIP-Atlas's cell-type classes are real tissues, so grouping per-experiment
metadata can never produce an "all cell types" bucket -- and the 2021 tree had
one, because it took ChIP-Atlas's own precomputed `*.AllCell.bed` aggregates
rather than deriving anything. Without it a release is missing the tissue that
`only_one_tissue` runs train on and that the pair builder profiles against.

Merging is exact rather than approximate. Each per-tissue track is already the
maximum score over the peaks in that tissue, so the maximum *across* tissues is
the maximum over every peak of that antigen anywhere -- identical to what
`binmax` would produce from the union of the peaks, without going back to them.
The same compressed-breakpoint routine does the merge.

This is strictly better than what it replaces. The 2021 pan-tissue track was
**not** a superset: on hg38 it carried ~1004 antigens but none of H3K27ac /
H3K4me1 / H3K4me3 / H3K27me3 / RNA polymerase II, because ChIP-Atlas publishes
those only per-tissue -- so pairing on it had no view of the marks that define
enhancers and promoters. Derived here, it covers every antigen that has data in
any tissue.

Usage:
    python -m chipatlas_forge.pantissue --data-dir ../ --org hg38 \\
        --release 2026-08 --task 3 --tasks 20
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from . import arrow_compat as compat
from . import layout
from .binmax import WRITE_OPTIONS, _chrom_column, max_runs
from .chunks import BEDGRAPH_COLUMNS, read_bedgraph

# ChIP-Atlas's own name and code for the bucket, so a release spells it the same
# way it spells every other cell-type class and `tissue_by_code(org, "ALL")`
# resolves it.
PAN_TISSUE = "All cell types"
PAN_CODE = "ALL"


def plan(release, pan_tissue=PAN_TISSUE):
    """``{(ag class, antigen): [source track, ...]}``, largest total first.

    Grouped by antigen class as well as antigen because that is where the output
    goes, and an antigen belongs to exactly one class in practice. Sorted by
    total input size so an array of tasks does not put every large antigen in
    one.
    """
    root = release.path("signal_root")
    if not root.is_dir():
        raise SystemExit("%s does not exist -- run adopt first" % root)
    groups = defaultdict(list)
    for ag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for tissue_dir in sorted(p for p in ag_dir.iterdir() if p.is_dir()):
            if tissue_dir.name == pan_tissue:
                continue          # never merge a previous run's own output
            for track in sorted(tissue_dir.glob("*.bedgraph")):
                groups[(ag_dir.name, track.stem)].append(track)
    ordered = sorted(groups.items(),
                     key=lambda kv: -sum(p.stat().st_size for p in kv[1]))
    return ordered


def merge_tracks(sources, target, chroms, block_size):
    """Max-merge several bedGraphs of one antigen into one.

    One chromosome resident at a time: the inputs are grouped by chromosome and
    a single antigen across twenty tissues is still only that chromosome's
    worth of runs.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    per_chrom = defaultdict(list)
    for source in sources:
        for chrom, arrays in read_bedgraph(source, chroms, block_size).items():
            per_chrom[chrom].append(arrays)

    n_runs = 0
    with open(target, "wb") as sink:
        # Sorted so the output keeps the genomic ordering every later stage
        # relies on; `chroms` is a dict and its order is not meaningful.
        for chrom in sorted(per_chrom, key=lambda c: (-chroms[c], c)):
            parts = per_chrom[chrom]
            starts = np.concatenate([p[0] for p in parts])
            ends = np.concatenate([p[1] for p in parts])
            values = np.concatenate([p[2] for p in parts])
            run_start, run_end, run_value = max_runs(starts, ends, values, 1)
            if not len(run_start):
                continue
            table = pa.table({
                "chrom": _chrom_column(chrom, len(run_start)),
                "start": compat.to_arrow(run_start.astype(np.int64), pa.int64()),
                "end": compat.to_arrow(run_end.astype(np.int64), pa.int64()),
                "value": compat.to_arrow(run_value, pa.int32()),
            })
            pacsv.write_csv(table, sink, WRITE_OPTIONS)
            n_runs += len(run_start)
    return {"runs": n_runs, "sources": len(sources),
            "bytes": target.stat().st_size if target.exists() else 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--org", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--pan-tissue", default=PAN_TISSUE)
    parser.add_argument("--task", default="env",
                        help="task index, 'all', or 'env' for SLURM_ARRAY_TASK_ID")
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=64 << 20)
    args = parser.parse_args(argv)

    release = layout.Release.open(args.data_dir, args.org, args.release)
    release.require("genome")
    chroms = {c: int(n) for c, n in release.manifest["chrom_sizes"].items()}

    work = plan(release, args.pan_tissue)
    if args.task == "all":
        mine, index = work, 0
    else:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID") if args.task == "env" else args.task
        if raw is None:
            raise SystemExit("--task env needs SLURM_ARRAY_TASK_ID")
        index = int(raw)
        if index >= args.tasks:
            print("task %d: nothing to do (%d tasks)" % (index, args.tasks))
            return 0
        # Strided over a size-sorted list, so the few very large antigens are
        # dealt to different tasks instead of piling into one.
        mine = work[index::args.tasks]

    print("task %d/%d: %d antigens, %d source tracks"
          % (index, args.tasks, len(mine), sum(len(v) for _, v in mine)),
          flush=True)

    started = time.time()
    totals = {"antigens": 0, "runs": 0, "bytes": 0}
    for (ag_class, antigen), sources in mine:
        target = release.signal_path(ag_class, args.pan_tissue, antigen)
        stats = merge_tracks(sources, target, chroms, args.block_size)
        totals["antigens"] += 1
        totals["runs"] += stats["runs"]
        totals["bytes"] += stats["bytes"]

    totals["seconds"] = round(time.time() - started, 1)
    stats_dir = release.root / "work" / "pantissue"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / ("task_%04d.json" % index)).write_text(json.dumps(totals))
    print("done: %(antigens)d antigens -> %(runs)d runs, %(bytes)d bytes  "
          "%(seconds)ss" % totals, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
