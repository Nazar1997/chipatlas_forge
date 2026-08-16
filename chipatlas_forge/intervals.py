"""Stage 8 -- the chunk grid, the training windows, and the train/val/test split.

Two grids, and they are not the same thing:

*The chunk grid* cuts every chromosome into fixed 64 kb pieces. It is a storage
decision -- every per-chunk DNA file and every omics row group is keyed by it --
and nothing about training depends on its size.

*Windows* are training examples: 8192, 65536 and 2**20 bases. A window is
materialised once per (strand, tissue), because that is what an example is --
the same locus read on the minus strand against a different tissue's omics is a
different sample. hg38 at 8192 comes to ~14 M rows.

Windows are dropped when more than half the bases are N (unassembled sequence
teaches nothing) or when they touch an ENCODE blacklist region at all. Both
tests are exact and computed against *runs* rather than per-base arrays: N
stretches and blacklist regions are each a few hundred intervals per
chromosome, so overlap is a searchsorted against a prefix sum instead of a
1 GB cumulative array for chr1.

**The split is assigned on 2**20 blocks and inherited downward.** Assigning it
per window independently would put an 8192 window in train while the 65536
window containing it is in test -- the same bases on both sides of the split.
Because 8192 and 65536 both divide 2**20, a block-level assignment nests
exactly: every window sits inside one block and takes its label, so all three
window sizes agree about which stretches of genome are held out.

Validation is whole chromosomes (chr8 + chr9, 9.18% of hg38 and 9.32% of mm10),
so no validation window shares a regulatory neighbourhood with a training one.
Test is 10% of the blocks that remain, drawn at the block level rather than by
chromosome: validation already costs a tenth of the genome and spending two
more whole chromosomes would take a fifth of the training signal.

Usage:
    python -m chipatlas_forge.intervals --data-dir ../ --org hg38 --release 2026-08
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import arrow_compat as compat
from . import layout

# The grid the split is decided on. Equal to the largest window so that window
# never straddles two labels, and a multiple of every smaller one so they nest.
SPLIT_BLOCK = 1 << 20

STRANDS = ("+", "-")


def read_runs(path, chrom_filter=None):
    """Blacklist BED as ``{chrom: (starts, ends)}``, sorted and merged."""
    by_chrom = {}
    if path is None or not Path(path).exists():
        return by_chrom
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if chrom_filter and chrom not in chrom_filter:
                continue
            by_chrom.setdefault(chrom, []).append((int(parts[1]), int(parts[2])))
    return {c: merge_runs(np.array([s for s, _ in v], dtype=np.int64),
                          np.array([e for _, e in v], dtype=np.int64))
            for c, v in by_chrom.items()}


def merge_runs(starts, ends):
    """Sort and union overlapping intervals, so overlap arithmetic can assume
    they are disjoint (otherwise a doubly-covered base is counted twice)."""
    if len(starts) == 0:
        return starts, ends
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    keep_s, keep_e = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:].tolist(), ends[1:].tolist()):
        if s <= keep_e[-1]:
            keep_e[-1] = max(keep_e[-1], e)
        else:
            keep_s.append(s)
            keep_e.append(e)
    return np.array(keep_s, dtype=np.int64), np.array(keep_e, dtype=np.int64)


def n_runs(sequence):
    """Contiguous stretches of N in a chromosome, as ``(starts, ends)``.

    hg38's unassembled sequence is a few hundred large blocks per chromosome
    (telomeres, centromere, gaps), so the run form is three orders of magnitude
    smaller than the per-base mask and makes every window's N count a couple of
    searchsorted lookups.
    """
    arr = np.frombuffer(sequence.encode("ascii", "ignore"), dtype=np.uint8)
    is_n = (arr == 0x4E) | (arr == 0x6E)          # 'N', 'n'
    if not is_n.any():
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    # Pad both sides so a run touching either end is still bounded by an edge.
    edges = np.flatnonzero(np.diff(np.concatenate(([0], is_n.view(np.uint8), [0]))))
    return edges[0::2].astype(np.int64), edges[1::2].astype(np.int64)


def overlap_bp(q_start, q_end, r_start, r_end):
    """Bases of each query window covered by the (disjoint, sorted) runs.

    ``C(x)`` is the number of run bases strictly before ``x``; the answer is
    ``C(end) - C(start)``. Computing ``C`` from a prefix sum makes this O(log r)
    per window with no array the size of the chromosome.
    """
    if len(r_start) == 0:
        return np.zeros(len(q_start), dtype=np.int64)
    cum = np.concatenate(([0], np.cumsum(r_end - r_start)))

    def covered_before(x):
        i = np.searchsorted(r_start, x, side="right")
        total = cum[i]
        # Run i-1 may straddle x; remove the part at or after it.
        prev = np.maximum(i - 1, 0)
        excess = np.where(i > 0, np.maximum(r_end[prev] - x, 0), 0)
        return total - excess

    return covered_before(q_end) - covered_before(q_start)


def window_starts(length, window):
    """Aligned window starts, with the last one pulled back to stay full width.

    A chromosome is rarely a whole number of windows, and a short final window
    would be a differently-shaped training example. Shifting it back to
    ``length - window`` overlaps its predecessor, which is the original
    behaviour and is preferable to either padding or discarding the tail.
    """
    if length < window:
        return np.zeros(1, dtype=np.int64)
    starts = np.arange(0, length, window, dtype=np.int64)
    starts = np.minimum(starts, length - window)
    return np.unique(starts)


def assign_blocks(chrom_sizes, val_chroms, test_fraction, seed):
    """Label every 2**20 block train/val/test. The one place the split is decided.

    Returns ``{chrom: array of labels}``, one entry per block, indexed by
    ``start // SPLIT_BLOCK``.
    """
    counts = {c: int(n // SPLIT_BLOCK) + 1 for c, n in chrom_sizes.items()}
    labels = {c: np.full(n, "train", dtype="<U5") for c, n in counts.items()}

    pool = []
    for chrom in sorted(counts):
        if chrom in val_chroms:
            labels[chrom][:] = "val"
        else:
            pool.extend((chrom, i) for i in range(counts[chrom]))

    n_test = int(round(len(pool) * test_fraction))
    if n_test:
        rng = np.random.default_rng(seed)
        for pick in rng.permutation(len(pool))[:n_test]:
            chrom, i = pool[pick]
            labels[chrom][i] = "test"
    return labels


def label_windows(chrom, starts, ends, blocks):
    """Each window's split, from the block holding its midpoint.

    The midpoint rather than the start: the final window of a chromosome is
    pulled back off the grid and can straddle two blocks, and the midpoint puts
    it with whichever block covers most of it. For every aligned window the
    midpoint is inside its own block, so this is the same as indexing by start.
    """
    mid = (starts + ends) // 2
    index = np.minimum(mid // SPLIT_BLOCK, len(blocks[chrom]) - 1)
    return blocks[chrom][index]


def build_windows(window, chrom_sizes, sequences, blacklist, blocks,
                  tissues, max_n_fraction):
    """Every training example at one window size, as an Arrow table."""
    chrom_col, start_col, end_col, split_col = [], [], [], []
    kept = dropped_n = dropped_bl = 0

    for chrom in sorted(chrom_sizes, key=lambda c: -chrom_sizes[c]):
        length = chrom_sizes[chrom]
        starts = window_starts(length, window)
        ends = np.minimum(starts + window, length)

        n_bp = overlap_bp(starts, ends, *sequences[chrom])
        bl_bp = overlap_bp(starts, ends, *blacklist.get(
            chrom, (np.empty(0, np.int64), np.empty(0, np.int64))))

        too_many_n = n_bp > (ends - starts) * max_n_fraction
        blacklisted = bl_bp > 0
        keep = ~(too_many_n | blacklisted)
        dropped_n += int(too_many_n.sum())
        dropped_bl += int((blacklisted & ~too_many_n).sum())

        starts, ends = starts[keep], ends[keep]
        if not len(starts):
            continue
        kept += len(starts)
        chrom_col.append(np.full(len(starts), chrom))
        start_col.append(starts)
        end_col.append(ends)
        split_col.append(label_windows(chrom, starts, ends, blocks))

    if not start_col:
        raise SystemExit("every window at %d bp was filtered out" % window)

    chroms = np.concatenate(chrom_col)
    starts = np.concatenate(start_col)
    ends = np.concatenate(end_col)
    splits = np.concatenate(split_col)

    # Materialise (window x strand x tissue). Tiled rather than looped so the
    # 14 M rows are built by numpy, and dictionary-encoded so the repeated
    # chromosome, strand, tissue and split strings cost one entry each rather
    # than one per row.
    n_win, n_rep = len(starts), len(STRANDS) * len(tissues)
    strand_of = np.repeat(np.arange(len(STRANDS)), len(tissues))
    tissue_of = np.tile(np.arange(len(tissues)), len(STRANDS))

    table = pa.table({
        "chrom": _dict_column(np.repeat(chroms, n_rep)),
        "start": compat.to_arrow(np.repeat(starts, n_rep), pa.int64()),
        "end": compat.to_arrow(np.repeat(ends, n_rep), pa.int64()),
        "strand": _dict_from_codes(np.tile(strand_of, n_win), list(STRANDS)),
        "tissue": _dict_from_codes(np.tile(tissue_of, n_win), list(tissues)),
        "split": _dict_column(np.repeat(splits, n_rep)),
    })
    stats = {"windows": kept, "rows": len(table),
             "dropped_n": dropped_n, "dropped_blacklist": dropped_bl}
    return table, stats


def split_counts(table):
    """Rows per split, counted on the dictionary codes rather than the values.

    The column is 14 M dictionary-encoded rows over three distinct strings, so
    comparing codes is a pass over int32s; materialising the strings first would
    build a 14 M-element Python list to answer three questions.
    """
    column = table.column("split").combine_chunks()
    names = column.dictionary.to_pylist()
    codes = compat.to_numpy(column.indices, np.int32)
    return {name: int(np.count_nonzero(codes == names.index(name)))
            if name in names else 0
            for name in ("train", "val", "test")}


def _dict_column(values):
    """Dictionary-encode a numpy array of strings."""
    uniques, codes = np.unique(values, return_inverse=True)
    return _dict_from_codes(codes.astype(np.int32), [str(u) for u in uniques])


def _dict_from_codes(codes, names):
    return pa.DictionaryArray.from_arrays(
        compat.to_arrow(np.ascontiguousarray(codes, dtype=np.int32), pa.int32()),
        pa.array(names))  # abi-ok: names is a Python list of str


def build_chunk_grid(chrom_sizes, chunk_size):
    """The storage grid: every 64 kb piece of every chromosome."""
    chrom_col, start_col, end_col = [], [], []
    for chrom in sorted(chrom_sizes):
        length = chrom_sizes[chrom]
        starts = np.arange(0, length, chunk_size, dtype=np.int64)
        chrom_col.append(np.full(len(starts), chrom))
        start_col.append(starts)
        end_col.append(np.minimum(starts + chunk_size, length))
    return pa.table({
        "chrom": _dict_column(np.concatenate(chrom_col)),
        "start": compat.to_arrow(np.concatenate(start_col), pa.int64()),
        "end": compat.to_arrow(np.concatenate(end_col), pa.int64()),
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--org", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--windows", type=int, nargs="+",
                        default=list(layout.DEFAULT_WINDOWS))
    parser.add_argument("--val-chroms", nargs="*",
                        default=list(layout.DEFAULT_VAL_CHROMS))
    parser.add_argument("--test-fraction", type=float,
                        default=layout.DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=layout.DEFAULT_SPLIT_SEED)
    parser.add_argument("--max-n-fraction", type=float, default=0.5,
                        help="drop a window whose bases are more than this "
                             "fraction unassembled")
    args = parser.parse_args(argv)

    release = layout.Release.open(args.data_dir, args.org, args.release)
    release.require("genome", "vocab")

    chrom_sizes = {c: int(n) for c, n in release.manifest["chrom_sizes"].items()}
    tissues = [t["name"] for t in
               json.loads(release.path("tissues").read_text())["tissues"]]

    bad = [c for c in args.val_chroms if c not in chrom_sizes]
    if bad:
        raise SystemExit("--val-chroms names %s, which %s does not have"
                         % (", ".join(bad), args.org))

    print("loading %s" % release.path("sequence"), flush=True)
    with open(release.path("sequence"), "rb") as fh:
        raw = pickle.load(fh)
    sequences = {c: n_runs(raw[c]) for c in chrom_sizes if c in raw}
    missing = sorted(set(chrom_sizes) - set(sequences))
    if missing:
        raise SystemExit("sequence.pkl has no entry for %s" % ", ".join(missing))
    del raw

    blacklist = read_runs(release.path("blacklist"), set(chrom_sizes))
    blocks = assign_blocks(chrom_sizes, set(args.val_chroms),
                           args.test_fraction, args.seed)

    held = sum(int((v == "val").sum()) for v in blocks.values())
    tested = sum(int((v == "test").sum()) for v in blocks.values())
    total = sum(len(v) for v in blocks.values())
    print("split over %d blocks of %d bp: %.1f%% val (%s), %.1f%% test"
          % (total, SPLIT_BLOCK, 100 * held / total, ", ".join(args.val_chroms),
             100 * tested / total), flush=True)

    index_dir = release.path("chunk_grid").parent
    index_dir.mkdir(parents=True, exist_ok=True)
    grid = build_chunk_grid(chrom_sizes, release.chunk_size)
    pq.write_table(grid, release.path("chunk_grid"), compression="zstd")
    print("chunk grid: %d chunks of %d bp" % (len(grid), release.chunk_size),
          flush=True)

    per_window = {}
    for window in sorted(args.windows):
        table, stats = build_windows(window, chrom_sizes, sequences,
                                     blacklist, blocks, tissues,
                                     args.max_n_fraction)
        pq.write_table(table, release.windows(window), compression="zstd")
        counts = split_counts(table)
        stats["by_split"] = counts
        per_window[str(window)] = stats
        print("  %8d bp: %d windows -> %d rows  (train %d / val %d / test %d);"
              " dropped %d N-rich, %d blacklisted"
              % (window, stats["windows"], stats["rows"], counts["train"],
                 counts["val"], counts["test"], stats["dropped_n"],
                 stats["dropped_blacklist"]), flush=True)

    release.manifest["split"] = {
        "val_chroms": list(args.val_chroms),
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "block_bp": SPLIT_BLOCK,
    }
    release.record("intervals", chunks=len(grid), windows=per_window,
                   max_n_fraction=args.max_n_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
