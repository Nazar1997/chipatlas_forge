"""Stage 10 -- IDF track weights and an omics-similarity pair index.

Moved here from the training repo (`src/data_preparation/build_omics_pairs.py`)
with the algorithm intact and the I/O re-pointed at a release. Two artifacts,
both built once and then read-only during training:

``track_weights.npz``
    ``w_t = log(1 / p_t)`` per omics track, where ``p_t`` is the fraction of
    genomic bins the track is active in. Rare tracks weigh more: sharing a
    tissue-specific TF says far more about two loci than sharing broad
    accessibility. ``log`` rather than ``1/p`` or ``1/p**2`` on purpose -- across
    a realistic popularity range (p from 0.5 down to 1e-4) those give a 5000x and
    a 25-million-fold spread, so a single shared obscure assay would decide the
    similarity on its own. ``log`` gives ~13x, which separates rare from common
    without letting one track dictate. This is plain IDF.

``omics_pairs.npz``
    For every window, its most omics-similar other windows, scored by weighted
    Jaccard over the IDF weights. Stored CSR-style (``indptr``/``indices``/
    ``scores``) so training reads window ``i``'s neighbours in O(1) with no search.

Similarity between two windows' active track sets A and B::

    J_w(A, B) = sum(w_t for t in A & B) / sum(w_t for t in A | B)

Scoring is a dense blocked matmul rather than an inverted index: there are only
about a thousand tracks, so the profile matrix is ``[n_windows, n_tracks]`` and
fits comfortably in memory. On hg38 at 8192 bp that is ~310k x 1009, and the
whole scoring pass is a BLAS GEMM -- no GPU needed, which also sidesteps the
numpy/torch ABI break in the cluster environment.

**The split now comes from the window table.** The original took `--val_chroms`
and `--test_chroms` arguments and re-derived the holdout itself, which is how it
came to disagree with `train/test_intervals_*.pkl` -- the pair index held out
chr8/chr9 while the interval files split within every chromosome. Reading the
`split` column means there is one answer to "is this window held out", written
by `intervals` and used by everything downstream.

Usage:
    python -m chipatlas_forge.pairs --data-dir ../ --org hg38 --release 2026-08
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import layout, read


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--org", required=True)
    p.add_argument("--release", default=None)
    p.add_argument("--tissues", nargs="+", default=["All cell types"],
                   help="Tissues to union into each window's profile. NOTE: the "
                        "all-cell-types track is NOT a superset -- on hg38 it "
                        "carries 1004 features but none of H3K27ac / H3K4me1 / "
                        "H3K4me3 / H3K27me3 / RNA_polymerase_II, which exist only "
                        "per-tissue. Pairing on it alone therefore has no view of "
                        "the marks that define enhancers and promoters")
    p.add_argument("--pair_bp", type=int, default=8192,
                   help="Window size pairs are defined over. Match the training context")
    p.add_argument("--chroms", nargs="*", default=None,
                   help="Restrict to these chromosomes. For smoke tests and for "
                        "sharding a full run; omit to use the whole assembly")

    p.add_argument("--min_p", type=float, default=1e-4,
                   help="Floor on track popularity. A track active in a handful of "
                        "bins would otherwise get an enormous weight from pure noise")
    p.add_argument("--min_tracks", type=int, default=2,
                   help="Windows with fewer active tracks carry no similarity signal")
    p.add_argument("--pair_threshold", type=float, default=0.5,
                   help="Weighted-Jaccard floor for a pair to be kept")
    p.add_argument("--max_pairs", type=int, default=64,
                   help="Neighbours kept per window, best first")
    p.add_argument("--disqualify_above", type=int, default=4096,
                   help="Windows with more neighbours than this above threshold are "
                        "generic -- they would be everyone's positive and teach "
                        "nothing. Dropped")
    p.add_argument("--min_separation_bp", type=int, default=1_000_000,
                   help="Neighbours closer than this on the same chromosome are "
                        "excluded: they are the same regulatory neighbourhood, not "
                        "a hard positive")

    p.add_argument("--block", type=int, default=2048, help="Query rows scored per matmul")
    p.add_argument("--seed", type=int, default=0, help="Seeds the tie-break jitter")
    return p


def window_table(release, pair_bp, chroms=None):
    """Unique windows with their split, from the release's own interval table.

    The stored table is materialised per (strand, tissue), so the same locus
    appears many times; pairs are a property of the locus, so it is collapsed
    back to one row per (chrom, start).
    """
    frame = read.windows(release, pair_bp)
    if chroms:
        frame = frame[frame["chrom"].isin(chroms)]
    table = (frame[["chrom", "start", "split"]]
             .drop_duplicates(subset=["chrom", "start"])
             .sort_values(["chrom", "start"], kind="stable")
             .reset_index(drop=True))
    table["window_id"] = np.arange(len(table), dtype=np.int64)
    return table


def build_profiles(release, tissues, features, table, pair_bp):
    """Dense ``[n_windows, n_tracks]`` uint8: is track t active anywhere in window w.

    Profiles are OR-ed across ``tissues``, so a track counts as present if any
    requested tissue calls a peak there.
    """
    feat_index = {f: i for i, f in enumerate(features)}
    profiles = np.zeros((len(table), len(features)), dtype=np.uint8)

    # window_id of the first window on each chromosome, so a genomic coordinate
    # maps to a row with one division.
    base = table.groupby("chrom", sort=False)["window_id"].min().to_dict()
    n_windows = table.groupby("chrom", sort=False).size().to_dict()
    sizes = {c: int(n) for c, n in release.manifest["chrom_sizes"].items()}
    chunk_size = release.chunk_size

    for tissue in tissues:
        for chrom in sorted(base):
            for start in range(0, sizes[chrom], chunk_size):
                df = read.load_chunk(release, tissue, chrom, start,
                                     min(start + chunk_size, sizes[chrom]))
                if df.shape[0] == 0:
                    continue
                feat = df["feature_name"].map(feat_index)
                keep = feat.notna()
                if not keep.any():
                    continue
                feat = feat[keep].to_numpy(dtype=np.int64)
                starts = df.loc[keep, "Start"].to_numpy(dtype=np.int64)
                ends = df.loc[keep, "End"].to_numpy(dtype=np.int64)

                # A peak marks every window it overlaps, not just the one it
                # starts in.
                first = starts // pair_bp
                last = (ends - 1) // pair_bp
                span = (last - first + 1).clip(min=1)
                row_local = np.repeat(first, span) + (
                    np.arange(span.sum()) - np.repeat(np.cumsum(span) - span, span)
                )
                col = np.repeat(feat, span)
                valid = (row_local >= 0) & (row_local < n_windows[chrom])
                profiles[base[chrom] + row_local[valid], col[valid]] = 1
    return profiles


def idf_weights(profiles, is_train, min_p):
    """``w_t = log(1/p_t)``, with ``p_t`` measured on training windows only."""
    active = profiles[is_train].sum(axis=0).astype(np.float64)
    p = np.clip(active / max(int(is_train.sum()), 1), min_p, 1.0)
    return np.log(1.0 / p), p


def find_pairs(profiles, weights, table, args):
    """Top-``max_pairs`` weighted-Jaccard neighbours per training window."""
    is_train = (table["split"] == "train").to_numpy()
    n_active = profiles.sum(axis=1)
    eligible = is_train & (n_active >= args.min_tracks)
    print(f"  eligible windows: {eligible.sum():,} of {len(table):,} "
          f"({(~is_train).sum():,} held out, "
          f"{(is_train & (n_active < args.min_tracks)).sum():,} too sparse)")

    idx = np.flatnonzero(eligible)
    Y = profiles[idx].astype(np.float32)         # [n, T]
    w = weights.astype(np.float32)
    row_w = Y @ w                                # weighted |A| per window

    # Windows are ordered by (chrom, start), so "same chromosome and within
    # min_separation" is a CONTIGUOUS range of rows. Zeroing that slice per row
    # avoids building two [block, n] boolean masks, which at hg38 scale would be
    # 775 MB each on top of the score matrix.
    chrom_code = pd.factorize(table["chrom"].to_numpy())[0][idx].astype(np.int64)
    key = chrom_code * (1 << 40) + table["start"].to_numpy()[idx]
    band_lo = np.searchsorted(key, key - args.min_separation_bp, side="left")
    band_hi = np.searchsorted(key, key + args.min_separation_bp, side="right")

    n = len(idx)
    per_block_gb = args.block * n * 4 * 2 / 1024 ** 3
    print(f"  scoring {n:,} x {n:,} in blocks of {args.block} "
          f"(~{per_block_gb:.1f} GB per block)")

    rng = np.random.default_rng(args.seed)
    all_idx, all_scores, all_counts = [], [], []
    t0 = time.time()
    for lo in range(0, n, args.block):
        hi = min(lo + args.block, n)
        # Weighted intersection. numpy hands this straight to BLAS; Y.T is a
        # view, so no copy of the [n, T] matrix is made.
        inter = (Y[lo:hi] * w) @ Y.T                              # [q, n]
        # In place: denom = |A| + |B| - inter, then jac = inter / denom. Two big
        # arrays live at once instead of three.
        denom = row_w[lo:hi, None] + row_w[None, :]
        denom -= inter
        np.maximum(denom, 1e-9, out=denom)
        jac = np.divide(inter, denom, out=inter)
        del denom

        # A window is not its own neighbour, and one a few hundred kb away is
        # the same regulatory neighbourhood rather than an independent example.
        for r in range(hi - lo):
            jac[r, band_lo[lo + r]:band_hi[lo + r]] = 0.0

        above = (jac >= args.pair_threshold).sum(axis=1)
        k = min(args.max_pairs, jac.shape[1])
        # Break ties at random. Exact ties are common -- two windows with the
        # same active track set score exactly 1.0 -- and argpartition resolves
        # them by index, so without this the lowest-numbered windows become
        # everyone's partner and the rest are never sampled. The jitter is far
        # below any real score gap, and the scores stored afterwards are true.
        jittered = jac + rng.random(jac.shape, dtype=np.float32) * 1e-6
        part = np.argpartition(-jittered, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(hi - lo)[:, None]
        order = np.argsort(-jittered[rows, part], axis=1)
        top_idx = part[rows, order]
        all_idx.append(top_idx)
        all_scores.append(jac[rows, top_idx])
        all_counts.append(above)
        if lo == 0 or (lo // args.block) % 25 == 0:
            print(f"    scored {hi:,}/{n:,} ({hi / n:5.1%}) "
                  f"{time.time() - t0:6.1f}s elapsed", flush=True)

    top_idx = np.concatenate(all_idx)
    top_scores = np.concatenate(all_scores)
    counts = np.concatenate(all_counts)

    # "Too many pairs" means generic: a window that resembles thousands of
    # others would be a positive for everyone and teaches nothing.
    generic = counts > args.disqualify_above
    print(f"  disqualified as generic (>{args.disqualify_above} matches): {generic.sum():,}")

    indptr, indices, scores = [0], [], []
    for row in range(len(idx)):
        if generic[row]:
            indptr.append(indptr[-1])
            continue
        keep = top_scores[row] >= args.pair_threshold
        indices.append(idx[top_idx[row][keep]])
        scores.append(top_scores[row][keep])
        indptr.append(indptr[-1] + int(keep.sum()))

    indices = np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64)
    scores = np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
    return (idx, np.array(indptr, dtype=np.int64), indices.astype(np.int64),
            scores.astype(np.float32))


def main(argv=None):
    args = build_parser().parse_args(argv)
    release = layout.Release.open(args.data_dir, args.org, args.release)
    release.require("intervals")

    features = read.features(release)
    print(f"org={args.org} release={release.id} tissues={','.join(args.tissues)} "
          f"pair_bp={args.pair_bp:,} tracks={len(features)}")

    known = set(read.tissues(release))
    unknown = [t for t in args.tissues if t not in known]
    if unknown:
        raise SystemExit("release %s has no tissue %s (have: %s)"
                         % (release.id, ", ".join(unknown), ", ".join(sorted(known))))

    table = window_table(release, args.pair_bp, args.chroms)
    print(f"[1/4] {len(table):,} windows "
          f"({(table['split'] == 'train').sum():,} train)")

    profiles = build_profiles(release, args.tissues, features, table, args.pair_bp)
    covered = (profiles.sum(axis=1) > 0).mean()
    print(f"[2/4] profiles built -- {covered:.1%} of windows carry at least one "
          f"track, mean {profiles.sum(axis=1).mean():.1f} tracks per window")

    is_train = (table["split"] == "train").to_numpy()
    weights, p = idf_weights(profiles, is_train, args.min_p)
    order = np.argsort(-p)
    show = list(dict.fromkeys(list(order[:3]) + list(order[-3:])))
    print(f"[3/4] IDF weights: range {weights.min():.2f}-{weights.max():.2f}  "
          f"(most and least popular tracks)")
    for i in show:
        print(f"        {features[i]:<24} p={p[i]:.5f}  w={weights[i]:.2f}")
    np.savez_compressed(release.path("track_weights"),
                        features=np.array(features, dtype=object),
                        weights=weights, popularity=p, min_p=args.min_p)

    idx, indptr, indices, scores = find_pairs(profiles, weights, table, args)
    n_src = int((np.diff(indptr) > 0).sum())
    print(f"[4/4] {len(indices):,} pairs from {n_src:,} source windows "
          f"(mean {len(indices) / max(n_src, 1):.1f} each, "
          f"mean score {scores.mean() if len(scores) else 0:.3f})")

    np.savez_compressed(
        release.path("omics_pairs"),
        window_ids=idx, indptr=indptr, indices=indices, scores=scores,
        chrom=table["chrom"].to_numpy().astype(str),
        start=table["start"].to_numpy(), split=table["split"].to_numpy().astype(str),
        pair_bp=args.pair_bp, threshold=args.pair_threshold,
    )
    release.record("pairs", pair_bp=args.pair_bp, tissues=list(args.tissues),
                   n_pairs=int(len(indices)), n_sources=n_src,
                   threshold=args.pair_threshold)
    print(f"wrote {release.path('track_weights')} and {release.path('omics_pairs')}")
    return 0


class OmicsPairIndex:
    """Random access to the precomputed pairs.

    ``sources`` are the windows that survived (enough tracks, not generic, at
    least one neighbour above threshold) -- sample seeds from there, then pull
    that seed's partners with ``neighbours``. Both are O(1); nothing is searched
    at training time.

        index = OmicsPairIndex.load(release.path("omics_pairs"))
        seed = index.sources[rng.integers(len(index.sources))]
        partners, scores = index.neighbours(seed)
    """

    def __init__(self, window_ids, indptr, indices, scores, chrom, start, split,
                 pair_bp):
        self.window_ids = window_ids
        self.indptr = indptr
        self.indices = indices
        self.scores = scores
        self.chrom = chrom
        self.start = start
        self.split = split
        self.pair_bp = int(pair_bp)
        # window_id -> row in the CSR structure
        self._row = np.full(len(chrom), -1, dtype=np.int64)
        self._row[window_ids] = np.arange(len(window_ids))
        self.sources = window_ids[np.diff(indptr) > 0]

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        return cls(z["window_ids"], z["indptr"], z["indices"], z["scores"],
                   z["chrom"], z["start"], z["split"], int(z["pair_bp"]))

    def neighbours(self, window_id):
        """Partner window ids and their weighted-Jaccard scores."""
        row = self._row[window_id]
        if row < 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        lo, hi = self.indptr[row], self.indptr[row + 1]
        return self.indices[lo:hi], self.scores[lo:hi]

    def interval(self, window_id):
        """``(chrom, start, end)`` for a window id."""
        s = int(self.start[window_id])
        return str(self.chrom[window_id]), s, s + self.pair_bp


if __name__ == "__main__":
    raise SystemExit(main())
