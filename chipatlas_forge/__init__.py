"""ChIP-Atlas peaks in, training-ready releases out.

Two halves, joined at `adopt`.

**The peak pipeline** turns the raw allPeaks archives into one signal track per
(organism, antigen class, tissue, antigen). It works inside this repository's
own tree, because it is a build: intermediates, rerunnable, safe to delete.

    manifest  meta zips            -> accession -> group id lookup
    shard     allPeaks_*.bed.gz    -> ordered, independently readable shards
    route     shard                -> parquet parts, bucketed by group   [array]
    collect   parquet parts        -> out/<org>/<ag>/<tissue>/<antigen>.bed
    binmax    overlapping peaks    -> a max-score bedGraph               [array]

**The prepare pipeline** turns those tracks into a *release*: a dated, immutable,
self-describing directory under `data/<org>/releases/` that training reads
directly. It is versioned because staleness is invisible otherwise -- the 2021
tree gave no sign on disk that its metadata was three years older than its peaks.

    adopt      signal tracks + groups.tsv into a new release
    vocab      which tissues and antigens the release trains on
    genome     reference, blacklist, chromosome lengths
    intervals  chunk grid, window tables, and the train/val/test split
    chunks     per-chunk DNA text and per-(tissue, chrom) omics parquet [array]
    pairs      IDF track weights and the omics-similarity pair index
    verify     prove it is complete before anything trains on it

`migrate` files a pre-release tree as a release without rewriting it, and
`read` is the reference reader -- everything it does is driven by the release's
own MANIFEST.json, so one code path handles every layout that has ever shipped.

See each module's docstring for why it is shaped the way it is.
"""

__all__ = [
    # peak pipeline
    "keys", "manifest", "shard", "route", "collect", "binmax",
    # prepare pipeline
    "adopt", "vocab", "genome", "intervals", "chunks", "pairs", "verify",
    # layout, reading, and migration
    "layout", "read", "migrate", "report",
]
