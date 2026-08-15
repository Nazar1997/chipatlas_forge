"""Split ChIP-Atlas allPeaks archives into per-(organism, antigen, tissue) BEDs.

Four stages, run as SLURM jobs:

    manifest  meta zips            -> accession -> group id lookup
    shard     allPeaks_*.bed.gz    -> ordered, independently readable shards
    route     shard                -> parquet parts, bucketed by group   [array]
    collect   parquet parts        -> out/<org>/<ag class>/<tissue>/<antigen>.bed

Only `shard` is serial, and only because plain gzip cannot be inflated in
parallel. See each module's docstring for why it is shaped the way it is.
"""

__all__ = ["keys", "manifest", "shard", "route", "collect", "report"]
