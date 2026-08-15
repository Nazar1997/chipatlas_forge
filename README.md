# chipatlas_forge

Splits the ChIP-Atlas `allPeaks_light` archives into one BED per
**(organism, antigen class, antigen, tissue)**.

```
out/hg38/Histone/Blood/H3K27ac.bed
out/hg38/TFs_and_others/Liver/CTCF.bed
out/mm10/Histone/Neural/H3K4me3.bed
```

Input is ~1.5 billion peaks for hg38 and ~1.1 billion for mm10. The design is
shaped almost entirely by two facts about that data, both verified rather than
assumed:

* **The archives are plain gzip, not BGZF.** A plain DEFLATE stream has no
  independently addressable blocks, so it cannot be seeked, split by byte range,
  or inflated by more than one core. `pigz` parallelises compression, not
  decompression. No arrangement of SLURM tasks makes a single 10 GB `.gz`
  inflate faster than one core can manage.
* **The peaks are already sorted by (chrom, start)** — 0 out-of-order starts in
  the leading 3 M rows — while accessions are scattered throughout (71,684
  distinct SRX in those same 3 M rows).

Sortedness is the important one. Because nothing in the pipeline reorders rows,
and both regrouping sorts are *stable*, the final BEDs come out sorted with **no
sort step anywhere**. Sorting 1.5 billion rows would otherwise cost more than
everything else combined.

## Stages

| stage | parallelism | what it does |
|---|---|---|
| `manifest` | 1 task | meta zips → accession → group id lookup |
| `shard` | 1 task/org | inflate once, cut into newline-aligned shards |
| `route` | **1 task/shard** | parse, resolve accessions, partition by bucket |
| `collect` | **1 task/bucket** | concatenate parts → final BEDs |

Only `shard` is serial, and only for the reason above. It is a one-time cost:
shards persist, so changing the grouping and rerunning `route` + `collect` never
pays for the inflate again.

## Run it

```bash
ORG=hg38 ./slurm/run_all.sh          # submits the whole chain, exits
ORG=mm10 ./slurm/run_all.sh

squeue -u $USER
python -m chipatlas_forge.report --root . --org hg38
```

`run_all.sh` submits `manifest` and `shard` concurrently (they read different
directories), then `route` gated on both, then `collect` gated on `route`.

Knobs, all environment variables:

| var | default | note |
|---|---|---|
| `BUCKETS` | 128 | stage-3 array width; **stages 2 and 3 must agree** |
| `CHUNK` | `512M` | uncompressed bytes/shard → sets stage-2 array width |
| `THROTTLE` | 100 | concurrent array tasks |
| `COMPRESS` | `none` | `gzip` to write `.bed.gz` |
| `PARTITION` | `cpu-e-quick` | 5 × 128 cores, 1-day limit |

## Why it is fast

**Dictionary-encoded accessions.** A shard holds ~15 M peaks drawn from only
~100 k distinct accessions. Reading the SRX column as
`dictionary<int32, string>` means the string→group lookup runs once per
*distinct* accession (~20 ms) instead of once per row. The per-row step is one
numpy take.

**pyarrow's CSV reader.** Multithreaded C++; a 512 MB shard parses in seconds.
Python-level line splitting over 1.5 B rows is the difference between minutes
and a day.

**Bucketing instead of per-group files.** One file per group per shard would be
110 × 3,172 ≈ 349 k files for hg38 alone. Hashing groups into `BUCKETS` keeps it
at 110 × 128, and since a group always lands in one bucket, `collect` finishes a
group by reading one bucket — no merge pass.

**Streaming reduce.** Group sizes are extremely skewed (ATAC-Seq/Blood dwarfs
most transcription factors), so `collect` appends part-by-part to open handles.
Peak memory tracks the largest *part*, not the largest group.

## Get current data first

**The archives from the Yandex share are an old release.** Measured against
ChIP-Atlas upstream:

| | local copy | upstream (2024-11-13) | |
|---|---|---|---|
| `allPeaks_light.hg38.05` | 10.24 GB | **22.17 GB** | 2.17× |
| `allPeaks_light.mm10.05` | 7.70 GB | **15.88 GB** | 2.06× |
| `chip_atlas_experiment_list` | Oct **2021**, 439,593 exp | `experimentList.tab` Oct **2025**, 845,824 exp | 1.92× |

The metadata gap is the one that silently corrupts results: against the 2021
zip, 3.2% of hg38 peaks cite accessions it has never heard of, and those peaks
are dropped — ~48 million at full scale. With `experimentList.tab` the unmapped
rate falls to **0.21%**.

```bash
ORGS="hg38 mm10" ./fetch_upstream.sh          # -> raw_fresh/ and meta/
```

It writes to `raw_fresh/`, not `raw/`, so the pipeline keeps working on whatever
you have until you swap. Verification is Content-Length **plus a full gzip CRC
check** — ChIP-Atlas publishes no checksums, and size alone accepts a truncated
file.

## Grouping

Default is `(antigen class, antigen, cell type class)` — **5,856 groups** for
hg38 off the live metadata. The finer `cell type` would roughly triple that; it
is left out because the exact cell type stays recoverable per-peak from the SRX
column every output row carries. Override with
`--group-by ag_class antigen ct_class celltype`.

`Unclassified` and `No description` are **kept as real categories**, not folded
into `NA`. They read like placeholders but are genuine ChIP-Atlas antigen
classes sitting beside `Histone` and `ATAC-Seq`, covering 14,221 and 10,449
hg38 experiments — 12.5% of the assembly between them. Only truly absent values
(`""`, `-`, `N/A`) collapse, and the live metadata contains none of those.

Names are slugified for the filesystem, which is lossy and therefore **not**
invertible. `manifest.py` refuses to build a manifest where two groups slug to
the same path, so the loss can never silently merge two antigens. The exact
original strings live in `work/manifest/<org>/groups.tsv`.

## Checking a run

`report.py` verifies the two things a SLURM exit code cannot:

* **conservation** — `peaks read − unmapped == peaks written`
* **order** — spot-checks that the largest outputs really are sorted

It also flags shards that were never routed, which is what a partially failed
array looks like.

## Notes for this cluster

* **The Arrow ↔ numpy boundary is broken in `myenv`.** pyarrow 20 is built
  against numpy 2.x and sits next to numpy 1.26.3, so `Array.to_numpy()`,
  `pa.array(ndarray)` and `table.take(ndarray)` all raise. This is the same
  break torch 2.5.1 hit with numpy 1.26.3 in the pair builder. `arrow_compat.py`
  crosses via the buffer protocol instead — zero-copy, and strictly less work
  than the converter. A guard test fails if anyone reaches for the broken API
  again, which is important because it works fine on a laptop.
* Never pass `--mem`: `cpu-e-quick` reports `RealMemory=1` because memory is not
  a scheduled resource, so any request is rejected outright.
* `/usr/bin/python` is 2.7.5. The scripts name
  `$HOME/.conda/envs/myenv/bin/python` by absolute path rather than relying on
  `module load` + `conda activate`, either of which can silently no-op in a
  batch job.
* `pytest` in `myenv` segfaults on import (`pytest --version` crashes). pyarrow
  itself is fine. Run `tests/` off-cluster.
* The meta CSV is **cp1252, not utf-8** — it carries smart quotes that make
  utf-8 decoding raise. `manifest.py` reads it as latin-1, which cannot fail.
