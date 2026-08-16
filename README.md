# chipatlas_forge

ChIP-Atlas peaks in, training-ready **releases** out.

Two halves. The **peak pipeline** splits the `allPeaks_light` archives into one
BED per (organism, antigen class, antigen, tissue) and collapses each into a
max-score signal track. The **prepare pipeline** turns those tracks into a
dated, immutable, self-describing release that the training repo reads directly.

```
out/hg38/Histone/Blood/H3K27ac.bed              # peak pipeline
out_binmax1/hg38/Histone/Blood/H3K27ac.bedgraph #   ... collapsed to signal

data/hg38/releases/2026-08/                     # prepare pipeline
data/hg38/latest -> releases/2026-08            #   ... what training resolves
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
| `binmax` | **1 task/file slice** | overlapping peaks → max-score signal track |
| `adopt` | 1 task | link signal + groups.tsv into a new release |
| `genome` | 1 task | reference, blacklist, chromosome lengths |
| `pantissue` | **1 task/antigen slice** | all-cell-types track: max over every tissue |
| `vocab` | 1 task | which tissues and antigens the release trains on |
| `intervals` | 1 task | chunk grid, window tables, train/val/test split |
| `chunks` | **1 task/tissue** | per-chunk DNA text, per-(tissue, chrom) omics parquet |
| `pairs` | 1 task | IDF track weights + omics-similarity pair index |
| `verify` | 1 task | prove the release is complete before promoting it |

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
ORGS="hg38 mm10" ./promote_fresh.sh           # raw_fresh/ -> raw/, once verified
```

`fetch_upstream.sh` writes to `raw_fresh/`, not `raw/`, so the pipeline keeps
working on whatever you have until you swap. Verification is Content-Length
**plus a full gzip CRC check** — ChIP-Atlas publishes no checksums, and size
alone accepts a truncated file.

`promote_fresh.sh` re-checks each archive before moving it, keeps the old ones
in `raw_superseded/` rather than deleting, and clears `work/` — shards left
behind from a previous archive would be silently mixed with new ones, since
stage 2 addresses shards positionally.

## Peaks vs. signal (`binmax`)

`out/` holds **raw peaks**, so they overlap — every experiment that called a peak
at a locus contributes its own row:

```
chr1  9915  10410  SRX24935670  644
chr1  9919  10288  SRX15914382  377
chr1  9921  10441  SRX24935673  643
```

That is a pile of intervals, not a signal. `binmax` collapses it to the maximum
score over every base, as a 4-column bedGraph of constant-value runs:

```bash
ORG=hg38 TASKS=100 BIN_SIZE=1 sbatch --array=0-99%50 slurm/05_binmax.sh
```
```
chr1  9903   10466  1328
chr1  10466  10475  1091
chr1  10475  10535  937
```

Disjoint, sorted, zero-coverage gaps omitted. Output goes to
`out_binmax<bin>/`, so `out/` is untouched and another resolution costs only
this stage.

**It does not allocate a dense genome.** Painting each peak across its ~570 bases
would be ~2.2 *trillion* writes over 3.89 B peaks. The maximum is piecewise
constant and can only change where a peak starts or ends, so ≤ 2n breakpoints
pin the whole function. Working in that compressed space is **exact at 1 bp** —
nothing is approximated — while memory becomes O(peaks), each write spans the
local overlap *depth* rather than a length in bases, and the output is
run-length encoded for free. Painting in ascending score order makes plain slice
assignment compute the maximum.

Measured on the full dataset: 3.89 B peaks → 673 M runs, **152 GB → 18.8 GB**,
slowest task 316 s.

Correctness is pinned against a dense per-base brute force, not by inspection —
any error in a scheme like this lands on a boundary, exactly where a spot-check
does not look. Two real files were verified byte-identical to brute force across
2,000,000 and 4,000,000 consecutive bases.

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


## Releases

A release is everything training needs for one organism, under one dated
directory, with nothing outside it:

```
data/hg38/
  latest -> releases/2026-08          # the only thing that decides what runs read
  releases/2026-08/
    MANIFEST.json
    genome/    genome.fa  blacklist.bed  sequence.pkl  chrom.sizes
    signal/    <antigen class>/<tissue>/<antigen>.bedgraph
    chunks/    dna/<chrom>/<start>_<end>.txt
               omics/<tissue>/<chrom>.parquet
    index/     tissues.json  features.json  availability.json
               chunk_grid.parquet  windows_<W>.parquet
               track_weights.npz  omics_pairs.npz  groups.tsv
```

**Why versioned.** Nothing on disk used to say which ChIP-Atlas snapshot
`data/hg38` came from. It turned out to be a 2021 metadata dump against 2024
peaks, dropping 3.2% of hg38 peaks as unrecognised accessions, and the only tell
was a suspiciously large `NA/Blood/NA.bed`. A release id in the path makes the
snapshot part of every filename a run touches, and rollback one `ln -sfn`.

**The manifest carries its own path templates.** This repository writes releases
and the training repo reads them. Rather than each holding a copy of the same
twenty path strings and drifting apart silently, every release states its layout
in `MANIFEST.json`:

```json
"paths": {"dna_chunk": "chunks/dna/{chrom}/{start}_{end}.txt", ...},
"omics_layout": "chrom-parquet",
"interval_layout": "split-column"
```

A reader formats those templates, so the data describes itself and old releases
stay readable after the layout moves on. That is what lets the migrated 2021
release keep its ~900,000 per-chunk omics pickles and its 4.9 GB interval
pickles while new releases use 475 parquet files and one window table per size:
the two declare different templates and the same reader handles both.

## The split

Validation is **chr8 + chr9, held out whole** — 9.18% of hg38 and 9.32% of
mm10, the closest any pair gets to a tenth on both assemblies at once. Whole
chromosomes, so no validation window shares a regulatory neighbourhood with a
training one.

Test is **10% of what remains**, drawn at the interval level: validation already
costs a tenth of the genome, and spending two more whole chromosomes on test
would take a fifth of the training signal.

The split is assigned on **2**20 blocks and inherited downward**. Assigning it
per window independently would put an 8192 window in train while the 65536
window containing it is in test — the same bases on both sides of the split.
Because 8192 and 65536 both divide 2**20, a block-level assignment nests
exactly, so all three window sizes agree about which stretches of genome are
held out. `verify` checks it: no locus may appear in two splits.

This also reconciles a real disagreement. `build_omics_pairs.py` held out
chr8/chr9 while `train/test_intervals_*.pkl` split *within* every chromosome, so
two artifacts feeding the same run disagreed about what "held out" meant.
`pairs` now reads the `split` column rather than re-deriving it.

## Omics storage: 475 files, not 900,000

The 2021 tree stored one pickle per (tissue, chromosome, chunk) — about 900,000
files per organism, enough that `du` times out. New releases store one parquet
per (tissue, chromosome) with **a row group per chunk**, and the chunk → row
group map in the file footer:

```python
handle = pq.ParquetFile(path)
present = json.loads(handle.schema_arrow.metadata[b"chipatlas_forge:chunks"])
rows = handle.read_row_group(present.index(chunk))
```

Same O(1) access to one chunk, 1/1900th the file count. A peak crossing a chunk
boundary is stored in **both** chunks with its true coordinates uncut — the read
path clips to the window and de-duplicates, and storing clipped copies would
silently shorten every boundary-crossing peak.

## Run the prepare stages

```bash
# after run_all.sh has finished stages 1-5
ORG=hg38 RELEASE=2026-08 slurm/run_prepare.sh

# promote only once verify exits 0
ORG=hg38 RELEASE=2026-08 PROMOTE=1 slurm/run_prepare.sh
```

Everything is chained with `--dependency=afterok`, and promotion is opt-in and
last: `latest` is what every training job resolves, so moving it is the single
action that changes what runs read.

## Filing the pre-release tree as a release

`migrate` turns `data/<org>/{DNA,OMICS,Subtables,SupportFiles}` into a release
without rewriting any of it — every move is an `os.rename`, so 152 GB and
900,000 files take the same instant as one file. What changes is the naming:
`Bld` becomes `Blood` and `His` becomes `Histone`, using the code table derived
from ChIP-Atlas's own `fileList.tab` rather than a hardcoded map.

It is destructive and one-way, so it prints its plan and stops:

```bash
python -m chipatlas_forge.migrate --data-dir ../ --org hg38 --release 2021-10
python -m chipatlas_forge.migrate --data-dir ../ --org hg38 --release 2021-10 --execute
```

## Vocabulary is frozen by default

The omics head's output dimension is the feature count rounded up to a power of
two, so changing the vocabulary makes every existing checkpoint architecturally
unloadable. `--freeze-features` pins the feature list to an existing release's;
features that vanished from the new data are kept as columns that are never
available in any tissue, because dropping them would renumber every column after
them — the same incompatibility by another name.

Refreshing the vocabulary is a deliberate act with a retraining budget attached,
not something a data rebuild does on its own.


## The all-cell-types track

ChIP-Atlas's cell-type classes are real tissues, so nothing grouped from
per-experiment metadata can produce an "all cell types" bucket. The 2021 tree
had one only because it took ChIP-Atlas's precomputed `*.AllCell.bed`
aggregates rather than deriving anything — and losing it is not cosmetic: it is
the tissue `only_one_tissue` runs train on, the default the embedding
datamodule falls back to, and what the pair builder profiles against.

`pantissue` derives it by max-merging each antigen across every cell type.
That is exact rather than approximate: each per-tissue track is already the
maximum over that tissue's peaks, so the maximum *across* tissues is the maximum
over every peak of that antigen anywhere — the same answer `binmax` would give
on the union of the raw peaks, without going back to them.

It is also strictly better than what it replaces. The 2021 pan-tissue track was
**not** a superset: on hg38 it carried ~1004 antigens but none of H3K27ac /
H3K4me1 / H3K4me3 / H3K27me3 / RNA polymerase II, because ChIP-Atlas publishes
those only per-tissue — so pairing on it had no view of the marks that define
enhancers and promoters.

## Freezing, and the names the 2021 vocabulary got wrong

`--freeze-features` pins the feature list, because the omics head's output
dimension is the feature count rounded up to a power of two and changing it
makes every existing checkpoint architecturally unloadable.

Two things that freezing must *not* do, both found by running against the real
data:

**The thresholds are not re-applied.** `min_tissues` exists to *choose* a
vocabulary; once frozen the choice is made. Re-running it over the new snapshot
marked 395 of hg38's 1,009 frozen features absent — 392 of them with data in
exactly two tissues — which would have zeroed 39% of the target matrix because
tissue counts shifted between snapshots.

**The old dot mangling is undone.** The 2021 preparation parsed
`05.<antigen>.AllCell.bed` by splitting on `.`, so a dotted antigen could not
survive a round trip; something upstream substituted the literal string
`PERIOD` for the dot and underscores for spaces. The frozen vocabulary holds
`H2APERIODX`, `H3PERIOD3_K27M_mutant` and `RNA_polymerase_II` where ChIP-Atlas
says `H2A.X`, `H3.3 K27M mutant` and `RNA polymerase II`. Sixteen of hg38's
seventeen and ten of mm10's eleven apparently-absent features are exactly this,
recovered by canonical matching and recorded in `features.json` as `aliases`.
Only unambiguous matches are taken — guessing wrong files one track's peaks
under another track's column.

After both fixes, hg38 has **1 genuinely absent feature out of 1,009** (`AllAg`,
an aggregate that was never an antigen) and mm10 has 2.
