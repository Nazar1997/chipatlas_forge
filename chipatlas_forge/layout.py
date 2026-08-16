"""Where a prepared release lives on disk, and how a reader finds its way around.

Everything the training code eventually opens sits under one release directory::

    data/<org>/releases/<release id>/
        MANIFEST.json
        genome/     genome.fa  genome.fa.fai  blacklist.bed  sequence.pkl
        signal/     <antigen class>/<tissue>/<antigen>.bedgraph
        chunks/     dna/<chrom>/<start>_<end>.txt
                    omics/...                       (layout-dependent, see below)
        index/      tissues.json  features.json  availability.json
                    chunk_grid.parquet  windows_<W>.parquet
                    track_weights.npz  omics_pairs.npz

    data/<org>/latest -> releases/<release id>

**Releases exist because staleness is invisible.** Nothing on disk used to say
which ChIP-Atlas snapshot `data/hg38` was built from; it turned out to be a
2021 metadata dump against 2024 peaks, and the only tell was a suspiciously
large `NA/Blood/NA.bed`. A release id in the path makes the snapshot part of
every filename a run touches, and makes rollback one `ln -sfn`.

**The manifest carries its own path templates.** This module writes releases and
the training repo reads them -- two repositories that would otherwise each hold
a copy of the same twenty path strings and drift apart silently, the failure
being a `FileNotFoundError` deep inside a dataloader worker. Instead every
release states its own layout in `MANIFEST.json`::

    "paths": {"dna_chunk": "chunks/dna/{chrom}/{start}_{end}.txt", ...}

and a reader formats those templates. The data describes itself, so there is
exactly one source of truth and old releases stay readable after the layout
moves on. It is what lets the 2021 release keep its pickle-per-chunk omics
while new releases use one parquet per (tissue, chromosome): the two declare
different `omics_chunk` templates and the same reader handles both.

`LAYOUT_VERSION` covers the *shape* of the manifest itself -- the keys a reader
looks for -- not the paths, which are data. Bump it only when a reader that
understands version N could not make sense of the file.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

LAYOUT_VERSION = 1

# Omics chunk layouts. The value is what goes in MANIFEST["omics_layout"]; the
# reader dispatches on it because the two need different loading code, not just
# a different path.
#
#   chunked-pickle  one pickle per (tissue, chrom, chunk) -- ~900k files per
#                   organism. What the 2021 tree used. Kept readable so old
#                   checkpoints stay reproducible, not because it is good:
#                   Lustre charges per file and this is a metadata storm.
#   chrom-parquet   one parquet per (tissue, chrom) -- 475 files per organism,
#                   with a row group per chunk so a single chunk is still read
#                   without touching the rest.
CHUNKED_PICKLE = "chunked-pickle"
CHROM_PARQUET = "chrom-parquet"

# Interval layouts, same idea for the window tables.
#
#   split-column   one parquet per window size with a `split` column. What new
#                  releases write.
#   per-split      one pickle per (split, window size), plus `_ALL` and
#                  `_subset` variants. The 2021 tree, kept as it is: those
#                  pickles run to 4.9 GB each and rebuilding them as parquet
#                  would change the very split the old checkpoints were
#                  measured against, which is the one thing the old release
#                  exists to preserve.
SPLIT_COLUMN = "split-column"
PER_SPLIT_PICKLE = "per-split"

# Path templates, relative to the release root. Written verbatim into every
# manifest; readers use the manifest's copy, never this one.
PATH_TEMPLATES = {
    "genome_fasta": "genome/genome.fa",
    "genome_index": "genome/genome.fa.fai",
    "blacklist": "genome/blacklist.bed",
    "sequence": "genome/sequence.pkl",
    "chrom_sizes": "genome/chrom.sizes",
    "signal": "signal/{ag_class}/{tissue}/{antigen}.bedgraph",
    "signal_root": "signal",
    "dna_chunk": "chunks/dna/{chrom}/{start}_{end}.txt",
    "omics_chunk": "chunks/omics/{tissue}/{chrom}.parquet",
    "omics_tissue": "chunks/omics/{tissue}",
    "tissues": "index/tissues.json",
    "features": "index/features.json",
    # One file for all tissues, not one per tissue. The 2021 tree kept a pickle
    # per tissue and the dataset loaded it *inside __getitem__* -- a disk read
    # and an unpickle per training sample, for a table of ~19 x 600 strings that
    # never changes. Whole-file is ~400 kB, so a reader loads it once and keeps
    # it.
    "availability": "index/availability.json",
    "chunk_grid": "index/chunk_grid.parquet",
    "windows": "index/windows_{window}.parquet",
    "track_weights": "index/track_weights.npz",
    "omics_pairs": "index/omics_pairs.npz",
    "groups": "index/groups.tsv",
}

# The 2021 tree, after `migrate` renames it. Only the omics chunks and the
# window tables differ; the rest is the same shape, which is why migration is a
# rename and not a rebuild. The `_all` and `_subset` variants are real files the
# datamodule branches on (single-tissue runs at 512 bp, and a small subset used
# for smoke runs), so they are declared rather than quietly dropped.
LEGACY_PATH_TEMPLATES = dict(
    PATH_TEMPLATES,
    omics_chunk="chunks/omics/{tissue}/{chrom}/{start}_{end}.pkl",
    windows_full="index/full_intervals_{window}.pkl",
    windows_train="index/train_intervals_{window}.pkl",
    windows_test="index/test_intervals_{window}.pkl",
    windows_full_all="index/full_intervals_{window}_ALL.pkl",
    windows_train_all="index/train_intervals_{window}_ALL.pkl",
    windows_test_all="index/test_intervals_{window}_ALL.pkl",
    windows_train_subset="index/train_intervals_subset.pkl",
    windows_test_subset="index/test_intervals_subset.pkl",
)
del LEGACY_PATH_TEMPLATES["windows"]

# The genome is cut into fixed 64 kb chunks and every per-chunk artifact is
# keyed by that grid. Windows (8192, 65536, 2**20) are a separate, coarser or
# finer grid laid over the same coordinates -- a 2**20 window spans 16 chunks,
# and the loader stitches them.
CHUNK_SIZE = 65536

DEFAULT_WINDOWS = (8192, 65536, 1 << 20)

# Validation is held out whole, by chromosome, so no window in it shares a
# regulatory neighbourhood with a training window. chr8 + chr9 is 9.18% of hg38
# and 9.32% of mm10 -- the closest any pair gets to a tenth on both assemblies
# at once -- and it is the holdout `build_omics_pairs` already assumed, so the
# interval splits and the pair index finally agree on what "held out" means.
DEFAULT_VAL_CHROMS = ("chr8", "chr9")

# Test is drawn from what is left at the interval level rather than by
# chromosome: with val already costing a tenth of the genome, spending two more
# whole chromosomes on test would take a fifth of the training signal.
DEFAULT_TEST_FRACTION = 0.1
DEFAULT_SPLIT_SEED = 42

_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def release_root(data_dir, org, release):
    """The directory holding one release of one organism."""
    return Path(data_dir) / org / "releases" / release


def releases_dir(data_dir, org):
    return Path(data_dir) / org / "releases"


def latest_link(data_dir, org):
    return Path(data_dir) / org / "latest"


def list_releases(data_dir, org):
    """Every release id present, oldest first.

    Sorted lexicographically, which orders `YYYY-MM` ids chronologically. That
    is the whole reason ids are dates: "latest" has to be answerable without a
    registry, from the directory listing alone.
    """
    root = releases_dir(data_dir, org)
    if not root.is_dir():
        return []
    found = [p.name for p in root.iterdir()
             if p.is_dir() and (p / "MANIFEST.json").exists()]
    return sorted(found)


def resolve_release(data_dir, org, release=None):
    """Turn ``None``/``"latest"`` into a concrete release id.

    ``latest`` follows the symlink when there is one, so a release can be
    promoted or rolled back without touching any caller. Falling back to the
    newest id by name means a freshly built release is usable before anyone
    remembers to move the link.
    """
    available = list_releases(data_dir, org)
    if release not in (None, "latest"):
        if release not in available:
            raise FileNotFoundError(
                "no release %r for %s under %s (have: %s)"
                % (release, org, releases_dir(data_dir, org),
                   ", ".join(available) or "none"))
        return release

    link = latest_link(data_dir, org)
    if link.is_symlink():
        target = os.readlink(link)
        name = Path(target).name
        if name in available:
            return name
        raise FileNotFoundError(
            "%s points at %r, which is not a readable release (have: %s)"
            % (link, target, ", ".join(available) or "none"))

    if not available:
        raise FileNotFoundError(
            "no releases for %s under %s -- run the prepare stages, or "
            "`migrate` an existing tree" % (org, releases_dir(data_dir, org)))
    return available[-1]


def check_release_id(release):
    """Reject ids that would escape the releases directory or confuse a glob."""
    if not _RELEASE_ID.match(release or ""):
        raise SystemExit(
            "release id %r must match %s -- it becomes a directory name"
            % (release, _RELEASE_ID.pattern))
    return release


def forge_commit():
    """The commit that built a release, or None outside a checkout.

    Recorded so a surprising number can be traced back to the code that
    produced it. `git describe` rather than a bare hash so a tagged release
    reads as a tag.
    """
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "-C", str(here), "describe", "--always", "--dirty", "--tags"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


class Release:
    """A release directory, plus the manifest that says how to read it.

    Writers construct one with :meth:`create`, fill in sections as stages
    complete, and call :meth:`save` -- the manifest is written incrementally so
    a pipeline that dies in stage 8 leaves a release that honestly reports
    having finished 7.
    """

    def __init__(self, root, manifest):
        self.root = Path(root)
        self.manifest = manifest
        self._templates = manifest.get("paths") or PATH_TEMPLATES

    # ---- construction -------------------------------------------------

    @classmethod
    def create(cls, data_dir, org, release, omics_layout=CHROM_PARQUET,
               interval_layout=None, templates=None, **extra):
        check_release_id(release)
        root = release_root(data_dir, org, release)
        root.mkdir(parents=True, exist_ok=True)
        legacy = omics_layout == CHUNKED_PICKLE
        if templates is None:
            templates = LEGACY_PATH_TEMPLATES if legacy else PATH_TEMPLATES
        if interval_layout is None:
            interval_layout = PER_SPLIT_PICKLE if legacy else SPLIT_COLUMN
        manifest = {
            "layout_version": LAYOUT_VERSION,
            "org": org,
            "release": release,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "forge_commit": forge_commit(),
            "omics_layout": omics_layout,
            "interval_layout": interval_layout,
            "chunk_size": CHUNK_SIZE,
            "paths": dict(templates),
            "stages": {},
        }
        manifest.update(extra)
        obj = cls(root, manifest)
        obj.save()
        return obj

    @classmethod
    def open(cls, data_dir, org, release=None):
        release = resolve_release(data_dir, org, release)
        root = release_root(data_dir, org, release)
        manifest = json.loads((root / "MANIFEST.json").read_text())
        version = manifest.get("layout_version")
        if version != LAYOUT_VERSION:
            raise SystemExit(
                "%s is layout version %r, this code writes %d -- refusing to "
                "mix them" % (root / "MANIFEST.json", version, LAYOUT_VERSION))
        return cls(root, manifest)

    def save(self):
        (self.root / "MANIFEST.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n")

    def record(self, stage, **stats):
        """Note that a stage finished, with whatever it counted."""
        stats["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.manifest.setdefault("stages", {})[stage] = stats
        self.save()

    def has(self, stage):
        return stage in (self.manifest.get("stages") or {})

    def require(self, *stages):
        """Fail before doing work rather than half way through it."""
        missing = [s for s in stages if not self.has(s)]
        if missing:
            raise SystemExit(
                "%s has not run %s for %s/%s -- run those stages first"
                % ("stage" if len(missing) == 1 else "stages",
                   ", ".join(missing), self.manifest["org"],
                   self.manifest["release"]))

    # ---- paths --------------------------------------------------------

    @property
    def org(self):
        return self.manifest["org"]

    @property
    def id(self):
        return self.manifest["release"]

    @property
    def omics_layout(self):
        return self.manifest.get("omics_layout", CHROM_PARQUET)

    @property
    def interval_layout(self):
        return self.manifest.get("interval_layout", SPLIT_COLUMN)

    @property
    def chunk_size(self):
        return int(self.manifest.get("chunk_size", CHUNK_SIZE))

    def path(self, key, **kw):
        """Resolve one of the manifest's path templates."""
        try:
            template = self._templates[key]
        except KeyError:
            raise KeyError(
                "release %s declares no path for %r (has: %s)"
                % (self.id, key, ", ".join(sorted(self._templates)))) from None
        return self.root / template.format(**kw)

    def signal_path(self, ag_class, tissue, antigen):
        return self.path("signal", ag_class=ag_class, tissue=tissue,
                         antigen=antigen)

    def dna_chunk(self, chrom, start, end):
        return self.path("dna_chunk", chrom=chrom, start=start, end=end)

    def omics_chunk(self, tissue, chrom, start=None, end=None):
        """The file holding a chunk's omics, whichever layout this release uses.

        Under `chrom-parquet` the whole chromosome is one file and
        ``start``/``end`` are ignored -- the caller then selects the chunk's row
        group. Passing them anyway is deliberate: it keeps one call signature
        across both layouts so the loader has a single code path up to the
        point where the formats genuinely differ.
        """
        if self.omics_layout == CHUNKED_PICKLE:
            if start is None or end is None:
                raise ValueError(
                    "release %s stores omics per chunk; start and end are "
                    "required" % self.id)
            return self.path("omics_chunk", tissue=tissue, chrom=chrom,
                             start=start, end=end)
        return self.path("omics_chunk", tissue=tissue, chrom=chrom)

    def windows(self, window):
        return self.path("windows", window=window)

    def availability(self):
        return self.path("availability")


def promote(data_dir, org, release):
    """Point ``latest`` at a release. The only step that changes what runs read.

    Written to a temporary name and renamed, so the link is never briefly
    absent -- a training job starting during promotion either sees the old
    release or the new one, never a missing path.
    """
    check_release_id(release)
    root = release_root(data_dir, org, release)
    if not (root / "MANIFEST.json").exists():
        raise SystemExit("%s is not a release -- no MANIFEST.json" % root)
    link = latest_link(data_dir, org)
    tmp = link.with_name(link.name + ".new")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(Path("releases") / release)
    os.replace(tmp, link)
    return link
