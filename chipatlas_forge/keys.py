"""Group keys and the filesystem names they turn into.

A "group" is one output BED: everything ChIP-Atlas recorded for one
(antigen class, antigen, cell-type class) triple within one genome assembly.
Group ids are small ints assigned once, in sorted key order, by ``manifest.py``
-- stable across reruns, which is what lets stage 2 and stage 3 run as separate
SLURM jobs that agree on what bucket 37 contains without talking to each other.
"""

import re
import unicodedata

# Fields of chip_atlas_experiment_list.csv, in file order. Only the first six
# are read; the rest (title, metadata blob, per-threshold download URLs) run to
# 450 MB and are not needed to group peaks.
META_COLUMNS = ["srx", "org", "ag_class", "antigen", "ct_class", "celltype"]

# The default grouping. `celltype` (1,698 distinct on hg38) is deliberately not
# in here: it would split hg38 into 10,583 groups instead of 3,172, and the fine
# cell type is still recoverable per-peak from the SRX column that every output
# row carries. Override with --group-by if you want it.
DEFAULT_GROUP_FIELDS = ("ag_class", "antigen", "ct_class")

# Genuinely-absent values only.
#
# "Unclassified" and "No description" are deliberately NOT here. They look like
# placeholders but are real ChIP-Atlas antigen classes, sitting in the same
# column as "Histone" and "ATAC-Seq" and covering 14,221 and 10,449 hg38
# experiments respectively -- 12.5% of the assembly between them. Folding them
# into one NA bucket merges two categories the source keeps apart, and merges
# them with true blanks on top of that.
#
# The live experimentList.tab has no empty fields and no dashes at all in these
# columns, so in practice this set only ever matches "NA" (which maps to itself)
# -- it is kept for the bundled 2021 CSV and for defensiveness, not because the
# current source needs it.
PLACEHOLDERS = {"", "N/A", "-"}

_SAFE = re.compile(r"[^A-Za-z0-9._+-]+")


def slugify(name: str) -> str:
    """A filesystem-safe form of an antigen or cell-type name.

    These names are free text out of GEO/SRA submissions and contain slashes
    ("CD4/CD8"), spaces, brackets, percent signs and non-ASCII characters, any
    of which either break a path or make one that is painful to quote in a
    shell. Runs of unsafe characters collapse to a single underscore.

    This is deliberately lossy and therefore NOT invertible -- "CD4/CD8" and
    "CD4 CD8" both become "CD4_CD8". The exact original strings are written to
    groups.tsv alongside the path, and ``manifest.py`` refuses to build a
    manifest in which two different groups slug to the same path, so the loss
    can never silently merge two antigens into one file.
    """
    if name is None:
        return "NA"
    text = unicodedata.normalize("NFKD", str(name).strip())
    text = text.encode("ascii", "ignore").decode("ascii")
    text = _SAFE.sub("_", text).strip("_.")
    return text or "NA"


def normalise_field(value: str) -> str:
    """Collapse ChIP-Atlas's several spellings of 'missing' to one token."""
    text = "" if value is None else str(value).strip()
    return "NA" if text in PLACEHOLDERS else text


def group_path(org: str, ag_class: str, antigen: str, ct_class: str) -> str:
    """Where a group's BED lands, relative to the output root.

    ``<org>/<antigen class>/<cell type class>/<antigen>.bed`` -- antigen class
    on the outside because it is the coarsest and most-filtered-on field (you
    almost always want "Histone" or "TFs and others", rarely both), and antigen
    as the leaf because that is what a downstream glob keys on.
    """
    return "%s/%s/%s/%s.bed" % (
        slugify(org), slugify(ag_class), slugify(ct_class), slugify(antigen)
    )
