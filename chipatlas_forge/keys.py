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

# ChIP-Atlas leaves these in as literal strings rather than empty fields.
PLACEHOLDERS = {"", "NA", "N/A", "-", "No description", "Unclassified"}

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
