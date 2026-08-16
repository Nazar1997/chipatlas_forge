"""Stage 6 -- decide which tissues and antigens a release actually trains on.

The peak stages produce every group ChIP-Atlas has: 5,856 of them on hg38,
across 24 cell-type classes and 10 antigen classes. Most are unusable as
training targets -- an antigen assayed in a single tissue teaches the model
nothing transferable, and a tissue with a handful of antigens gives a
near-empty target matrix. This stage applies the two thresholds that turn the
full catalogue into a vocabulary, and writes down what it decided.

**Read from groups.tsv, not from a directory listing.** The old
`get_target_tissues_and_features` did `os.listdir(OMICS)` and split filenames
on "." at indices 1 and 3, so it depended on a flat layout with dotted names
that the data has not had for some time -- it cannot run against the tree it
supposedly built. groups.tsv is the manifest stage's own output and carries
(ag_class, antigen, ct_class, n_experiments) as columns, so nothing is parsed
out of a path.

**Freezing.** `--freeze-features` pins the feature list to an existing
release's. The omics head's output dimension is the feature count rounded up to
a power of two, so a vocabulary change makes every existing checkpoint
architecturally unloadable. A refresh is a deliberate act with a retraining
budget attached, not something a data rebuild does on its own. Frozen features
missing from the new data are kept as columns that are simply never available
in any tissue -- dropping them would renumber every column after them, which is
the same incompatibility by another name.

Usage:
    python -m chipatlas_forge.vocab --data-dir ../ --org hg38 --release 2026-08 \\
        --freeze-features ../hg38/releases/2021-10/index/features.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import layout

# A tissue needs at least this many distinct antigens to be worth a column of
# the target matrix, and an antigen at least this many tissues to be worth a
# row. Carried over from the original preparation (src/config.py
# HYPERPARAMETERS["feature_thresholds"]) so a rebuild is comparable to what came
# before.
MIN_ANTIGENS_PER_TISSUE = 50
MIN_TISSUES_PER_ANTIGEN = 3

# Not a real assay: an epitope tag marks whatever construct was transfected, so
# the "antigen" names a purification handle rather than anything in the genome.
EXCLUDED_ANTIGENS = frozenset({"Epitope_tags"})

# Antigen classes that are not measurements of a genomic feature. Input control
# is the background a peak caller subtracts, and the other two are ChIP-Atlas's
# own uncertainty about what was assayed -- all real rows, none of them a
# target. They stay in `signal/` (they are legitimate data and a future release
# may want them); they just are not vocabulary.
EXCLUDED_AG_CLASSES = frozenset({"Input control", "Annotation tracks"})


def read_class_codes(meta_dir):
    """ChIP-Atlas's own three-letter codes for antigen and cell-type classes.

    Derived from fileList.tab rather than hardcoded. Its first column is
    ``<ag code>.<ct code>.<threshold>.<antigen>.<cell type>`` and columns 3 and
    5 are the same two classes spelled out, so the rows where antigen and cell
    type are both "All" state the mapping directly -- 9 antigen classes and 30
    cell-type classes, maintained upstream.

    Worth carrying because the 2021 tree is *named* in these codes
    (``OMICS/His/Bld/``), so they are what translates an old path or an old
    ``target_tissues.pkl`` into the full names a new release uses.
    """
    path = Path(meta_dir) / "fileList.tab"
    if not path.exists():
        return {}, {}
    ag_codes, ct_codes = {}, {}
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            parts = line.split("\t", 5)
            if len(parts) < 5:
                continue
            name = parts[0].split(".")
            if len(name) < 5 or name[3] != "AllAg" or name[4] != "AllCell":
                continue
            ag_codes.setdefault(parts[2], name[0])
            ct_codes.setdefault(parts[4], name[1])
    return ag_codes, ct_codes


def read_groups(release):
    """The manifest stage's groups.tsv, as a frame.

    ``keep_default_na=False`` is load-bearing, not tidiness. `normalise_field`
    writes the literal string ``NA`` for a genuinely-absent antigen or cell
    type, and pandas' default NA list contains ``"NA"`` -- so without this those
    rows come back as float ``nan``. That fails loudly here (``sorted`` cannot
    order float against str) but would otherwise have produced a vocabulary with
    a ``nan`` entry in it, and `keys.py` is explicit that ``NA`` is a category
    ChIP-Atlas keeps rather than a blank.
    """
    path = release.path("groups")
    if not path.exists():
        raise SystemExit(
            "%s is missing -- run `manifest` and copy its groups.tsv into the "
            "release, or run the prepare pipeline in order" % path)
    frame = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[],
                        dtype={"ag_class": str, "antigen": str, "ct_class": str})
    for column in ("ag_class", "antigen", "ct_class"):
        blank = frame[column].isna() | (frame[column] == "")
        if blank.any():
            raise SystemExit(
                "%s has %d row(s) with an empty %s; `manifest` should have "
                "normalised those to 'NA'" % (path, int(blank.sum()), column))
    return frame


def select(groups, min_antigens=MIN_ANTIGENS_PER_TISSUE,
           min_tissues=MIN_TISSUES_PER_ANTIGEN,
           excluded_antigens=EXCLUDED_ANTIGENS,
           excluded_ag_classes=EXCLUDED_AG_CLASSES):
    """Apply the two thresholds, and report what each one cost.

    The thresholds are mutually dependent -- dropping a sparse tissue can push
    an antigen below its tissue count, and vice versa -- so they are iterated to
    a fixed point rather than applied once. One pass in whichever order leaves
    survivors that violate the other threshold, which is how a "min 3 tissues"
    feature list ends up containing features present in two.
    """
    usable = groups[
        ~groups["ag_class"].isin(excluded_ag_classes)
        & ~groups["antigen"].isin(excluded_antigens)
    ]
    pairs = set(zip(usable["ct_class"], usable["antigen"]))

    rounds = 0
    while True:
        rounds += 1
        by_tissue, by_antigen = defaultdict(set), defaultdict(set)
        for tissue, antigen in pairs:
            by_tissue[tissue].add(antigen)
            by_antigen[antigen].add(tissue)
        keep_tissues = {t for t, a in by_tissue.items() if len(a) >= min_antigens}
        keep_antigens = {a for a, t in by_antigen.items() if len(t) >= min_tissues}
        pruned = {(t, a) for t, a in pairs
                  if t in keep_tissues and a in keep_antigens}
        if pruned == pairs:
            break
        pairs = pruned

    tissues = sorted(keep_tissues)
    antigens = sorted(keep_antigens)
    availability = {t: sorted(a for tt, a in pairs if tt == t) for t in tissues}
    return tissues, antigens, availability, rounds


def apply_freeze(frozen, antigens, availability):
    """Keep the frozen column order; report what moved underneath it.

    Returns the frozen list unchanged -- that is the point -- along with the
    antigens that vanished from the data (kept as permanently-unavailable
    columns) and the ones that appeared and are being left out.
    """
    present = set(antigens)
    vanished = [a for a in frozen if a not in present]
    appeared = sorted(present - set(frozen))
    allowed = set(frozen)
    trimmed = {t: [a for a in feats if a in allowed]
               for t, feats in availability.items()}
    return list(frozen), trimmed, vanished, appeared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="the data/ directory holding <org>/releases/")
    parser.add_argument("--org", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--meta-dir", type=Path, default=Path("meta"),
                        help="forge's meta/ directory, for fileList.tab")
    parser.add_argument("--freeze-features", type=Path, default=None,
                        help="a features.json (or a legacy target_features.pkl) "
                             "whose column order this release must reproduce")
    parser.add_argument("--min-antigens", type=int,
                        default=MIN_ANTIGENS_PER_TISSUE)
    parser.add_argument("--min-tissues", type=int,
                        default=MIN_TISSUES_PER_ANTIGEN)
    args = parser.parse_args(argv)

    release = layout.Release.open(args.data_dir, args.org, args.release)
    groups = read_groups(release)
    ag_codes, ct_codes = read_class_codes(args.meta_dir)

    tissues, antigens, availability, rounds = select(
        groups, args.min_antigens, args.min_tissues)
    print("from %d groups: %d tissues, %d antigens (converged in %d round%s)"
          % (len(groups), len(tissues), len(antigens), rounds,
             "" if rounds == 1 else "s"), flush=True)

    frozen_from, vanished, appeared = None, [], []
    if args.freeze_features:
        frozen = read_feature_list(args.freeze_features)
        frozen_from = str(args.freeze_features)
        antigens, availability, vanished, appeared = apply_freeze(
            frozen, antigens, availability)
        print("frozen to %d features from %s" % (len(antigens), frozen_from))
        print("  %d frozen features absent from this data (kept as never-available"
              " columns)%s" % (len(vanished),
                               ": " + ", ".join(vanished[:8]) if vanished else ""))
        print("  %d new features left out to preserve the column order%s"
              % (len(appeared),
                 ": " + ", ".join(appeared[:8]) if appeared else ""))
        # A tissue can fall below threshold once the new-only antigens are
        # removed; it would otherwise be listed with almost nothing available.
        thin = [t for t in tissues if len(availability[t]) < args.min_antigens]
        if thin:
            print("  dropping %d tissue(s) that fall below %d features under the "
                  "frozen vocabulary: %s"
                  % (len(thin), args.min_antigens, ", ".join(thin)))
            tissues = [t for t in tissues if t not in thin]
            availability = {t: availability[t] for t in tissues}

    index_dir = release.path("tissues").parent
    index_dir.mkdir(parents=True, exist_ok=True)

    release.path("tissues").write_text(json.dumps({
        "tissues": [
            {"name": t, "code": ct_codes.get(t), "n_features": len(availability[t])}
            for t in tissues
        ],
    }, indent=2) + "\n")

    release.path("features").write_text(json.dumps({
        "features": antigens,
        "frozen_from": frozen_from,
        "absent_from_data": vanished,
        "excluded_by_freeze": appeared,
    }, indent=2) + "\n")

    release.availability().write_text(
        json.dumps({t: availability[t] for t in tissues}, indent=2) + "\n")

    release.manifest["class_codes"] = {"antigen": ag_codes, "cell_type": ct_codes}
    release.record("vocab",
                   n_tissues=len(tissues), n_features=len(antigens),
                   min_antigens=args.min_antigens, min_tissues=args.min_tissues,
                   frozen_from=frozen_from)
    print("wrote %s, %s and %s"
          % (release.path("tissues").name, release.path("features").name,
             release.availability().name), flush=True)
    return 0


def read_feature_list(path):
    """A feature list from either a new features.json or a legacy pickle."""
    path = Path(path)
    if path.suffix == ".json":
        blob = json.loads(path.read_text())
        return list(blob["features"] if isinstance(blob, dict) else blob)
    from joblib import load          # only the legacy path needs joblib
    return [str(f) for f in load(path)]


if __name__ == "__main__":
    raise SystemExit(main())
