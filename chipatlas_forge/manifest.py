"""Stage 0 -- turn the ChIP-Atlas experiment list into an SRX -> group lookup.

Runs in about a minute on one core and everything downstream depends on it, so
it is a separate stage rather than something each of 100 array tasks redoes.

The output is deliberately boring: two parallel numpy arrays (accession, group
id) plus a human-readable groups.tsv. Stage 2 loads the arrays into a dict and
stage 3 reads only the tsv, so neither has to parse the 450 MB CSV.

Usage:
    python -m chipatlas_forge.manifest --root . --org hg38 mm10
"""

import argparse
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .keys import (DEFAULT_GROUP_FIELDS, META_COLUMNS, group_path,
                   normalise_field)


def read_experiment_list(meta_dir: Path) -> pd.DataFrame:
    """The six identifying columns of chip_atlas_experiment_list.csv.

    Read straight out of the zip -- the CSV expands to 450 MB and nothing else
    needs it on disk.

    ``encoding="latin-1"``, not utf-8: the file carries cp1252 smart quotes in
    free-text titles (byte 0x91 at offset 6899, for one) and utf-8 decoding dies
    on them. latin-1 cannot fail -- every byte is a valid code point -- and the
    six columns we keep are ASCII in practice, so nothing is mangled. The
    alternative, errors="replace", would silently corrupt an antigen name.
    """
    archive = meta_dir / "chip_atlas_experiment_list.zip"
    if not archive.exists():
        raise SystemExit("missing %s -- fetch the meta/ folder first" % archive)

    with zipfile.ZipFile(archive) as zf:
        inner = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(inner) != 1:
            raise SystemExit("expected one CSV in %s, found %r" % (archive, inner))
        with zf.open(inner[0]) as fh:
            frame = pd.read_csv(
                fh,
                usecols=range(len(META_COLUMNS)),
                names=META_COLUMNS,
                header=0,
                dtype=str,
                keep_default_na=False,
                encoding="latin-1",
                engine="c",
                on_bad_lines="warn",
            )
    return frame


def build_for_org(frame: pd.DataFrame, org: str, group_fields) -> tuple:
    """(srx array, group id array, groups frame) for one genome assembly.

    An accession is unique only *within* an assembly -- ChIP-Atlas maps the same
    experiment onto both hg19 and hg38, so SRX5668296 appears in each with the
    same antigen. Building one lookup per org keeps that from being ambiguous
    and keeps each lookup at ~100k entries.
    """
    sub = frame[frame["org"] == org].copy()
    if sub.empty:
        raise SystemExit("no experiments for org=%s" % org)

    for column in ("ag_class", "antigen", "ct_class", "celltype"):
        sub[column] = sub[column].map(normalise_field)

    # Sorted key order, so group ids are reproducible across reruns and across
    # machines. Stage 2 and stage 3 are separate jobs and only agree on which
    # rows belong to bucket b because this ordering is deterministic.
    keys = sub[list(group_fields)].agg("\x1f".join, axis=1)
    uniq = np.array(sorted(keys.unique()))
    code = {key: i for i, key in enumerate(uniq)}
    group_id = keys.map(code).to_numpy(dtype=np.int32)

    parts = pd.DataFrame([k.split("\x1f") for k in uniq], columns=list(group_fields))
    parts.insert(0, "group_id", np.arange(len(uniq), dtype=np.int32))
    parts["n_experiments"] = parts["group_id"].map(Counter(group_id)).astype(np.int64)
    parts["path"] = [
        group_path(org, row.get("ag_class", "NA"), row.get("antigen", "NA"),
                   row.get("ct_class", "NA"))
        for _, row in parts.iterrows()
    ]

    # A slug collision would append two different antigens into one BED and
    # nothing downstream could tell. Refuse rather than warn.
    clashes = parts["path"].value_counts()
    clashes = clashes[clashes > 1]
    if len(clashes):
        example = clashes.index[0]
        rows = parts[parts["path"] == example]
        raise SystemExit(
            "%d output paths are claimed by more than one group; e.g. %s is "
            "claimed by:\n%s\nWiden slugify() or add a field to --group-by."
            % (len(clashes), example, rows.to_string(index=False))
        )

    duplicated = sub["srx"].duplicated()
    if duplicated.any():
        # Same accession listed twice for one assembly. Keep the first and say
        # so -- silently letting pandas pick would make the mapping order-dependent.
        print("  note: %d duplicate accessions within %s, keeping first"
              % (int(duplicated.sum()), org), file=sys.stderr)
        keep = ~duplicated
        sub, group_id = sub[keep], group_id[keep]

    return sub["srx"].to_numpy().astype("U16"), group_id, parts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="project root holding meta/ and work/")
    parser.add_argument("--org", nargs="+", default=["hg38", "mm10"])
    parser.add_argument("--group-by", nargs="+", default=list(DEFAULT_GROUP_FIELDS),
                        choices=["ag_class", "antigen", "ct_class", "celltype"])
    args = parser.parse_args(argv)

    print("reading experiment list ...", flush=True)
    frame = read_experiment_list(args.root / "meta")
    print("  %d experiments across %d assemblies"
          % (len(frame), frame["org"].nunique()))

    for org in args.org:
        srx, group_id, groups = build_for_org(frame, org, args.group_by)
        out = args.root / "work" / "manifest" / org
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "srx.npy", srx)
        np.save(out / "group_id.npy", group_id)
        groups.to_csv(out / "groups.tsv", sep="\t", index=False)
        print("%-6s %7d experiments -> %5d groups  (%s)"
              % (org, len(srx), len(groups), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
