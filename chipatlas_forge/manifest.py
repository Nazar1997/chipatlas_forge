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


# Everything here reads as latin-1, never utf-8. The ChIP-Atlas metadata carries
# cp1252 smart quotes in free-text titles (byte 0x91 at offset 6899 of the 2021
# CSV, for one) and utf-8 decoding raises on them. latin-1 cannot fail -- every
# byte is a valid code point -- and the six columns kept here are ASCII in
# practice, so nothing is mangled. errors="replace" would instead corrupt an
# antigen name silently.
ENCODING = "latin-1"

LIVE_LIST = "experimentList.tab"
BUNDLED_ZIP = "chip_atlas_experiment_list.zip"


def read_live_list(path: Path) -> pd.DataFrame:
    """Parse ChIP-Atlas's canonical experimentList.tab.

    Fetch or refresh it with:
        curl -o meta/experimentList.tab \\
             https://chip-atlas.dbcls.jp/data/metadata/experimentList.tab

    Split manually rather than with pandas because the file is *ragged*: it has
    no header, and the trailing free-text metadata column contains tabs of its
    own, so rows have varying field counts and any parser told to expect six
    columns will either error or misalign. ``split("\\t", 6)`` stops after the
    six fields that matter and leaves the rest as one blob.
    """
    rows = []
    with open(path, encoding=ENCODING) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", len(META_COLUMNS))
            if len(parts) >= len(META_COLUMNS):
                rows.append(parts[:len(META_COLUMNS)])
    return pd.DataFrame(rows, columns=META_COLUMNS)


def read_bundled_zip(path: Path) -> pd.DataFrame:
    """The six identifying columns of the bundled chip_atlas_experiment_list.zip.

    Read straight out of the zip -- the CSV expands to 450 MB and nothing else
    needs it on disk.
    """
    with zipfile.ZipFile(path) as zf:
        inner = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(inner) != 1:
            raise SystemExit("expected one CSV in %s, found %r" % (path, inner))
        with zf.open(inner[0]) as fh:
            return pd.read_csv(
                fh, usecols=range(len(META_COLUMNS)), names=META_COLUMNS,
                header=0, dtype=str, keep_default_na=False,
                encoding=ENCODING, engine="c", on_bad_lines="warn",
            )


def read_experiment_list(meta_dir: Path, source: str = "auto") -> pd.DataFrame:
    """Accession metadata, preferring the live list over the bundled zip.

    The zip that ships in the Yandex share is stamped **October 2021** and is
    badly out of date against the peak archives: 3.2% of hg38 peaks cite
    accessions it has never heard of, which at full scale is ~48 million peaks
    dropped. The live experimentList.tab (2025-10-01) carries 845,824
    experiments against the zip's 439,593 -- 197,044 for hg38 rather than
    103,765 -- and closes that gap.

    So `auto` takes the live file when it is present and falls back to the zip
    with a warning when it is not.
    """
    live, bundled = meta_dir / LIVE_LIST, meta_dir / BUNDLED_ZIP

    if source in ("auto", "live") and live.exists():
        print("  source: %s (%.0f MB)" % (live.name, live.stat().st_size / 1e6))
        return read_live_list(live)
    if source == "live":
        raise SystemExit("missing %s; download it from chip-atlas.dbcls.jp" % live)

    if not bundled.exists():
        raise SystemExit("no metadata in %s -- expected %s or %s"
                         % (meta_dir, LIVE_LIST, BUNDLED_ZIP))
    print("  source: %s -- this snapshot is from 2021 and misses ~3%% of "
          "accessions in current peak archives; prefer %s"
          % (bundled.name, LIVE_LIST), file=sys.stderr)
    return read_bundled_zip(bundled)


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
    parser.add_argument("--meta-source", choices=["auto", "live", "bundled"],
                        default="auto",
                        help="'live' requires meta/experimentList.tab; 'bundled' "
                             "forces the stale 2021 zip")
    args = parser.parse_args(argv)

    print("reading experiment list ...", flush=True)
    frame = read_experiment_list(args.root / "meta", args.meta_source)
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
