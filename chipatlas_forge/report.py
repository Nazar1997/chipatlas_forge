"""Post-run audit: did every peak survive, and is the output actually sorted?

Worth running every time. The pipeline's two silent failure modes are peaks
vanishing (an accession missing from the manifest, a stage-2 task that died and
left its shard unrouted) and output that is not in genomic order (which nothing
downstream would notice until a bedtools call gave wrong answers). Both are
cheap to check and neither shows up in a SLURM exit code.

Usage:
    python -m chipatlas_forge.report --root . --org hg38
"""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def _load(directory: Path) -> list:
    if not directory.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]


def check_sorted(path: Path, limit: int) -> str:
    """Confirm one output BED is ordered by (chrom, start) within each chrom."""
    script = (
        'BEGIN{pc="";ps=-1;bad=0} '
        '{ if($1==pc){ if($2+0 < ps){bad++} } else {pc=$1} ps=$2+0 } '
        'END{print bad+0, NR}'
    )
    opener = "zcat" if path.suffix == ".gz" else "cat"
    out = subprocess.run(
        ["bash", "-c", "%s %s | head -%d | awk '%s'" % (opener, path, limit, script)],
        capture_output=True, text=True,
    )
    bad, rows = (out.stdout.split() + ["?", "?"])[:2]
    return "%s rows=%s out-of-order=%s" % (path.name, rows, bad)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--org", required=True)
    parser.add_argument("--spot-check", type=int, default=5,
                        help="how many output BEDs to verify are sorted")
    parser.add_argument("--check-lines", type=int, default=2_000_000)
    args = parser.parse_args(argv)

    root, org = args.root, args.org
    routed = _load(root / "work" / "stats" / org)
    collected = _load(root / "work" / "collect_stats" / org)

    if not routed:
        raise SystemExit("no stage-2 stats for %s; has route run?" % org)

    shards = json.loads((root / "work" / "shards" / org / "shards.json").read_text())
    rows_in = sum(s["rows"] for s in routed)
    unmapped = sum(s["unmapped"] for s in routed)
    bad = sum(s["bad_rows"] for s in routed)
    rows_out = sum(s["rows"] for s in collected)

    print("=== %s ===" % org)
    print("shards            : %d expected, %d routed"
          % (shards["n_shards"], len(routed)))
    if len(routed) != shards["n_shards"]:
        print("  !! %d shards were never routed -- rerun those array tasks"
              % (shards["n_shards"] - len(routed)))
    print("peaks read        : %15d" % rows_in)
    print("  unmapped        : %15d  (%.4f%%)"
          % (unmapped, 100.0 * unmapped / max(rows_in, 1)))
    print("  malformed rows  : %15d" % bad)
    print("peaks written     : %15d" % rows_out)

    expected = rows_in - unmapped
    if collected:
        delta = rows_out - expected
        verdict = "OK" if delta == 0 else "!! %+d" % delta
        print("conservation      : read-unmapped == written  %s" % verdict)
        print("buckets           : %d collected, %d groups written"
              % (len(collected), sum(c["groups_written"] for c in collected)))

    groups = pd.read_csv(root / "work" / "manifest" / org / "groups.tsv", sep="\t")
    print("groups defined    : %d" % len(groups))

    out_root = root / "out"
    beds = sorted(out_root.rglob("*.bed")) + sorted(out_root.rglob("*.bed.gz"))
    beds = [b for b in beds if b.parts[len(out_root.parts)] == org]
    if beds:
        sizes = sorted(((b.stat().st_size, b) for b in beds), reverse=True)
        total = sum(s for s, _ in sizes)
        print("output files      : %d, %.1f GB" % (len(beds), total / 1e9))
        print("largest:")
        for size, path in sizes[:8]:
            print("  %8.2f GB  %s" % (size / 1e9, path.relative_to(out_root)))
        print("sortedness spot-check (first %d lines):" % args.check_lines)
        for _, path in sizes[:args.spot_check]:
            print("  " + check_sorted(path, args.check_lines))
    else:
        print("output files      : none yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
