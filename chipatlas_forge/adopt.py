"""Stage 5b -- the seam between the peak pipeline and a release.

Stages 0-5 work inside forge's own tree (`work/`, `out/`, `out_binmax1/`),
because they are a build: intermediates, rerunnable, safe to delete. A release
is the opposite -- training input, immutable once promoted, and it lives in
`data/` rather than in this repository. This stage carries the two outputs that
cross that line:

    out_binmax<n>/<org>/<ag class>/<tissue>/<antigen>.bedgraph
        -> <release>/signal/<ag class>/<tissue>/<antigen>.bedgraph
    work/manifest/<org>/groups.tsv
        -> <release>/index/groups.tsv

Hard links, so 18.8 GB of signal costs nothing and two releases built from the
same peaks share the bytes. Nothing here rewrites a file in place, so there is
no way for one release to mutate another's signal through a shared inode.

Copying `groups.tsv` in matters more than it looks: it is what `vocab` reads to
decide the vocabulary, and having it inside the release means the release
records the catalogue it was selected from, rather than depending on a working
directory that the next pipeline run will overwrite.

Usage:
    python -m chipatlas_forge.adopt --root . --data-dir ../ --org hg38 \\
        --release 2026-08 --binmax out_binmax1
"""

import argparse
from pathlib import Path

from . import layout
from .genome import adopt as adopt_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="forge's working tree, holding out_binmax*/ and work/")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--org", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--binmax", default="out_binmax1",
                        help="which binmax run to adopt as this release's signal")
    parser.add_argument("--suffix", default=".bedgraph")
    args = parser.parse_args(argv)

    source = args.root / args.binmax / args.org
    if not source.is_dir():
        raise SystemExit(
            "%s does not exist -- run stages 1-5 for %s first" % (source, args.org))

    # `create` rather than `open`: this is the first stage of a release, so it
    # is also what brings the release into being.
    root = layout.release_root(args.data_dir, args.org, args.release)
    if (root / "MANIFEST.json").exists():
        release = layout.Release.open(args.data_dir, args.org, args.release)
    else:
        release = layout.Release.create(args.data_dir, args.org, args.release)

    tracks = sorted(source.rglob("*" + args.suffix))
    if not tracks:
        raise SystemExit("no %s files under %s" % (args.suffix, source))

    signal_root = release.path("signal_root")
    how = {}
    total = 0
    for track in tracks:
        strategy = adopt_file(track, signal_root / track.relative_to(source))
        how[strategy] = how.get(strategy, 0) + 1
        total += track.stat().st_size

    groups = args.root / "work" / "manifest" / args.org / "groups.tsv"
    if not groups.exists():
        raise SystemExit(
            "%s is missing -- vocab reads it to choose the vocabulary" % groups)
    adopt_file(groups, release.path("groups"))

    release.manifest["binmax_source"] = args.binmax
    release.record("adopt", tracks=len(tracks), bytes=total,
                   strategies=how)
    print("adopted %d tracks (%.1f GB) and groups.tsv into %s  [%s]"
          % (len(tracks), total / 1e9, release.root,
             ", ".join("%d %s" % (n, s) for s, n in sorted(how.items()))),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
