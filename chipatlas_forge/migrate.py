"""Turn the pre-release `data/<org>/` tree into a release, by moving it.

The 2021 tree predates all of this: `DNA/`, `OMICS/`, `Subtables/`,
`SupportFiles/`, with ChIP-Atlas's three-letter codes for directory names
(`OMICS/His/Bld/05.H3K27ac.AllCell.bed`). It is also the data every existing
checkpoint was trained against, so it cannot simply be deleted and rebuilt --
its splits and its vocabulary are the definition of what those checkpoints mean.

So it becomes a release like any other, and keeps its contents. What changes is
the *names*: `Bld` becomes `Blood`, `His` becomes `Histone`, and the four
top-level directories become the five a release has. What does not change is any
of the data -- the 900k omics pickles and the 4.9 GB interval pickles are moved,
not rewritten, and the manifest declares the layouts they are in so a reader
handles them without converting anything.

**Everything is `os.rename`.** Within one filesystem that is a directory-entry
update whatever the subtree weighs, so moving 152 GB and 900,000 files takes the
same instant as moving one. The stage refuses to start if the source and
destination are on different devices, where rename would silently become a
152 GB copy.

Destructive and one-way, so it prints what it would do and stops. Pass
`--execute` to actually move anything.

Usage:
    python -m chipatlas_forge.migrate --data-dir ../ --org hg38 --release 2021-10
    python -m chipatlas_forge.migrate --data-dir ../ --org hg38 --release 2021-10 --execute
"""

import argparse
import json
import os
import re
from pathlib import Path

from . import layout
from .vocab import read_class_codes

# `OMICS/<ag code>/<ct code>/<threshold>.<antigen>.AllCell.bed`
OMICS_FILE = re.compile(r"^(\d+)\.(.+)\.AllCell\.bed$")

# Interval pickles, by the shape of their name. `full`/`train`/`test` cross
# window size and the `_ALL` / `_subset` variants.
INTERVAL_FILE = re.compile(
    r"^(full|train|test)_intervals(?:_(\d+))?(_ALL|_subset)?\.pkl$")


def plan_genome(src, org, release):
    """FASTA, blacklist and the sequence pickle."""
    moves = []
    for name, key in ((f"{org}.fa", "genome_fasta"),
                      (f"{org}-blacklist.v2.bed", "blacklist")):
        path = src / "DNA" / name
        if path.exists():
            moves.append((path, release.path(key)))
    seq = src / "SupportFiles" / f"{org}_DNA_seq.pkl"
    if seq.exists():
        moves.append((seq, release.path("sequence")))
    return moves


def plan_signal(src, release, ag_names, ct_names):
    """`OMICS/His/Bld/05.H3K27ac.AllCell.bed` -> `signal/Histone/Blood/H3K27ac.bedgraph`.

    Per-file rather than per-directory because both levels are renamed and the
    leaf carries a threshold prefix and an `AllCell` suffix that the new name
    drops. The threshold is recorded once in the manifest instead of being
    repeated in 8,487 filenames.
    """
    moves, thresholds, unknown = [], set(), set()
    root = src / "OMICS"
    if not root.is_dir():
        return moves, thresholds, unknown
    for ag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ag_class = ag_names.get(ag_dir.name)
        if ag_class is None:
            unknown.add("antigen class %r" % ag_dir.name)
            ag_class = ag_dir.name
        for ct_dir in sorted(p for p in ag_dir.iterdir() if p.is_dir()):
            tissue = ct_names.get(ct_dir.name)
            if tissue is None:
                unknown.add("cell-type class %r" % ct_dir.name)
                tissue = ct_dir.name
            for bed in sorted(ct_dir.glob("*.bed")):
                match = OMICS_FILE.match(bed.name)
                if not match:
                    unknown.add("file name %r" % bed.name)
                    continue
                thresholds.add(match.group(1))
                moves.append((bed, release.signal_path(ag_class, tissue,
                                                       match.group(2))))
    return moves, thresholds, unknown


def plan_chunks(src, release, ct_names):
    """Whole-directory renames: `Subtables/dna` and each `Subtables/omics/<code>`.

    One rename per tissue rather than per file. The 900,000 omics pickles never
    appear in the plan at all -- they ride along inside their tissue directory,
    which is the only reason this stage is instant.
    """
    moves = []
    dna = src / "Subtables" / "dna"
    if dna.is_dir():
        moves.append((dna, release.root / "chunks" / "dna"))
    omics = src / "Subtables" / "omics"
    if omics.is_dir():
        for tissue_dir in sorted(p for p in omics.iterdir() if p.is_dir()):
            tissue = ct_names.get(tissue_dir.name, tissue_dir.name)
            moves.append((tissue_dir, release.root / "chunks" / "omics" / tissue))
    return moves


def plan_index(src, release):
    """Interval pickles and the pair index, into `index/` under their own names."""
    moves = []
    support = src / "SupportFiles"
    if not support.is_dir():
        return moves
    index = release.path("chunk_grid").parent
    for path in sorted(support.glob("*.pkl")):
        if INTERVAL_FILE.match(path.name) or path.name == "all_intervals.pkl":
            moves.append((path, index / path.name))
    for name in ("omics_pairs.npz", "track_weights.npz"):
        if (support / name).exists():
            moves.append((support / name, index / name))
    return moves


def build_vocab_files(src, release, ct_names):
    """`target_*.pkl` and `Subtables/avl/*.pkl` -> tissues.json, features.json,
    availability.json, with the three-letter codes spelled out.

    Read and rewritten rather than moved: these are the only legacy artifacts
    whose *content* is a set of tissue names, so leaving them as codes would
    make the old release the one place a reader still had to know that `Bld`
    means blood.
    """
    from joblib import load

    support = src / "SupportFiles"
    features = [str(f) for f in load(support / "target_features.pkl")]
    codes = [str(t) for t in load(support / "target_tissues.pkl")]

    availability = {}
    avl_dir = src / "Subtables" / "avl"
    for code in codes:
        path = avl_dir / ("%s.pkl" % code)
        name = ct_names.get(code, code)
        availability[name] = sorted(str(f) for f in load(path)) if path.exists() else []

    index = release.path("tissues").parent
    index.mkdir(parents=True, exist_ok=True)
    release.path("tissues").write_text(json.dumps({
        "tissues": [{"name": ct_names.get(c, c), "code": c,
                     "n_features": len(availability[ct_names.get(c, c)])}
                    for c in codes],
    }, indent=2) + "\n")
    release.path("features").write_text(json.dumps({
        "features": features, "frozen_from": None,
        "absent_from_data": [], "excluded_by_freeze": [],
    }, indent=2) + "\n")
    release.availability().write_text(json.dumps(availability, indent=2) + "\n")
    return features, codes


def chrom_sizes_from_chunks(release):
    """Chromosome lengths read off the DNA chunk names.

    The last chunk of a chromosome is `<start>_<length>.txt`, so the largest end
    is the length. Cheaper and no less exact than the alternatives: the FASTA is
    3.2 GB to scan and `sequence.pkl` is 960 MB to unpickle, for 24 integers a
    directory listing already contains.
    """
    root = release.root / "chunks" / "dna"
    sizes = {}
    if not root.is_dir():
        return sizes
    for chrom_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ends = [int(p.stem.split("_")[1]) for p in chrom_dir.glob("*.txt")
                if "_" in p.stem]
        if ends:
            sizes[chrom_dir.name] = max(ends)
    return dict(sorted(sizes.items()))


def same_device(a, b):
    """Whether a rename between these two paths stays within one filesystem."""
    probe = b
    while not probe.exists():
        probe = probe.parent
    return os.stat(a).st_dev == os.stat(probe).st_dev


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--org", required=True)
    parser.add_argument("--release", required=True,
                        help="the id to file the existing tree under, e.g. 2021-10")
    parser.add_argument("--meta-dir", type=Path, default=Path("meta"),
                        help="forge's meta/, for fileList.tab's code mapping")
    parser.add_argument("--execute", action="store_true",
                        help="actually move files; without it nothing is touched")
    args = parser.parse_args(argv)

    src = Path(args.data_dir) / args.org
    if not src.is_dir():
        raise SystemExit("%s does not exist" % src)
    if (src / "releases" / args.release).exists():
        raise SystemExit(
            "%s already exists -- migration is one-way and will not overwrite"
            % (src / "releases" / args.release))

    ag_names, ct_names = {}, {}
    ag_codes, ct_codes = read_class_codes(args.meta_dir)
    if not ct_codes:
        raise SystemExit(
            "%s/fileList.tab is missing -- it is what maps `Bld` to `Blood`, and "
            "without it every directory would keep its three-letter name"
            % args.meta_dir)
    ag_names = {code: name for name, code in ag_codes.items()}
    ct_names = {code: name for name, code in ct_codes.items()}

    release = layout.Release.create(
        args.data_dir, args.org, args.release,
        omics_layout=layout.CHUNKED_PICKLE,
        interval_layout=layout.PER_SPLIT_PICKLE,
        note="the pre-release tree, moved and renamed by chipatlas_forge.migrate")

    moves = list(plan_genome(src, args.org, release))
    signal, thresholds, unknown = plan_signal(src, release, ag_names, ct_names)
    moves += signal
    moves += plan_chunks(src, release, ct_names)
    moves += plan_index(src, release)

    if unknown:
        print("unrecognised names (kept verbatim): %s"
              % ", ".join(sorted(unknown)), flush=True)
    if len(thresholds) > 1:
        raise SystemExit("OMICS mixes peak thresholds %s -- refusing to flatten "
                         "them into one name" % sorted(thresholds))

    total = sum(1 for _ in moves)
    print("%d moves planned for %s -> %s" % (total, src, release.root))
    for source, target in moves[:6]:
        print("  %s\n    -> %s" % (source.relative_to(src),
                                   target.relative_to(release.root)))
    if total > 6:
        print("  ... and %d more" % (total - 6))

    if not args.execute:
        # The release directory was created to resolve the target paths; leave
        # nothing behind after a dry run.
        (release.root / "MANIFEST.json").unlink()
        try:
            release.root.rmdir()
            release.root.parent.rmdir()
        except OSError:
            pass
        print("\ndry run -- nothing moved. Re-run with --execute.")
        return 0

    for source, target in moves:
        if not same_device(source, target):
            raise SystemExit(
                "%s and %s are on different filesystems; rename would become a "
                "full copy. Move the tree yourself, then re-run." % (source, target))

    done = 0
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
        done += 1
        if done % 2000 == 0:
            print("  %d/%d" % (done, total), flush=True)

    features, codes = build_vocab_files(src, release, ct_names)

    sizes = chrom_sizes_from_chunks(release)
    if not sizes:
        raise SystemExit(
            "no DNA chunks under %s -- cannot determine chromosome lengths"
            % (release.root / "chunks" / "dna"))
    release.manifest["chrom_sizes"] = sizes
    release.manifest["source_threshold"] = thresholds.pop() if thresholds else None
    release.manifest["class_codes"] = {"antigen": ag_codes, "cell_type": ct_codes}
    release.record("migrate", moved=done, n_features=len(features),
                   n_tissues=len(codes))
    for stage in ("genome", "vocab", "intervals", "chunks_dna", "chunks_omics"):
        release.record(stage, inherited="migrated from the pre-release tree")

    leftovers = sorted(p.name for p in src.iterdir()
                       if p.name not in ("releases", "latest"))
    print("\nmoved %d entries into %s" % (done, release.root))
    if leftovers:
        print("left in place (empty or unrecognised): %s" % ", ".join(leftovers))
    print("promote it with:  python -m chipatlas_forge.layout  # or layout.promote()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
