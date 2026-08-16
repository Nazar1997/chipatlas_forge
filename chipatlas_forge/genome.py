"""Stage 7 -- put the reference sequence and the blacklist into the release.

The genome is the one input that does not change between releases: hg38 is
hg38 whichever ChIP-Atlas snapshot the peaks came from. So this stage does not
download anything by default -- it adopts the files an existing release (or the
pre-release tree) already has.

**Hard links, not copies.** The reference is 3.1 GB and `sequence.pkl` another
3.2 GB; every release that copied them would cost 6 GB for bytes that are
identical. A hard link is free, instant, and -- because nothing here ever
rewrites these files in place, only replaces them -- there is no way for one
release to mutate another's genome through the shared inode. Links fall back to
a copy across filesystems, which is the only case where they cannot work.

Chromosome lengths are derived here rather than hardcoded. They used to live in
`src/config.py` as a literal dict per assembly, which is a second source of
truth for something the FASTA already states, and silently wrong for any
assembly nobody remembered to add.

Usage:
    python -m chipatlas_forge.genome --data-dir ../ --org hg38 --release 2026-08 \\
        --fasta ../hg38/DNA/hg38.fa --blacklist ../hg38/DNA/hg38-blacklist.v2.bed \\
        --sequence ../hg38/SupportFiles/hg38_DNA_seq.pkl
"""

import argparse
import os
import pickle
import shutil
from pathlib import Path

from . import layout


def adopt(source, target):
    """Hard-link ``source`` to ``target``, copying only if that is impossible.

    Returns the strategy used, so the manifest can record whether a release
    owns its genome bytes or shares them.
    """
    source = Path(source).resolve()
    if not source.exists():
        raise SystemExit("%s does not exist" % source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.exists() and target.samefile(source):
            return "already linked"
        target.unlink()
    try:
        os.link(source, target)
        return "hard link"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def chrom_sizes_from_fasta(path):
    """``{chrom: length}`` from a FASTA, without holding a sequence in memory.

    Counts residue bytes per record rather than reading sequences: the file is
    3.1 GB and the answer is 24 integers.
    """
    sizes, name, total = {}, None, 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                if name is not None:
                    sizes[name] = total
                name = line[1:].split()[0].decode("ascii")
                total = 0
            else:
                total += len(line.strip())
    if name is not None:
        sizes[name] = total
    return sizes


def chrom_sizes_from_sequence(path):
    """``{chrom: length}`` from the ``{chrom: str}`` pickle, if that is all we have."""
    with open(path, "rb") as fh:
        seq = pickle.load(fh)
    return {str(k): len(v) for k, v in seq.items()}


def write_chrom_sizes(sizes, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))
    path.write_text("".join("%s\t%d\n" % (c, n) for c, n in ordered))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--org", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--fasta", type=Path, default=None)
    parser.add_argument("--blacklist", type=Path, default=None)
    parser.add_argument("--sequence", type=Path, default=None,
                        help="the {chrom: sequence} pickle the chunk stage reads")
    parser.add_argument("--from-release", default=None,
                        help="adopt genome/ from another release of this organism "
                             "instead of naming each file")
    parser.add_argument("--keep-chroms", nargs="*", default=None,
                        help="restrict the recorded chromosomes; default keeps "
                             "everything the FASTA has that is not a scaffold")
    args = parser.parse_args(argv)

    release = layout.Release.open(args.data_dir, args.org, args.release)

    fasta, blacklist, sequence = args.fasta, args.blacklist, args.sequence
    if args.from_release:
        donor = layout.Release.open(args.data_dir, args.org, args.from_release)
        fasta = fasta or donor.path("genome_fasta")
        blacklist = blacklist or donor.path("blacklist")
        sequence = sequence or donor.path("sequence")

    if fasta is None and sequence is None:
        raise SystemExit(
            "nothing to adopt: pass --fasta and/or --sequence, or --from-release")

    how = {}
    if fasta is not None:
        how["genome.fa"] = adopt(fasta, release.path("genome_fasta"))
    if blacklist is not None:
        how["blacklist.bed"] = adopt(blacklist, release.path("blacklist"))
    if sequence is not None:
        how["sequence.pkl"] = adopt(sequence, release.path("sequence"))
    for name, strategy in sorted(how.items()):
        print("  %-16s %s" % (name, strategy), flush=True)

    if fasta is not None:
        sizes = chrom_sizes_from_fasta(release.path("genome_fasta"))
    else:
        sizes = chrom_sizes_from_sequence(release.path("sequence"))

    # Unplaced scaffolds and alt haplotypes carry almost no ChIP-Atlas signal
    # and would each add a chunk directory; "chr" + digits/X/Y/M is the set the
    # peak files actually use.
    if args.keep_chroms:
        sizes = {c: n for c, n in sizes.items() if c in set(args.keep_chroms)}
    else:
        sizes = {c: n for c, n in sizes.items() if is_primary(c)}
    if not sizes:
        raise SystemExit("no chromosomes survived filtering -- check --keep-chroms")

    write_chrom_sizes(sizes, release.path("chrom_sizes"))
    release.manifest["chrom_sizes"] = dict(sorted(sizes.items()))
    release.record("genome", n_chroms=len(sizes), total_bp=sum(sizes.values()),
                   adopted=how)
    print("%d chromosomes, %.2f Gb" % (len(sizes), sum(sizes.values()) / 1e9),
          flush=True)
    return 0


def is_primary(chrom):
    """A real chromosome rather than a scaffold, patch or alt haplotype."""
    if not chrom.startswith("chr"):
        return False
    rest = chrom[3:]
    return rest.isdigit() or rest in ("X", "Y", "M", "MT")


if __name__ == "__main__":
    raise SystemExit(main())
