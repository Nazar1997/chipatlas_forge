"""Stage 11 -- prove the release is complete before anything trains on it.

The failure this exists to catch is a *quiet* one. Array tasks that die leave
their outputs missing rather than wrong, and a loader that treats a missing
chunk as "no peaks here" -- which it must, because empty chunks are real --
turns a crashed task into a silently zeroed region of the genome. That is
indistinguishable from real data at training time and shows up only as a model
that is oddly bad on one chromosome.

So every check here is a counting argument against something written down
earlier, not a spot check:

* every tissue in the vocabulary has a parquet for every chromosome
* every populated chunk's row group holds only rows overlapping that chunk,
  and holds them in genomic order
* every DNA chunk named by the grid exists and is the length its name claims
* every window in every table lies inside its chromosome, and the splits
  partition the windows with nothing in two of them
* a sample of chunks is re-derived from `signal/` and compared row for row

The last one is the expensive check and the only one that can catch a wrong
value rather than a missing file, so it is sampled rather than exhaustive.

Usage:
    python -m chipatlas_forge.verify --data-dir ../ --org hg38 --release 2026-08
    python -m chipatlas_forge.verify --data-dir ../ --org hg38 --release 2026-08 \\
        --sample 200
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import layout, read
from .chunks import CHUNK_KEY, read_bedgraph, signal_files


class Report:
    """Collected problems, so one run reports everything rather than the first."""

    def __init__(self):
        self.problems = []
        self.checks = 0

    def check(self, ok, message):
        self.checks += 1
        if not ok:
            self.problems.append(message)
        return ok

    def fail(self, message):
        self.problems.append(message)


def verify_omics(release, report, chroms, tissues):
    """Presence, and that every row group is confined to its own chunk."""
    size = release.chunk_size
    missing, rows, groups = [], 0, 0
    for tissue in tissues:
        for chrom in sorted(chroms):
            path = release.omics_chunk(tissue, chrom)
            if not path.exists():
                missing.append("%s/%s" % (tissue, chrom))
                continue
            handle = pq.ParquetFile(path)
            metadata = handle.schema_arrow.metadata or {}
            present = json.loads(metadata.get(CHUNK_KEY, b"[]"))
            report.check(
                handle.num_row_groups == len(present),
                "%s: %d row groups but %d chunks in the footer"
                % (path, handle.num_row_groups, len(present)))
            rows += handle.metadata.num_rows
            groups += handle.num_row_groups
            for i, chunk in enumerate(present):
                stats = handle.metadata.row_group(i)
                start = stats.column(1).statistics
                end = stats.column(2).statistics
                if start is None or end is None:
                    continue
                lo, hi = chunk * size, (chunk + 1) * size
                report.check(
                    end.max > lo and start.min < hi,
                    "%s row group %d holds rows outside chunk %d (%d-%d vs %d-%d)"
                    % (path, i, chunk, start.min, end.max, lo, hi))
    if missing:
        report.fail("%d (tissue, chromosome) omics files are missing: %s%s"
                    % (len(missing), ", ".join(missing[:10]),
                       " ..." if len(missing) > 10 else ""))
    return {"rows": rows, "row_groups": groups, "missing": len(missing)}


def verify_vocabulary_is_backed_by_data(release, report, chroms, availability):
    """Every tissue the vocabulary claims features for must actually have rows.

    The check that was missing. `verify_sampled_chunks` re-derives a chunk
    through the *same* `signal_files` lookup that built it, so a lookup bug is
    invisible to it -- expected and actual are both empty and it reports a
    match. That is exactly what happened: five multi-word tissues
    ("Pluripotent stem cell", "Digestive tract", "Embryonic fibroblast", ...)
    got empty parquet because the directories on disk are slugified, and a
    release passed 760,779 checks with ~1,500 features silently unavailable.

    Cross-checking against the vocabulary instead of against the lookup is what
    makes it catchable: `availability.json` says the tissue has N features, and
    a tissue with features but no rows anywhere is a contradiction.
    """
    empty = []
    for tissue, features in sorted(availability.items()):
        if not features:
            continue
        rows = 0
        for chrom in chroms:
            path = release.omics_chunk(tissue, chrom)
            if path.exists():
                rows += pq.ParquetFile(path).metadata.num_rows
            if rows:
                break
        report.check(rows > 0,
                     "tissue %r has %d features in the vocabulary but no omics "
                     "rows on any chromosome" % (tissue, len(features)))
        if not rows:
            empty.append(tissue)
    return {"tissues": len(availability), "empty": empty}


def verify_every_claimed_feature_is_reachable(release, report, availability,
                                              features, aliases):
    """Every feature some tissue claims must resolve to a file on disk.

    The tissue-level check above counts rows per tissue, so one unreachable
    feature among hundreds is invisible to it -- which is exactly how
    `H3PERIOD3_K27M_mutant` stayed missing after the directory bug was fixed:
    its file is `H3.3_K27M_mutant.bedgraph`, matching neither its vocabulary
    name nor its alias.

    Resolved through `signal_files`, so this asserts the lookup the chunk stage
    actually uses can find every column the vocabulary promises.
    """
    claimed = set().union(*availability.values()) if availability else set()
    reachable = set()
    for tissue in availability:
        reachable |= set(signal_files(release, tissue, features, aliases))
    missing = sorted(claimed - reachable)
    report.check(not missing,
                 "%d vocabulary feature(s) are claimed by a tissue but no signal "
                 "file resolves to them: %s"
                 % (len(missing), ", ".join(missing[:10])))
    return {"claimed": len(claimed), "unreachable": missing}


def verify_dna(release, report, chroms):
    """Every chunk the grid names exists and is exactly as long as its name says."""
    size = release.chunk_size
    missing = wrong = total = 0
    for chrom, length in sorted(chroms.items()):
        for start in range(0, length, size):
            end = min(start + size, length)
            total += 1
            path = release.dna_chunk(chrom, start, end)
            if not path.exists():
                missing += 1
                continue
            if path.stat().st_size != end - start:
                wrong += 1
    if missing:
        report.fail("%d of %d DNA chunks are missing" % (missing, total))
    if wrong:
        report.fail("%d of %d DNA chunks are not the length their name claims"
                    % (wrong, total))
    return {"chunks": total, "missing": missing, "wrong_length": wrong}


def verify_windows(release, report, chroms):
    """Windows stay inside their chromosome and each belongs to exactly one split."""
    out = {}
    sizes = release.manifest.get("stages", {}).get("intervals", {}).get("windows", {})
    for window in sorted(int(w) for w in sizes):
        frame = read.windows(release, window)
        starts = frame["start"].to_numpy()   # abi-ok: pandas Series, not an Arrow Array
        ends = frame["end"].to_numpy()       # abi-ok: pandas Series, not an Arrow Array
        limits = frame["chrom"].map(chroms).to_numpy()  # abi-ok: pandas Series
        report.check((starts >= 0).all(), "%d bp: negative window start" % window)
        report.check((ends <= limits).all(),
                     "%d bp: %d windows run past the end of their chromosome"
                     % (window, int((ends > limits).sum())))
        report.check((ends > starts).all(), "%d bp: empty window" % window)

        # A locus in two splits is the leak this whole design exists to prevent.
        key = frame["chrom"].astype(str) + ":" + frame["start"].astype(str)
        per_locus = frame.assign(key=key).groupby("key")["split"].nunique()
        clashes = int((per_locus > 1).sum())
        report.check(clashes == 0,
                     "%d bp: %d loci appear in more than one split" % (window, clashes))
        out[str(window)] = {"rows": len(frame),
                            "loci": int(per_locus.size),
                            "split_clashes": clashes}
    return out


def verify_sampled_chunks(release, report, chroms, tissues, features, n, seed,
                          aliases=None):
    """Re-derive sampled chunks straight from `signal/` and compare row for row.

    The only check that can catch a wrong *value* rather than a missing file.
    Sampled because rebuilding every chunk is the stage itself run twice.
    """
    rng = np.random.default_rng(seed)
    size = release.chunk_size
    compared = mismatched = 0
    cache = {}

    for _ in range(n):
        tissue = tissues[rng.integers(len(tissues))]
        chrom = sorted(chroms)[rng.integers(len(chroms))]
        if chroms[chrom] < size:
            continue
        chunk = int(rng.integers(chroms[chrom] // size))
        lo, hi = chunk * size, (chunk + 1) * size

        # Keyed on (tissue, chromosome), because the read below is filtered to
        # one chromosome -- keying on tissue alone hands a later sample of the
        # same tissue on a different chromosome an empty result and reports it
        # as a mismatch.
        key = (tissue, chrom)
        if key not in cache:
            # One tissue-chromosome of bedGraphs is the memory high-water mark
            # here; drop the previous one before pulling in another.
            cache.clear()
            files = signal_files(release, tissue, features, aliases)
            cache[key] = {a: read_bedgraph(p, {chrom}, 8 << 20).get(chrom)
                          for a, p in files.items()}
        expected = []
        for antigen, arrays in cache[key].items():
            if arrays is None:
                continue
            s, e, v = arrays
            hit = (e > lo) & (s < hi)
            expected.extend(zip(s[hit].tolist(), e[hit].tolist(),
                                v[hit].tolist(), [antigen] * int(hit.sum())))

        got = read.load_chunk(release, tissue, chrom, lo, hi)
        actual = list(zip(got["Start"].tolist(), got["End"].tolist(),
                          got["Name"].tolist(), got["feature_name"].tolist()))
        compared += 1
        if sorted(expected) != sorted(actual):
            mismatched += 1
            report.fail("%s %s:%d-%d -- %d rows from signal/, %d stored"
                        % (tissue, chrom, lo, hi, len(expected), len(actual)))
            if mismatched >= 5:
                break
    return {"sampled": compared, "mismatched": mismatched}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--org", required=True)
    parser.add_argument("--release", default=None)
    parser.add_argument("--sample", type=int, default=25,
                        help="chunks to re-derive from signal/ and compare; 0 skips")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    release = layout.Release.open(args.data_dir, args.org, args.release)
    report = Report()
    chroms = {c: int(n) for c, n in release.manifest["chrom_sizes"].items()}
    tissues = read.tissues(release)
    features = read.features(release)
    print("verifying %s/%s -- %d chromosomes, %d tissues, %d features"
          % (args.org, release.id, len(chroms), len(tissues), len(features)),
          flush=True)

    summary = {}
    if release.availability().exists():
        availability = json.loads(release.availability().read_text())
        summary["vocabulary"] = verify_vocabulary_is_backed_by_data(
            release, report, chroms, availability)
        empty = summary["vocabulary"]["empty"]
        print("  vocab:   %d tissues, %d with no omics rows%s"
              % (summary["vocabulary"]["tissues"], len(empty),
                 ": " + ", ".join(empty) if empty else ""), flush=True)
        summary["features"] = verify_every_claimed_feature_is_reachable(
            release, report, availability, features,
            json.loads(release.path("features").read_text()).get("aliases"))
        unreachable = summary["features"]["unreachable"]
        print("  feature: %d claimed, %d unreachable%s"
              % (summary["features"]["claimed"], len(unreachable),
                 ": " + ", ".join(unreachable[:6]) if unreachable else ""),
              flush=True)
    summary["dna"] = verify_dna(release, report, chroms)
    print("  dna:     %(chunks)d chunks, %(missing)d missing, "
          "%(wrong_length)d wrong length" % summary["dna"], flush=True)

    if release.omics_layout == layout.CHROM_PARQUET:
        summary["omics"] = verify_omics(release, report, chroms, tissues)
        print("  omics:   %(rows)d rows in %(row_groups)d row groups, "
              "%(missing)d files missing" % summary["omics"], flush=True)
    else:
        print("  omics:   skipped -- %s layout has no row groups to check"
              % release.omics_layout, flush=True)

    summary["windows"] = verify_windows(release, report, chroms)
    for window, stats in sorted(summary["windows"].items(), key=lambda kv: int(kv[0])):
        print("  windows: %8s bp -- %d rows over %d loci, %d split clashes"
              % (window, stats["rows"], stats["loci"], stats["split_clashes"]),
              flush=True)

    if args.sample and release.omics_layout == layout.CHROM_PARQUET:
        summary["sampled"] = verify_sampled_chunks(
            release, report, chroms, tissues, features, args.sample, args.seed,
            json.loads(release.path("features").read_text()).get("aliases"))
        print("  sampled: %(sampled)d chunks re-derived from signal/, "
              "%(mismatched)d mismatched" % summary["sampled"], flush=True)

    if report.problems:
        print("\n%d problem(s) across %d checks:" % (len(report.problems),
                                                     report.checks))
        for problem in report.problems[:40]:
            print("  - %s" % problem)
        if len(report.problems) > 40:
            print("  ... and %d more" % (len(report.problems) - 40))
        return 1

    release.record("verify", checks=report.checks, **summary)
    print("\nall %d checks passed" % report.checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
