"""Stage 9 -- cut the genome and the signal into per-chunk pieces a loader can seek.

Training reads a window of a few thousand to a million bases and needs, for that
window only, the DNA and every omics peak overlapping it. Neither a 3.1 GB FASTA
nor a 4 GB bedGraph answers that without a full scan, so both are cut on the
64 kb chunk grid.

DNA becomes one plain text file per chunk. Omics becomes **one parquet per
(tissue, chromosome), with a row group per chunk** -- 475 files per organism
rather than the ~900,000 pickles the 2021 tree used. Both give O(1) access to a
chunk; the difference is that Lustre charges per file open, and 900k tiny files
is a metadata storm that makes `du` time out and a cold epoch crawl. A row group
is seekable on its own, so nothing is given up: the footer says which row group
holds which chunk and the reader fetches exactly that byte range.

**A peak is stored in every chunk it overlaps, uncut.** That is what
`create_feature_matrix` expects -- it clips to the window itself and
de-duplicates on (Chromosome, Start, End, feature_name) when a window spans more
than one chunk. Storing clipped copies instead would silently shorten any peak
crossing a boundary, which is the exact class of edge artifact that was found
and fixed on the read side.

**Parallelised by tissue, not by (tissue, chromosome).** A task owns every
bedGraph for one tissue and streams each exactly once, splitting rows into
per-chromosome buffers as it goes. Sharding by chromosome as well would look
more parallel and read all 18.8 GB twenty-five times over.

Usage:
    python -m chipatlas_forge.chunks --data-dir ../ --org hg38 --release 2026-08 \\
        --what omics --task 3 --tasks 19
    python -m chipatlas_forge.chunks --data-dir ../ --org hg38 --release 2026-08 \\
        --what dna --from-release 2021-10
"""

import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from . import arrow_compat as compat
from . import layout
from .genome import adopt

BEDGRAPH_COLUMNS = ["chrom", "start", "end", "value"]

# Parquet footer key holding the chunk index of each row group, in order. The
# alternative is scanning row-group statistics at open time; an explicit list
# makes chunk -> row group a dict lookup and survives a writer that reorders.
CHUNK_KEY = b"chipatlas_forge:chunks"


def out_schema():
    """The frame the training loader expects back.

    pyranges' column spelling, kept deliberately: `create_feature_matrix` and
    `table_to_sparse` address these by name (`table.Name.values`, and the
    de-duplication key), so renaming them is a change to the read path rather
    than to the data. `Name` holds the max-score value, which the reader scales
    by 1/1000 into [0, 1].
    """
    return pa.schema([
        ("Chromosome", pa.dictionary(pa.int32(), pa.string())),
        ("Start", pa.int64()),
        ("End", pa.int64()),
        ("Name", pa.int32()),
        ("feature_name", pa.dictionary(pa.int32(), pa.string())),
    ])


def signal_files(release, tissue, features, aliases=None):
    """Every bedGraph for one tissue whose antigen is in the vocabulary.

    Walks `signal/<ag class>/<tissue>/<antigen>.bedgraph` across all antigen
    classes, so a tissue's histone marks and its transcription factors arrive
    together as one flat feature list -- the class is a grouping for humans, not
    a dimension of the target matrix.

    Keys are **vocabulary** names, not filenames. `aliases` maps the two apart
    for features whose 2021 name was mangled (`H2APERIODX` for `H2A.X`), so the
    file is found under its real name and the peaks are still filed under the
    column the model has. Without it those columns silently stay empty.
    """
    root = release.path("signal_root")
    if not root.is_dir():
        raise SystemExit("%s does not exist -- run binmax into the release first"
                         % root)
    # filename stem -> vocabulary name
    from_file = {name: name for name in features}
    for name, data_name in (aliases or {}).items():
        from_file[data_name] = name
    found = {}
    for ag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        path = ag_dir / tissue
        if not path.is_dir():
            continue
        for bedgraph in sorted(path.glob("*.bedgraph")):
            name = from_file.get(bedgraph.stem)
            if name is not None:
                # An antigen can appear under two classes; either file is the
                # same measurement, so first wins and the duplicate is skipped
                # rather than concatenated into a doubled track.
                found.setdefault(name, bedgraph)
    return found


def read_bedgraph(path, chrom_filter, block_size):
    """``{chrom: (starts, ends, values)}`` for one track, read once.

    Streamed in batches rather than read whole: a single tissue's tracks come to
    gigabytes and only the per-chromosome arrays need to survive the read.
    """
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=BEDGRAPH_COLUMNS,
                                       use_threads=True, block_size=block_size),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(column_types={
            "chrom": pa.dictionary(pa.int32(), pa.string()),
            "start": pa.int64(), "end": pa.int64(), "value": pa.int32(),
        }),
    )
    out = defaultdict(lambda: ([], [], []))
    try:
        for batch in reader:
            column = batch.column(0)
            names = column.dictionary.to_pylist()
            codes = compat.to_numpy(column.indices, np.int32)
            starts = compat.to_numpy(batch.column(1), np.int64)
            ends = compat.to_numpy(batch.column(2), np.int64)
            values = compat.to_numpy(batch.column(3), np.int32)
            # Files are grouped by chromosome, so a batch holds at most a few
            # contiguous runs and the split is a couple of slices.
            bounds = np.flatnonzero(np.diff(codes)) + 1
            for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(codes)]):
                name = names[codes[a]]
                if chrom_filter and name not in chrom_filter:
                    continue
                buf = out[name]
                buf[0].append(starts[a:b])
                buf[1].append(ends[a:b])
                buf[2].append(values[a:b])
    finally:
        reader.close()
    return {c: (np.concatenate(s), np.concatenate(e), np.concatenate(v))
            for c, (s, e, v) in out.items()}


def explode_to_chunks(starts, ends, values, feature_ids, chunk_size):
    """Repeat each peak once per chunk it overlaps, keeping its true coordinates.

    ``np.repeat`` plus an arange-minus-offset gives the per-copy chunk index
    without a Python loop; almost every peak spans one or two chunks, so the
    output is barely larger than the input.
    """
    if len(starts) == 0:
        e64, e32 = np.empty(0, np.int64), np.empty(0, np.int32)
        return e64, e64, e32, e32, e64
    first = starts // chunk_size
    # A zero-width row would otherwise land in the chunk before it.
    last = np.maximum(ends - 1, starts) // chunk_size
    span = (last - first + 1).astype(np.int64)
    offsets = np.arange(int(span.sum())) - np.repeat(np.cumsum(span) - span, span)
    return (np.repeat(starts, span), np.repeat(ends, span),
            np.repeat(values, span), np.repeat(feature_ids, span),
            np.repeat(first, span) + offsets)


def build_table(chrom, starts, ends, values, feature_ids, features):
    """The Arrow table for one (tissue, chromosome), sorted by (chunk, start)."""
    return pa.table({
        "Chromosome": _constant(chrom, len(starts)),
        "Start": compat.to_arrow(starts, pa.int64()),
        "End": compat.to_arrow(ends, pa.int64()),
        "Name": compat.to_arrow(values, pa.int32()),
        "feature_name": pa.DictionaryArray.from_arrays(
            compat.to_arrow(np.ascontiguousarray(feature_ids, np.int32), pa.int32()),
            pa.array(features)),  # abi-ok: features is a Python list of str
    })


def _constant(value, n):
    """An n-long string column of one repeated value, built without Python."""
    return pa.DictionaryArray.from_arrays(
        compat.to_arrow(np.zeros(n, dtype=np.int32), pa.int32()),
        pa.array([value]))


def write_tissue_chromosome(release, tissue, chrom, tracks, features, chunk_size):
    """One (tissue, chromosome) parquet, one row group per populated chunk."""
    target = release.omics_chunk(tissue, chrom)
    target.parent.mkdir(parents=True, exist_ok=True)

    parts = [(a, np.full(len(s), features.index(a), dtype=np.int32))
             for a, (s, _, _) in tracks.items() if len(s)]
    if not parts:
        # An empty file, not a missing one: the loader must be able to tell
        # "this tissue has nothing on this chromosome" from "the stage never ran".
        pq.ParquetWriter(target, out_schema().with_metadata({CHUNK_KEY: b"[]"}),
                         compression="zstd").close()
        return {"rows": 0, "chunks": 0, "bytes": target.stat().st_size}

    starts = np.concatenate([tracks[a][0] for a, _ in parts])
    ends = np.concatenate([tracks[a][1] for a, _ in parts])
    values = np.concatenate([tracks[a][2] for a, _ in parts])
    feature_ids = np.concatenate([ids for _, ids in parts])

    starts, ends, values, feature_ids, chunk = explode_to_chunks(
        starts, ends, values, feature_ids, chunk_size)

    # Sorted by (chunk, start) so each chunk's rows are contiguous -- that is
    # what makes one row group per chunk possible -- and so a chunk's rows come
    # back in genomic order, which the read path already assumes.
    order = np.lexsort((starts, chunk))
    starts, ends = starts[order], ends[order]
    values, feature_ids, chunk = values[order], feature_ids[order], chunk[order]

    edges = np.flatnonzero(np.diff(chunk)) + 1
    bounds = np.r_[0, edges, len(chunk)]
    present = chunk[np.r_[0, edges]].astype(np.int64).tolist()

    table = build_table(chrom, starts, ends, values, feature_ids, features)
    schema = table.schema.with_metadata(
        {CHUNK_KEY: json.dumps(present).encode()})
    writer = pq.ParquetWriter(target, schema, compression="zstd")
    try:
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            writer.write_table(table.slice(int(lo), int(hi - lo)))
    finally:
        writer.close()
    return {"rows": len(table), "chunks": len(present),
            "bytes": target.stat().st_size}


def build_omics(release, tissue, features, chroms, chunk_size, block_size,
                aliases=None):
    """Every chromosome's parquet for one tissue, reading each bedGraph once."""
    files = signal_files(release, tissue, features, aliases)
    if not files:
        print("  %s: no signal files for any vocabulary antigen" % tissue,
              flush=True)
    by_chrom = defaultdict(dict)
    for antigen, path in files.items():
        for chrom, arrays in read_bedgraph(path, chroms, block_size).items():
            by_chrom[chrom][antigen] = arrays

    totals = {"files": len(files), "rows": 0, "chunks": 0, "bytes": 0,
              "chroms": 0}
    for chrom in sorted(chroms):
        stats = write_tissue_chromosome(release, tissue, chrom,
                                        by_chrom.get(chrom, {}), features,
                                        chunk_size)
        totals["chroms"] += 1
        for key in ("rows", "chunks", "bytes"):
            totals[key] += stats[key]
    return totals


def build_dna(release, chroms, chunk_size, donor=None, dna_dir=None):
    """One uppercase text file per chunk, adopted from an existing cut if given.

    The reference does not change between releases, so a rebuild links to the
    chunks something else already cut -- 47,000 hard links instead of 47,000
    slices of a 3.1 GB pickle.

    ``dna_dir`` names a bare ``<chrom>/<start>_<end>.txt`` tree rather than a
    release. That is what allows a new release to be built while the
    pre-release tree is still in place and still being read by a running job:
    linking only reads the source directory, so nothing has to be migrated
    first, and a hard link means the new release does not duplicate the 3 GB
    either.
    """
    totals = {"chunks": 0, "linked": 0, "written": 0}
    if donor is not None or dna_dir is not None:
        source_name = donor.id if donor is not None else str(dna_dir)
        for chrom in sorted(chroms):
            length = chroms[chrom]
            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                if donor is not None:
                    source = donor.dna_chunk(chrom, start, end)
                else:
                    source = Path(dna_dir) / chrom / ("%d_%d.txt" % (start, end))
                if not source.exists():
                    raise SystemExit(
                        "%s has no chunk %s:%d-%d -- its grid does not match "
                        "this release's %d bp chunks"
                        % (source_name, chrom, start, end, chunk_size))
                adopt(source, release.dna_chunk(chrom, start, end))
                totals["chunks"] += 1
                totals["linked"] += 1
        return totals

    with open(release.path("sequence"), "rb") as fh:
        sequence = pickle.load(fh)
    for chrom in sorted(chroms):
        seq = sequence[chrom]
        length = chroms[chrom]
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            target = release.dna_chunk(chrom, start, end)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(seq[start:end].upper())
            totals["chunks"] += 1
            totals["written"] += 1
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--org", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--what", choices=("omics", "dna"), required=True)
    parser.add_argument("--from-release", default=None,
                        help="dna only: adopt chunks from this release instead "
                             "of cutting them out of sequence.pkl")
    parser.add_argument("--dna-dir", type=Path, default=None,
                        help="dna only: adopt chunks from a bare "
                             "<chrom>/<start>_<end>.txt tree, e.g. a pre-release "
                             "Subtables/dna. Lets a release be built before the "
                             "old tree is migrated, and while it is still in use")
    parser.add_argument("--task", default="env",
                        help="task index, 'all', or 'env' for SLURM_ARRAY_TASK_ID")
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=64 << 20)
    args = parser.parse_args(argv)

    release = layout.Release.open(args.data_dir, args.org, args.release)
    release.require("genome", "vocab", "intervals")
    chroms = {c: int(n) for c, n in release.manifest["chrom_sizes"].items()}

    if args.what == "dna":
        donor = (layout.Release.open(args.data_dir, args.org, args.from_release)
                 if args.from_release else None)
        if donor is not None and args.dna_dir is not None:
            raise SystemExit("pass --from-release or --dna-dir, not both")
        started = time.time()
        totals = build_dna(release, chroms, release.chunk_size, donor,
                           args.dna_dir)
        totals["seconds"] = round(time.time() - started, 1)
        release.record("chunks_dna", **totals)
        print("dna: %(chunks)d chunks (%(linked)d linked, %(written)d written)"
              " in %(seconds)ss" % totals, flush=True)
        return 0

    vocabulary = json.loads(release.path("features").read_text())
    features = vocabulary["features"]
    aliases = vocabulary.get("aliases") or {}
    tissues = [t["name"] for t in
               json.loads(release.path("tissues").read_text())["tissues"]]

    mine = select_tasks(tissues, args.task, args.tasks)
    if mine is None:
        return 0
    print("task: %d of %d tissues -- %s"
          % (len(mine), len(tissues), ", ".join(mine)), flush=True)

    started = time.time()
    stats_dir = release.root / "work" / "chunks"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for tissue in mine:
        at = time.time()
        totals = build_omics(release, tissue, features, chroms,
                             release.chunk_size, args.block_size, aliases)
        totals["seconds"] = round(time.time() - at, 1)
        (stats_dir / ("%s.json" % tissue)).write_text(json.dumps(totals))
        print("  %-28s %4d tracks  %10d rows  %6.1f MB  %ss"
              % (tissue, totals["files"], totals["rows"],
                 totals["bytes"] / 1e6, totals["seconds"]), flush=True)
    print("done in %.1fs" % (time.time() - started), flush=True)
    return 0


def select_tasks(items, task, tasks):
    """This task's slice of a work list, strided so sizes interleave."""
    if task == "all":
        return list(items)
    raw = os.environ.get("SLURM_ARRAY_TASK_ID") if task == "env" else task
    if raw is None:
        raise SystemExit("--task env needs SLURM_ARRAY_TASK_ID")
    index = int(raw)
    if index >= tasks:
        print("task %d: nothing to do (%d tasks)" % (index, tasks))
        return None
    return list(items)[index::tasks]


if __name__ == "__main__":
    raise SystemExit(main())
