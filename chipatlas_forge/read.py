"""Reading a release back: the reference implementation of the manifest contract.

Everything here is driven by `MANIFEST.json` rather than by hardcoded paths, so
the same functions read a 2021 release with 900k per-chunk pickles and a 2026
one with per-chromosome parquet. Callers ask for a chunk; which of the two
shapes is on disk is this module's problem.

`chipatlas_forge.pairs` uses it, and it is the file to read when writing a
reader anywhere else -- the training repo has its own, deliberately independent,
because the two repositories are not installed into each other. The manifest is
what keeps them honest: a layout change that one side misses shows up as a
missing path key at startup, not as quietly wrong data.
"""

import functools
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import layout

# Column order `create_feature_matrix` expects, and what an empty result must
# still have -- a frame with no columns raises on `.feature_name` rather than
# returning nothing, which turns "no peaks here" into a crash in a dataloader.
OMICS_COLUMNS = ["Chromosome", "Start", "End", "Name", "feature_name"]

CHUNK_KEY = b"chipatlas_forge:chunks"


def empty_omics():
    return pd.DataFrame({c: pd.Series(dtype="object" if c in
                                      ("Chromosome", "feature_name") else "int64")
                         for c in OMICS_COLUMNS})


@functools.lru_cache(maxsize=256)
def _row_group_index(path):
    """``{chunk index: row group}`` from the parquet footer.

    Cached because a dataloader worker hits the same 25 files for a whole epoch
    and the footer read is the only per-open cost. Keyed on the path, so a
    rebuilt file within one process would need the cache cleared -- releases are
    immutable once promoted, which is what makes that safe.
    """
    handle = pq.ParquetFile(path)
    metadata = handle.schema_arrow.metadata or {}
    present = json.loads(metadata.get(CHUNK_KEY, b"[]"))
    return handle, {int(c): i for i, c in enumerate(present)}


def load_chunk(release, tissue, chrom, start, end):
    """Every omics row stored against one chunk, as the loader's frame."""
    if release.omics_layout == layout.CHUNKED_PICKLE:
        path = release.omics_chunk(tissue, chrom, start, end)
        if not path.exists():
            return empty_omics()
        frame = pd.read_pickle(path)
        return frame if len(frame) else empty_omics()

    path = release.omics_chunk(tissue, chrom)
    if not path.exists():
        return empty_omics()
    handle, index = _row_group_index(path)
    row_group = index.get(int(start) // release.chunk_size)
    if row_group is None:
        return empty_omics()
    return handle.read_row_group(row_group).to_pandas()


def load_window(release, tissue, chrom, begin, end):
    """Omics for an arbitrary interval, stitched across the chunks it spans.

    Rows are *not* de-duplicated here even though a peak crossing a chunk
    boundary is stored in both: `create_feature_matrix` drops duplicates on
    (Chromosome, Start, End, feature_name) itself, and doing it twice would be
    pure cost on the hot path.
    """
    size = release.chunk_size
    frames = []
    for chunk in range(int(begin) // size, (int(end) - 1) // size + 1):
        lo = chunk * size
        frames.append(load_chunk(release, tissue, chrom, lo, lo + size))
    frames = [f for f in frames if len(f)]
    if not frames:
        return empty_omics()
    return pd.concat(frames, ignore_index=True)


def features(release):
    return list(json.loads(release.path("features").read_text())["features"])


def tissues(release):
    return [t["name"] for t in
            json.loads(release.path("tissues").read_text())["tissues"]]


def availability(release):
    """``{tissue: [feature, ...]}`` -- which tracks a tissue actually has."""
    return json.loads(release.availability().read_text())


def windows(release, window, split=None, tissue=None):
    """The window table for one size, optionally one split, as a frame.

    Both interval layouts come back with the same columns, so a caller never
    branches on which release it opened.
    """
    if release.interval_layout == layout.PER_SPLIT_PICKLE:
        return _legacy_windows(release, window, split, tissue)
    frame = pq.read_table(release.windows(window)).to_pandas()
    if split is not None:
        frame = frame[frame["split"] == split]
    if tissue is not None:
        frame = frame[frame["tissue"] == tissue]
    return frame.reset_index(drop=True)


def _legacy_windows(release, window, split, tissue):
    """The 2021 pickles, given the same column names as the parquet form.

    They hold a bare positional frame -- chrom, start, end, strand, tissue, with
    no header and no split column -- so the split has to come from which file it
    was in. `val` is absent by construction: that release had no validation
    holdout, it sampled 1000 rows out of test.
    """
    names = {"train": "windows_train", "test": "windows_test",
             None: "windows_full"}
    if split == "val":
        raise SystemExit(
            "release %s has no validation split -- it predates the chr8+chr9 "
            "holdout and sampled validation out of test" % release.id)
    frame = pd.read_pickle(release.path(names[split], window=window))
    frame = pd.DataFrame(np.asarray(frame)[:, :5],
                         columns=["chrom", "start", "end", "strand", "tissue"])
    frame["start"] = frame["start"].astype("int64")
    frame["end"] = frame["end"].astype("int64")
    frame["split"] = split or "unknown"
    if tissue is not None:
        frame = frame[frame["tissue"] == tissue]
    return frame.reset_index(drop=True)


def dna(release, chrom, begin, end):
    """The reference sequence for an interval, stitched across chunk files."""
    size = release.chunk_size
    out = []
    for chunk in range(int(begin) // size, (int(end) - 1) // size + 1):
        lo = chunk * size
        path = release.dna_chunk(chrom, lo, lo + size)
        if not path.exists():
            # The final chunk of a chromosome is short and named by its true end.
            matches = sorted(path.parent.glob("%d_*.txt" % lo))
            if not matches:
                raise FileNotFoundError("no DNA chunk at %s:%d" % (chrom, lo))
            path = matches[0]
        text = path.read_text()
        out.append(text[max(0, int(begin) - lo):max(0, int(end) - lo)])
    return "".join(out)
