"""Tests for the prepare stages: vocab, genome, intervals, chunks, read, migrate.

Built around one synthetic organism assembled once per session, because the
stages are a pipeline -- the thing worth testing is that stage 9 can read what
stage 8 wrote, which a per-test fixture would never exercise. The toy genome is
deliberately awkward: an N gap that straddles window boundaries, a blacklist
region, a chromosome shorter than one 2**20 split block, and a tissue with no
signal at all on one chromosome.

The numeric helpers (`overlap_bp`, `n_runs`, `window_starts`) are additionally
checked against brute force over random inputs, since they are the parts where a
subtle error produces plausible data rather than a crash.
"""

import json
import pickle

import numpy as np
import pytest

pytest.importorskip("pyarrow")
import pyarrow.parquet as pq                                       # noqa: E402

from chipatlas_forge import (chunks, genome, intervals, layout,    # noqa: E402
                             migrate, pantissue, read, verify, vocab)

CHROM_SIZES = {"chr1": 400_000, "chr8": 300_000, "chr9": 120_000}
TISSUES = ["Blood", "Liver", "Neural"]
# The derived all-cell-types track, built by `pantissue` and not present in
# groups.tsv -- ChIP-Atlas has no such cell-type class to group by.
PAN = "All cell types"
ALL_TISSUES = sorted(TISSUES + [PAN])
ANTIGENS = ["AG%03d" % i for i in range(60)]
N_GAP = (50_000, 70_000)
BLACKLIST = [("chr1", 120_000, 130_000), ("chr8", 10_000, 11_000)]

FILE_LIST = "".join(
    "%s.%s.05.AllAg.AllCell\ttoy\t%s\t-\t%s\t-\t05\tSRX1\n" % (ag, ct, agn, ctn)
    for ag, agn in (("His", "Histone"), ("Oth", "TFs and others"))
    for ct, ctn in (("Bld", "Blood"), ("Liv", "Liver"), ("Neu", "Neural"),
                    ("ALL", PAN)))


def antigen_class(antigen):
    return "Histone" if int(antigen[2:]) % 3 == 0 else "TFs and others"


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    """A complete release, built by running every stage in order."""
    root = tmp_path_factory.mktemp("forge")
    data, meta, src = root / "data", root / "meta", root / "src"
    meta.mkdir()
    src.mkdir()
    rng = np.random.default_rng(7)

    sequence = {}
    for chrom, length in CHROM_SIZES.items():
        letters = np.array(list("ACGT"))[rng.integers(0, 4, length)]
        letters[N_GAP[0]:N_GAP[1]] = "N"
        sequence[chrom] = "".join(letters.tolist())
    with open(src / "sequence.pkl", "wb") as fh:
        pickle.dump(sequence, fh)
    with open(src / "genome.fa", "w") as fh:
        for chrom, seq in sequence.items():
            fh.write(">%s\n" % chrom)
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")
    (src / "blacklist.bed").write_text(
        "".join("%s\t%d\t%d\tGap\n" % b for b in BLACKLIST))
    (meta / "fileList.tab").write_text(FILE_LIST)

    release = layout.Release.create(data, "toy", "2026-08")

    rows = ["group_id\tag_class\tantigen\tct_class\tn_experiments\tpath"]
    gid = 0
    for tissue in TISSUES:
        for antigen in ANTIGENS:
            rows.append("%d\t%s\t%s\t%s\t10\tx"
                        % (gid, antigen_class(antigen), antigen, tissue))
            gid += 1
    # Decoys, each of which one filter must remove.
    rows.append("%d\tInput control\tInput\tBlood\t99\tx" % gid); gid += 1
    rows.append("%d\tHistone\tEpitope_tags\tBlood\t99\tx" % gid); gid += 1
    rows.append("%d\tHistone\tLONELY\tBlood\t99\tx" % gid); gid += 1
    for antigen in ANTIGENS[:10]:
        rows.append("%d\tHistone\t%s\tSparse\t5\tx" % (gid, antigen)); gid += 1
    release.path("groups").parent.mkdir(parents=True, exist_ok=True)
    release.path("groups").write_text("\n".join(rows) + "\n")

    args = ["--data-dir", str(data), "--org", "toy", "--release", "2026-08"]
    assert genome.main(args + ["--fasta", str(src / "genome.fa"),
                               "--blacklist", str(src / "blacklist.bed"),
                               "--sequence", str(src / "sequence.pkl")]) == 0

    release = layout.Release.open(data, "toy", "2026-08")
    for tissue in TISSUES:
        for antigen in ANTIGENS:
            path = (release.path("signal_root")
                    / antigen_class(antigen).replace(" ", "_") / tissue
                    / ("%s.bedgraph" % antigen))
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            # Nothing on chr9 at all, so the "empty but present" case is real.
            for chrom in ("chr1", "chr8"):
                starts = np.sort(rng.integers(0, CHROM_SIZES[chrom] - 4000, 40))
                lines += ["%s\t%d\t%d\t%d" % (chrom, s, s + rng.integers(200, 3000),
                                              rng.integers(50, 4000))
                          for s in starts.tolist()]
            path.write_text("\n".join(lines) + "\n")

    # Derived from the per-tissue tracks, so it has to come after them and
    # before vocab -- which lists it as a tissue.
    assert pantissue.main(args + ["--task", "all"]) == 0
    assert vocab.main(args + ["--meta-dir", str(meta)]) == 0
    assert intervals.main(args + ["--windows", "8192", "65536",
                                  "--val-chroms", "chr8", "chr9"]) == 0
    assert chunks.main(args + ["--what", "dna"]) == 0
    assert chunks.main(args + ["--what", "omics", "--task", "all"]) == 0
    layout.promote(data, "toy", "2026-08")
    return {"data": data, "meta": meta, "src": src, "sequence": sequence,
            "release": layout.Release.open(data, "toy", "2026-08")}


# --------------------------------------------------------------------------
# numeric helpers, against brute force


def test_overlap_bp_matches_a_dense_mask():
    """Subsumes the training repo's `get_omics_vector` tests, which are now gone.

    Those seven cases guarded a real bug: the old code narrowed candidate rows
    with ``ind_left = searchsorted(...) - 1``, which is ``-1`` when no interval
    starts at or before ``begin``, and ``.iloc[-1:k]`` counts from the END of
    the frame and returns an empty slice. Every chromosome's first region starts
    at ``begin=0``, so blacklist coverage there came back all-zero and
    blacklisted windows were never excluded.

    `overlap_bp` cannot reproduce it -- there is no row pre-selection to get
    wrong -- and this checks the property those cases sampled by hand against a
    dense mask over 300 random inputs, including windows starting at 0.
    """
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(0, 12))
        starts = np.sort(rng.integers(0, 500, n)).astype(np.int64)
        ends = starts + rng.integers(1, 40, n)
        starts, ends = intervals.merge_runs(starts, ends)
        mask = np.zeros(700, dtype=bool)
        for a, b in zip(starts.tolist(), ends.tolist()):
            mask[a:b] = True
        q = int(rng.integers(1, 10))
        q_start = np.sort(rng.integers(0, 500, q)).astype(np.int64)
        q_end = q_start + rng.integers(1, 60, q)
        expected = np.array([mask[a:b].sum()
                             for a, b in zip(q_start.tolist(), q_end.tolist())])
        got = intervals.overlap_bp(q_start, q_end, starts, ends)
        assert np.array_equal(got, expected)


@pytest.mark.parametrize("seq", ["ACGT", "NNNN", "ACNNNGT", "NACGTN", "",
                                 "nnACGTnn", "NNACGTNNNNAC"])
def test_n_runs_matches_a_per_base_scan(seq):
    starts, ends = intervals.n_runs(seq)
    flat = [i for a, b in zip(starts.tolist(), ends.tolist()) for i in range(a, b)]
    assert flat == [i for i, c in enumerate(seq) if c in "Nn"]


@pytest.mark.parametrize("length,window", [(100, 10), (105, 10), (7, 10),
                                           (1 << 20, 8192), (248956422, 1 << 20)])
def test_window_starts_are_full_width_and_cover_the_chromosome(length, window):
    starts = intervals.window_starts(length, window)
    ends = np.minimum(starts + window, length)
    assert starts[0] == 0
    assert ends[-1] == length
    assert (np.diff(starts) > 0).all()
    if length >= window:
        assert (ends - starts == window).all()


def test_merge_runs_makes_overlapping_intervals_disjoint():
    starts = np.array([10, 5, 12, 40], dtype=np.int64)
    ends = np.array([20, 15, 14, 50], dtype=np.int64)
    got_s, got_e = intervals.merge_runs(starts, ends)
    assert got_s.tolist() == [5, 40]
    assert got_e.tolist() == [20, 50]


# --------------------------------------------------------------------------
# vocab


def test_vocab_applies_both_thresholds_to_a_fixed_point(toy):
    features = json.loads(toy["release"].path("features").read_text())["features"]
    tissues = json.loads(toy["release"].path("tissues").read_text())["tissues"]
    assert features == ANTIGENS
    assert [t["name"] for t in tissues] == ALL_TISSUES
    # "Sparse" had 10 antigens, below the 50 threshold; LONELY had one tissue,
    # below 3; Epitope_tags and Input control are excluded outright.
    assert "Sparse" not in [t["name"] for t in tissues]
    for dropped in ("LONELY", "Epitope_tags", "Input"):
        assert dropped not in features


def test_vocab_derives_class_codes_from_filelist(toy):
    tissues = json.loads(toy["release"].path("tissues").read_text())["tissues"]
    codes = {t["name"]: t["code"] for t in tissues}
    assert codes == {"Blood": "Bld", "Liver": "Liv", "Neural": "Neu",
                     PAN: "ALL"}


def test_the_pan_tissue_carries_every_antigen(toy):
    """It is a genuine superset, unlike the 2021 one it replaces.

    That one had ~1004 antigens but none of H3K27ac / H3K4me1 / H3K4me3 /
    H3K27me3 / RNA polymerase II, because ChIP-Atlas publishes those only
    per-tissue -- so pairing on it saw none of the marks that define enhancers
    and promoters.
    """
    have = json.loads(toy["release"].availability().read_text())
    assert set(have[PAN]) == set(ANTIGENS)
    for tissue in TISSUES:
        assert set(have[tissue]) <= set(have[PAN])


def test_freezing_keeps_column_order_and_reports_the_drift():
    frozen = ["KEPT", "GONE"]
    pairs = {("Blood", "KEPT"), ("Blood", "BRAND_NEW")}
    out, trimmed, tissues, thin, vanished, appeared = vocab.apply_freeze(
        frozen, pairs, {}, min_antigens=1)
    assert out == frozen, "the frozen order must survive verbatim"
    assert vanished == ["GONE"], "a frozen feature absent from the data is kept"
    assert appeared == ["BRAND_NEW"], "a new feature is reported, not silently added"
    assert trimmed["Blood"] == ["KEPT"]
    assert tissues == ["Blood"] and thin == []


def test_freezing_does_not_re_apply_the_antigen_threshold():
    """The threshold picks a vocabulary; a frozen one has already been picked.

    On real hg38 this was 395 of 1,009 columns -- 392 of them with data in
    exactly two tissues -- marked absent and zeroed for no reason.
    """
    frozen = ["ONLY_TWO_TISSUES"]
    pairs = {("Blood", "ONLY_TWO_TISSUES"), ("Liver", "ONLY_TWO_TISSUES")}
    _, trimmed, tissues, _, vanished, _ = vocab.apply_freeze(
        frozen, pairs, {}, min_antigens=1)
    assert vanished == [], "a two-tissue frozen feature still has data"
    assert sorted(tissues) == ["Blood", "Liver"]
    assert trimmed["Blood"] == ["ONLY_TWO_TISSUES"]


def test_the_old_dot_mangling_is_undone():
    """`H2APERIODX` and `H2A.X` are the same track under a corrupted name."""
    frozen = ["H2APERIODX", "RNA_polymerase_II", "H3PERIOD3_K27M_mutant"]
    present = {"H2A.X", "RNA polymerase II", "H3.3 K27M mutant", "CTCF"}
    aliases, ambiguous = vocab.resolve_aliases(frozen, present)
    assert not ambiguous
    assert aliases == {"H2APERIODX": "H2A.X",
                       "RNA_polymerase_II": "RNA polymerase II",
                       "H3PERIOD3_K27M_mutant": "H3.3 K27M mutant"}


def test_an_ambiguous_alias_is_refused_rather_than_guessed():
    """Guessing wrong feeds one track's peaks into another track's column.

    `H2A.X` and `H2A.x` share a canonical form, so neither may be chosen.
    """
    aliases, ambiguous = vocab.resolve_aliases(
        ["H2APERIODX"], {"H2A.X", "H2A.x"})
    assert aliases == {}, "an ambiguous name must not be mapped at all"
    assert ambiguous == {"H2APERIODX": ["H2A.X", "H2A.x"]}


def test_canonical_does_not_collapse_genuinely_different_names():
    """It must be loose enough to match the mangling and no looser."""
    assert vocab.canonical("H2APERIODX") == vocab.canonical("H2A.X")
    assert vocab.canonical("RNA_polymerase_II") == vocab.canonical("RNA polymerase II")
    assert vocab.canonical("H3.3") != vocab.canonical("H3_3")
    assert vocab.canonical("CTCF") != vocab.canonical("CTCFL")


def test_aliased_features_are_available_under_their_vocabulary_name():
    frozen = ["H2APERIODX"]
    pairs = {("Blood", "H2A.X")}
    _, trimmed, tissues, _, vanished, appeared = vocab.apply_freeze(
        frozen, pairs, {"H2APERIODX": "H2A.X"}, min_antigens=1)
    assert vanished == [], "the alias should have found the data"
    assert trimmed["Blood"] == ["H2APERIODX"], "filed under the model's column name"
    assert appeared == [], "the data name is not also reported as new"


def test_signal_files_are_found_through_an_alias(toy):
    """The file is named H2A.X; the column is called H2APERIODX."""
    release = toy["release"]
    path = (release.path("signal_root") / "Histone" / "Blood" / "H2A.X.bedgraph")
    path.write_text("chr1\t0\t10\t500\n")
    found = chunks.signal_files(release, "Blood", ["H2APERIODX"],
                                {"H2APERIODX": "H2A.X"})
    assert set(found) == {"H2APERIODX"}, "keyed by vocabulary name, not filename"
    assert found["H2APERIODX"].name == "H2A.X.bedgraph"
    path.unlink()


def test_select_converges_when_the_thresholds_interact():
    """A tissue dropped for sparsity can push a feature below its tissue count.

    One pass in either order leaves survivors violating the other threshold,
    which is how a "min 3 tissues" list ends up holding 2-tissue features.
    """
    import pandas as pd
    rows = []
    for tissue in ("A", "B", "C"):
        for i in range(60):
            rows.append(("Histone", "F%02d" % i, tissue))
    # D has only the one antigen, so D goes; X then has just A, B and goes too.
    rows.append(("Histone", "X", "D"))
    for tissue in ("A", "B"):
        rows.append(("Histone", "X", tissue))
    frame = pd.DataFrame(rows, columns=["ag_class", "antigen", "ct_class"])
    frame["n_experiments"] = 1
    tissues, antigens, _, rounds = vocab.select(frame, 50, 3)
    assert tissues == ["A", "B", "C"]
    assert "X" not in antigens
    assert rounds > 1, "a single pass would have kept X"


# --------------------------------------------------------------------------
# intervals


def test_val_is_exactly_the_val_chromosomes(toy):
    frame = read.windows(toy["release"], 8192)
    assert set(frame[frame.split == "val"].chrom) == {"chr8", "chr9"}
    on_val = frame[frame.chrom.isin(["chr8", "chr9"])]
    assert (on_val.split == "val").all()


def test_blacklisted_and_n_rich_windows_are_dropped(toy):
    frame = read.windows(toy["release"], 8192)
    for chrom, lo, hi in BLACKLIST:
        hit = frame[(frame.chrom == chrom) & (frame.start < hi) & (frame.end > lo)]
        assert len(hit) == 0, "blacklisted windows survived on %s" % chrom
    inside_gap = frame[(frame.chrom == "chr1") & (frame.start >= N_GAP[0])
                       & (frame.end <= N_GAP[1])]
    assert len(inside_gap) == 0


def test_the_split_nests_across_window_sizes(toy):
    """An 8192 window must never disagree with the 2**20 block containing it.

    This is the leak the block-level assignment exists to prevent: the same
    bases in train at one window size and test at another.
    """
    small = read.windows(toy["release"], 8192)
    large = read.windows(toy["release"], 65536)
    block_label = {}
    for chrom, start, end, split in zip(large.chrom, large.start, large.end,
                                        large.split):
        block_label[(chrom, ((start + end) // 2) // intervals.SPLIT_BLOCK)] = split
    for chrom, start, end, split in zip(small.chrom, small.start, small.end,
                                        small.split):
        key = (chrom, ((start + end) // 2) // intervals.SPLIT_BLOCK)
        if key in block_label:
            assert block_label[key] == split, "%s:%d is %s but its block is %s" % (
                chrom, start, split, block_label[key])


def test_windows_are_materialised_per_strand_and_tissue(toy):
    frame = read.windows(toy["release"], 8192)
    assert set(frame.strand) == {"+", "-"}
    assert set(frame.tissue) == set(ALL_TISSUES)
    loci = frame[["chrom", "start"]].drop_duplicates()
    assert len(frame) == len(loci) * 2 * len(ALL_TISSUES)


def test_no_locus_appears_in_two_splits(toy):
    for window in (8192, 65536):
        frame = read.windows(toy["release"], window)
        key = frame.chrom.astype(str) + ":" + frame.start.astype(str)
        assert frame.assign(k=key).groupby("k").split.nunique().max() == 1


def test_chunk_grid_tiles_every_chromosome_exactly(toy):
    release = toy["release"]
    grid = pq.read_table(release.path("chunk_grid")).to_pandas()
    for chrom, length in release.manifest["chrom_sizes"].items():
        rows = grid[grid.chrom == chrom].sort_values("start")
        assert rows.start.iloc[0] == 0
        assert rows.end.iloc[-1] == length
        assert (rows.start.values[1:] == rows.end.values[:-1]).all(), "gap or overlap"


# --------------------------------------------------------------------------
# chunks


def test_dna_chunks_are_the_source_sequence(toy):
    release, sequence = toy["release"], toy["sequence"]
    size = release.chunk_size
    for chrom, length in release.manifest["chrom_sizes"].items():
        for start in range(0, length, size):
            end = min(start + size, length)
            got = release.dna_chunk(chrom, start, end).read_text()
            assert got == sequence[chrom][start:end].upper()


def test_every_peak_is_stored_once_per_chunk_it_overlaps(toy):
    """Conservation: nothing lost at a chunk boundary, nothing duplicated inside one."""
    release = toy["release"]
    size = release.chunk_size
    expected = 0
    for antigen in ANTIGENS:
        path = (release.path("signal_root")
                / antigen_class(antigen).replace(" ", "_") / "Blood"
                / ("%s.bedgraph" % antigen))
        for line in path.read_text().splitlines():
            chrom, start, end, _ = line.split("\t")
            if chrom != "chr1":
                continue
            expected += (max(int(end) - 1, int(start)) // size) - (int(start) // size) + 1
    stored = pq.ParquetFile(release.omics_chunk("Blood", "chr1")).metadata.num_rows
    assert stored == expected


def test_each_row_group_is_confined_to_its_chunk_and_sorted(toy):
    release = toy["release"]
    handle = pq.ParquetFile(release.omics_chunk("Blood", "chr1"))
    present = json.loads(handle.schema_arrow.metadata[chunks.CHUNK_KEY])
    assert handle.num_row_groups == len(present)
    size = release.chunk_size
    for i, chunk in enumerate(present):
        rows = handle.read_row_group(i).to_pandas()
        lo, hi = chunk * size, (chunk + 1) * size
        assert ((rows.End > lo) & (rows.Start < hi)).all()
        assert rows.Start.is_monotonic_increasing


def test_a_chromosome_with_no_signal_is_empty_not_missing(toy):
    """The loader must be able to tell "nothing here" from "the stage died"."""
    path = toy["release"].omics_chunk("Blood", "chr9")
    assert path.exists()
    assert pq.ParquetFile(path).metadata.num_rows == 0


def test_explode_to_chunks_handles_a_peak_spanning_many_chunks():
    starts = np.array([0, 100, 65535], dtype=np.int64)
    ends = np.array([200_000, 300, 65537], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int32)
    features = np.array([0, 1, 2], dtype=np.int32)
    s, e, v, f, chunk = chunks.explode_to_chunks(starts, ends, values, features,
                                                 65536)
    assert chunk[f == 0].tolist() == [0, 1, 2, 3]     # spans four chunks
    assert chunk[f == 1].tolist() == [0]
    assert chunk[f == 2].tolist() == [0, 1]           # straddles the boundary
    assert (s[f == 0] == 0).all() and (e[f == 0] == 200_000).all(), \
        "coordinates must stay uncut"


# --------------------------------------------------------------------------
# read


def test_load_window_returns_every_overlapping_peak(toy):
    release = toy["release"]
    got = read.load_window(release, "Blood", "chr1", 60_000, 200_000)
    whole = pq.read_table(release.omics_chunk("Blood", "chr1")).to_pandas()
    want = whole[(whole.End > 60_000) & (whole.Start < 200_000)]
    key = ["Start", "End", "feature_name"]
    missing = (set(map(tuple, want[key].values.tolist()))
               - set(map(tuple, got[key].values.tolist())))
    assert not missing


def test_load_chunk_of_an_empty_chromosome_has_the_right_columns(toy):
    frame = read.load_chunk(toy["release"], "Blood", "chr9", 0, 65536)
    assert len(frame) == 0
    assert set(read.OMICS_COLUMNS) <= set(frame.columns)


def test_dna_stitches_across_chunk_boundaries(toy):
    got = read.dna(toy["release"], "chr1", 65_000, 132_000)
    assert got == toy["sequence"]["chr1"][65_000:132_000].upper()


def test_windows_can_be_filtered_by_split_and_tissue(toy):
    frame = read.windows(toy["release"], 8192, split="val", tissue="Blood")
    assert set(frame.chrom) == {"chr8", "chr9"}
    assert set(frame.tissue) == {"Blood"}


# --------------------------------------------------------------------------
# layout


def test_latest_resolves_through_the_symlink(toy):
    assert layout.resolve_release(toy["data"], "toy") == "2026-08"
    assert layout.resolve_release(toy["data"], "toy", "latest") == "2026-08"


def test_an_unknown_release_is_rejected_by_name(toy):
    with pytest.raises(FileNotFoundError) as excinfo:
        layout.resolve_release(toy["data"], "toy", "1999-01")
    assert "2026-08" in str(excinfo.value), "the error should list what is available"


def test_a_release_id_cannot_escape_the_releases_directory():
    for bad in ("../evil", "/abs", "", "a/b"):
        with pytest.raises(SystemExit):
            layout.check_release_id(bad)


def test_the_manifest_carries_its_own_path_templates(toy):
    """The training repo formats these; if they vanish it must fail loudly."""
    paths = toy["release"].manifest["paths"]
    for key in ("dna_chunk", "omics_chunk", "windows", "features", "tissues",
                "availability", "sequence", "blacklist"):
        assert key in paths
    assert "{chrom}" in paths["dna_chunk"] and "{start}" in paths["dna_chunk"]


def test_opening_a_release_from_a_future_layout_version_refuses(toy, tmp_path):
    root = layout.release_root(toy["data"], "toy", "2026-08")
    blob = json.loads((root / "MANIFEST.json").read_text())
    blob["layout_version"] = layout.LAYOUT_VERSION + 1
    other = tmp_path / "data" / "toy" / "releases" / "9999-99"
    other.mkdir(parents=True)
    (other / "MANIFEST.json").write_text(json.dumps(blob))
    with pytest.raises(SystemExit):
        layout.Release.open(tmp_path / "data", "toy", "9999-99")


def test_require_names_the_stage_that_has_not_run(toy):
    with pytest.raises(SystemExit) as excinfo:
        toy["release"].require("nonexistent_stage")
    assert "nonexistent_stage" in str(excinfo.value)


# --------------------------------------------------------------------------
# verify


def test_verify_passes_a_complete_release(toy):
    assert verify.main(["--data-dir", str(toy["data"]), "--org", "toy",
                        "--release", "2026-08", "--sample", "10"]) == 0


def test_verify_fails_when_a_chunk_is_missing(toy, tmp_path):
    """The guard has to bite, or it is worse than not having it."""
    release = toy["release"]
    victim = release.omics_chunk("Neural", "chr1")
    saved = victim.read_bytes()
    victim.unlink()
    try:
        rc = verify.main(["--data-dir", str(toy["data"]), "--org", "toy",
                          "--release", "2026-08", "--sample", "0"])
    finally:
        victim.write_bytes(saved)
    assert rc == 1


def test_verify_fails_when_a_dna_chunk_is_truncated(toy):
    release = toy["release"]
    victim = release.dna_chunk("chr1", 0, 65536)
    saved = victim.read_text()
    victim.write_text(saved[:100])
    try:
        rc = verify.main(["--data-dir", str(toy["data"]), "--org", "toy",
                          "--release", "2026-08", "--sample", "0"])
    finally:
        victim.write_text(saved)
    assert rc == 1


# --------------------------------------------------------------------------
# migrate


@pytest.fixture
def legacy_tree(tmp_path):
    """A miniature of the 2021 layout, in the three-letter naming."""
    from joblib import dump

    data = tmp_path / "data"
    org = data / "old"
    (org / "DNA").mkdir(parents=True)
    (org / "DNA" / "old.fa").write_text(">chr1\nACGT\n")
    (org / "DNA" / "old-blacklist.v2.bed").write_text("chr1\t0\t10\tGap\n")

    for ag, ct, antigen in (("His", "Bld", "H3K27ac"), ("Oth", "Liv", "CTCF")):
        path = org / "OMICS" / ag / ct
        path.mkdir(parents=True)
        (path / ("05.%s.AllCell.bed" % antigen)).write_text("chr1\t0\t10\t500\n")

    for code in ("Bld", "Liv"):
        chunk = org / "Subtables" / "omics" / code / "chr1"
        chunk.mkdir(parents=True)
        (chunk / "0_65536.pkl").write_bytes(b"x")
    dna = org / "Subtables" / "dna" / "chr1"
    dna.mkdir(parents=True)
    (dna / "0_65536.txt").write_text("A" * 65536)
    (dna / "65536_100000.txt").write_text("C" * 34464)
    avl = org / "Subtables" / "avl"
    avl.mkdir(parents=True)
    dump(["H3K27ac"], avl / "Bld.pkl")
    dump(["CTCF"], avl / "Liv.pkl")

    support = org / "SupportFiles"
    support.mkdir(parents=True)
    dump(["H3K27ac", "CTCF"], support / "target_features.pkl")
    dump(["Bld", "Liv"], support / "target_tissues.pkl")
    dump(None, support / "old_DNA_seq.pkl")
    for name in ("full_intervals_8192.pkl", "train_intervals_8192.pkl",
                 "test_intervals_8192.pkl", "all_intervals.pkl"):
        (support / name).write_bytes(b"x")

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "fileList.tab").write_text(
        "His.Bld.05.AllAg.AllCell\told\tHistone\t-\tBlood\t-\t05\tSRX1\n"
        "Oth.Liv.05.AllAg.AllCell\told\tTFs and others\t-\tLiver\t-\t05\tSRX1\n")
    return {"data": data, "meta": meta, "org": org}


def test_migrate_dry_run_moves_nothing(legacy_tree):
    rc = migrate.main(["--data-dir", str(legacy_tree["data"]), "--org", "old",
                       "--release", "2021-10",
                       "--meta-dir", str(legacy_tree["meta"])])
    assert rc == 0
    assert (legacy_tree["org"] / "OMICS").is_dir(), "dry run touched the source"
    assert not (legacy_tree["org"] / "releases").exists(), "dry run left a release"


def test_migrate_renames_codes_to_full_names(legacy_tree):
    args = ["--data-dir", str(legacy_tree["data"]), "--org", "old",
            "--release", "2021-10", "--meta-dir", str(legacy_tree["meta"])]
    assert migrate.main(args + ["--execute"]) == 0
    release = layout.Release.open(legacy_tree["data"], "old", "2021-10")

    assert release.signal_path("Histone", "Blood", "H3K27ac").exists()
    assert release.signal_path("TFs and others", "Liver", "CTCF").exists()
    assert release.omics_chunk("Blood", "chr1", 0, 65536).exists()
    assert release.dna_chunk("chr1", 0, 65536).exists()
    assert release.path("genome_fasta").exists()
    assert release.path("windows_full", window=8192).exists()

    tissues = json.loads(release.path("tissues").read_text())["tissues"]
    assert [t["name"] for t in tissues] == ["Blood", "Liver"]
    assert [t["code"] for t in tissues] == ["Bld", "Liv"]
    assert json.loads(release.availability().read_text()) == {
        "Blood": ["H3K27ac"], "Liver": ["CTCF"]}


def test_migrate_reads_chromosome_lengths_off_the_chunk_names(legacy_tree):
    args = ["--data-dir", str(legacy_tree["data"]), "--org", "old",
            "--release", "2021-10", "--meta-dir", str(legacy_tree["meta"]),
            "--execute"]
    assert migrate.main(args) == 0
    release = layout.Release.open(legacy_tree["data"], "old", "2021-10")
    assert release.manifest["chrom_sizes"] == {"chr1": 100000}


def test_migrate_refuses_to_overwrite_an_existing_release(legacy_tree):
    args = ["--data-dir", str(legacy_tree["data"]), "--org", "old",
            "--release", "2021-10", "--meta-dir", str(legacy_tree["meta"]),
            "--execute"]
    assert migrate.main(args) == 0
    with pytest.raises(SystemExit):
        migrate.main(args)


def test_a_migrated_release_reads_back_through_the_same_api(legacy_tree):
    args = ["--data-dir", str(legacy_tree["data"]), "--org", "old",
            "--release", "2021-10", "--meta-dir", str(legacy_tree["meta"]),
            "--execute"]
    assert migrate.main(args) == 0
    release = layout.Release.open(legacy_tree["data"], "old", "2021-10")
    assert release.omics_layout == layout.CHUNKED_PICKLE
    assert release.interval_layout == layout.PER_SPLIT_PICKLE
    assert read.features(release) == ["H3K27ac", "CTCF"]
    assert read.tissues(release) == ["Blood", "Liver"]
    assert read.dna(release, "chr1", 0, 10) == "A" * 10
    # The old release genuinely has no validation holdout; asking must say so
    # rather than silently returning the test set.
    with pytest.raises(SystemExit):
        read.windows(release, 8192, split="val")


# --------------------------------------------------------------------------
# adopt


def test_adopt_hard_links_signal_and_groups_into_a_release(tmp_path):
    """The seam between the build tree and the release must not copy 18.8 GB."""
    from chipatlas_forge import adopt

    root = tmp_path / "forge"
    track = root / "out_binmax1" / "toy" / "Histone" / "Blood" / "H3K27ac.bedgraph"
    track.parent.mkdir(parents=True)
    track.write_text("chr1\t0\t10\t500\n")
    groups = root / "work" / "manifest" / "toy" / "groups.tsv"
    groups.parent.mkdir(parents=True)
    groups.write_text("group_id\tag_class\tantigen\tct_class\tn_experiments\tpath\n")

    data = tmp_path / "data"
    assert adopt.main(["--root", str(root), "--data-dir", str(data),
                       "--org", "toy", "--release", "2026-08"]) == 0

    release = layout.Release.open(data, "toy", "2026-08")
    landed = release.signal_path("Histone", "Blood", "H3K27ac")
    assert landed.exists()
    assert landed.samefile(track), "signal was copied instead of linked"
    assert release.path("groups").exists()
    assert release.manifest["stages"]["adopt"]["tracks"] == 1


def test_adopt_refuses_when_the_peak_stages_have_not_run(tmp_path):
    from chipatlas_forge import adopt

    with pytest.raises(SystemExit):
        adopt.main(["--root", str(tmp_path / "forge"),
                    "--data-dir", str(tmp_path / "data"),
                    "--org", "toy", "--release", "2026-08"])


def test_dna_chunks_can_be_linked_from_a_bare_tree(toy, tmp_path):
    """Building a release must not require migrating the tree still being read.

    A run in flight holds the pre-release `Subtables/dna`, so the new release
    links from it rather than copying or waiting for a maintenance window.
    """
    release = toy["release"]
    bare = tmp_path / "Subtables" / "dna"
    for chrom, length in release.manifest["chrom_sizes"].items():
        for start in range(0, length, release.chunk_size):
            end = min(start + release.chunk_size, length)
            target = bare / chrom / ("%d_%d.txt" % (start, end))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(release.dna_chunk(chrom, start, end).read_text())

    data = tmp_path / "data"
    fresh = layout.Release.create(data, "toy", "2027-01")
    fresh.manifest["chrom_sizes"] = release.manifest["chrom_sizes"]
    for stage in ("genome", "vocab", "intervals"):
        fresh.record(stage, inherited="test fixture")

    assert chunks.main(["--data-dir", str(data), "--org", "toy",
                        "--release", "2027-01", "--what", "dna",
                        "--dna-dir", str(bare)]) == 0
    fresh = layout.Release.open(data, "toy", "2027-01")
    assert fresh.manifest["stages"]["chunks_dna"]["linked"] > 0
    assert fresh.manifest["stages"]["chunks_dna"]["written"] == 0, "should link, not copy"
    linked = fresh.dna_chunk("chr1", 0, 65536)
    assert linked.samefile(bare / "chr1" / "0_65536.txt")
    assert linked.read_text() == toy["sequence"]["chr1"][:65536].upper()


def test_a_bare_tree_missing_the_final_chunk_is_cut_instead(toy, tmp_path):
    """The 2021 tree is not a clean tiling and cannot supply every chunk.

    Its final chunk per chromosome was pulled BACK to keep a full 65,536 bases,
    so chr1 ends at 248890886_248956422 rather than 248905728_248956422 and the
    last two overlap. A release tiles cleanly -- chunk index is
    start // chunk_size, which the parquet row-group mapping depends on -- so
    the names that cannot match are cut from the sequence rather than bending
    the grid to fit them.
    """
    release = toy["release"]
    bare = tmp_path / "dna"
    written_short = []
    for chrom, length in release.manifest["chrom_sizes"].items():
        for start in range(0, length, release.chunk_size):
            end = min(start + release.chunk_size, length)
            if end - start < release.chunk_size:
                written_short.append((chrom, start, end))
                continue                      # exactly what the old tree lacks
            target = bare / chrom / ("%d_%d.txt" % (start, end))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(release.dna_chunk(chrom, start, end).read_text())
    assert written_short, "fixture must have at least one short final chunk"

    data = tmp_path / "data"
    fresh = layout.Release.create(data, "toy", "2027-02")
    fresh.manifest["chrom_sizes"] = release.manifest["chrom_sizes"]
    for stage in ("vocab", "intervals"):
        fresh.record(stage, inherited="test fixture")
    fresh.record("genome", inherited="test fixture")
    genome.adopt(release.path("sequence"), fresh.path("sequence"))

    assert chunks.main(["--data-dir", str(data), "--org", "toy",
                        "--release", "2027-02", "--what", "dna",
                        "--dna-dir", str(bare)]) == 0
    fresh = layout.Release.open(data, "toy", "2027-02")
    stats = fresh.manifest["stages"]["chunks_dna"]
    assert stats["written"] == len(written_short), "short chunks must be cut"
    assert stats["linked"] == stats["chunks"] - stats["written"]

    for chrom, start, end in written_short:
        got = fresh.dna_chunk(chrom, start, end).read_text()
        assert got == toy["sequence"][chrom][start:end].upper()
        assert len(got) == end - start


def test_dna_source_options_are_mutually_exclusive(toy, tmp_path):
    with pytest.raises(SystemExit):
        chunks.main(["--data-dir", str(toy["data"]), "--org", "toy",
                     "--release", "2026-08", "--what", "dna",
                     "--from-release", "2026-08", "--dna-dir", str(tmp_path)])


def test_groups_tsv_keeps_NA_as_a_literal_antigen_name(tmp_path):
    """pandas' default NA list contains "NA", which is a real ChIP-Atlas value.

    Left alone it comes back as float nan, which fails loudly in `sorted` but
    would otherwise put a nan in the vocabulary. keys.py is explicit that NA is
    a category the source keeps, not a blank.
    """
    data = tmp_path / "data"
    release = layout.Release.create(data, "toy", "2026-08")
    rows = ["group_id\tag_class\tantigen\tct_class\tn_experiments\tpath"]
    gid = 0
    for tissue in ("Blood", "Liver", "Neural"):
        for antigen in ["NA"] + ["AG%03d" % i for i in range(60)]:
            rows.append("%d\tHistone\t%s\t%s\t10\tx" % (gid, antigen, tissue))
            gid += 1
    release.path("groups").parent.mkdir(parents=True, exist_ok=True)
    release.path("groups").write_text("\n".join(rows) + "\n")

    frame = vocab.read_groups(release)
    assert frame["antigen"].map(type).eq(str).all(), "a name became float nan"
    assert "NA" in set(frame["antigen"])

    tissues, antigens, _, _ = vocab.select(frame, 50, 3)
    assert "NA" in antigens, "NA must survive as an ordinary antigen"
    assert all(isinstance(a, str) for a in antigens)


def test_an_empty_class_in_groups_tsv_is_rejected(tmp_path):
    """manifest should have normalised it; a blank would slug to the same path."""
    data = tmp_path / "data"
    release = layout.Release.create(data, "toy", "2026-08")
    release.path("groups").parent.mkdir(parents=True, exist_ok=True)
    release.path("groups").write_text(
        "group_id\tag_class\tantigen\tct_class\tn_experiments\tpath\n"
        "0\tHistone\t\tBlood\t10\tx\n")
    with pytest.raises(SystemExit) as excinfo:
        vocab.read_groups(release)
    assert "antigen" in str(excinfo.value)


# --------------------------------------------------------------------------
# pantissue


def _runs(path):
    """A bedGraph as {(chrom, position): max value covering it}.

    The maximum, not the last writer: the fixture's per-tissue tracks contain
    overlapping intervals on purpose, so that the merge is exercised against
    input it would have to collapse rather than input that is already flat.
    """
    out = {}
    for line in path.read_text().splitlines():
        chrom, start, end, value = line.split("\t")
        value = int(value)
        for pos in range(int(start), int(end)):
            key = (chrom, pos)
            if value > out.get(key, 0):
                out[key] = value
    return out


def test_the_pan_tissue_track_is_the_max_over_every_tissue(toy):
    """Exact, per base, against a brute-force maximum.

    Each per-tissue track is already the max over that tissue's peaks, so the
    max across tissues is the max over every peak of the antigen anywhere --
    the same answer binmax would give on the union, without revisiting it.
    """
    release = toy["release"]
    antigen = ANTIGENS[0]
    ag_class = antigen_class(antigen).replace(" ", "_")

    expected = {}
    for tissue in TISSUES:
        for key, value in _runs(release.path("signal_root") / ag_class / tissue
                                / ("%s.bedgraph" % antigen)).items():
            expected[key] = max(expected.get(key, 0), value)

    got = _runs(release.path("signal_root") / ag_class / PAN
                / ("%s.bedgraph" % antigen))
    assert got == expected, "pan-tissue track is not the per-base maximum"


def test_the_pan_tissue_track_is_sorted_and_run_length_encoded(toy):
    """Later stages assume genomic order, and adjacent equal runs waste space."""
    release = toy["release"]
    antigen = ANTIGENS[0]
    path = (release.path("signal_root") / antigen_class(antigen).replace(" ", "_")
            / PAN / ("%s.bedgraph" % antigen))
    rows = [line.split("\t") for line in path.read_text().splitlines()]
    seen, previous = [], None
    for chrom, start, end, value in rows:
        if chrom != (previous or (None,))[0]:
            assert chrom not in seen, "chromosome %s is not contiguous" % chrom
            seen.append(chrom)
        else:
            assert int(start) >= int(previous[2]), "runs overlap or are unsorted"
            if int(start) == int(previous[2]):
                assert value != previous[3], "adjacent equal runs were not merged"
        previous = (chrom, start, end, value)


def test_pantissue_never_merges_its_own_output(toy):
    """Re-running must be idempotent, not compound the pan track into itself."""
    work = pantissue.plan(toy["release"])
    for _, sources in work:
        assert all(p.parent.name != PAN for p in sources)


def test_sequence_is_read_through_joblib_not_raw_pickle(tmp_path):
    """The pre-release sequence.pkl is joblib-compressed, not a plain pickle.

    Raw `pickle.load` on it fails with `UnpicklingError: invalid load key, 'x'`
    -- 'x' being the first byte of the zlib magic. It failed on the cluster only
    after adopt and genome had both succeeded, because nothing before this stage
    opens the file.
    """
    from joblib import dump

    path = tmp_path / "sequence.pkl"
    dump({"chr1": "ACGTNNNN", "chr2": "TTTT"}, path, compress=3)
    with pytest.raises(Exception):
        with open(path, "rb") as fh:
            pickle.load(fh)
    assert genome.load_sequence(path) == {"chr1": "ACGTNNNN", "chr2": "TTTT"}
    assert genome.chrom_sizes_from_sequence(path) == {"chr1": 8, "chr2": 4}


def test_a_plain_pickled_sequence_still_reads(tmp_path):
    path = tmp_path / "sequence.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"chr1": "ACGT"}, fh)
    assert genome.load_sequence(path) == {"chr1": "ACGT"}


# --------------------------------------------------------------------------
# the slugified-directory bug


def test_signal_files_finds_a_tissue_whose_directory_is_slugified(toy):
    """`keys.group_path` slugifies, so a multi-word tissue is not its own name.

    On the real data this silently emptied five tissues -- "Pluripotent stem
    cell", "Digestive tract", "Embryonic fibroblast" and two more, ~1,500
    features between them -- because `signal/Histone/Pluripotent_stem_cell/`
    does not match the vocabulary's "Pluripotent stem cell", and an empty result
    is also what "nothing here" legitimately looks like.
    """
    release = toy["release"]
    ag_dir = release.path("signal_root") / "Histone"
    slugged = ag_dir / "Pluripotent_stem_cell"
    slugged.mkdir(parents=True, exist_ok=True)
    (slugged / ("%s.bedgraph" % ANTIGENS[0])).write_text("chr1\t0\t10\t500\n")
    try:
        found = chunks.signal_files(release, "Pluripotent stem cell", ANTIGENS)
        assert set(found) == {ANTIGENS[0]}, "slugified directory was not found"
        assert found[ANTIGENS[0]].parent.name == "Pluripotent_stem_cell"
    finally:
        (slugged / ("%s.bedgraph" % ANTIGENS[0])).unlink()
        slugged.rmdir()


def test_signal_files_still_finds_an_unslugified_directory(toy):
    """`pantissue` writes the plain name, so both spellings must resolve."""
    found = chunks.signal_files(toy["release"], PAN, ANTIGENS)
    assert len(found) == len(ANTIGENS)


def test_verify_catches_a_tissue_with_features_but_no_rows(toy, tmp_path):
    """The check that the sampled comparison structurally cannot make.

    `verify_sampled_chunks` re-derives through the same lookup that built the
    chunk, so when the lookup is what is broken, expected and actual are both
    empty and it reports a match. A release passed 760,779 checks that way with
    ~1,500 features unavailable.
    """
    release = toy["release"]
    victim = release.omics_chunk("Blood", "chr1")
    saved = {c: release.omics_chunk("Blood", c).read_bytes()
             for c in release.manifest["chrom_sizes"]
             if release.omics_chunk("Blood", c).exists()}
    try:
        # Empty every chromosome for Blood, exactly as the slug bug did.
        for chrom in release.manifest["chrom_sizes"]:
            path = release.omics_chunk("Blood", chrom)
            if path.exists():
                import pyarrow.parquet as _pq
                _pq.ParquetWriter(
                    path, chunks.out_schema().with_metadata(
                        {chunks.CHUNK_KEY: b"[]"})).close()
        rc = verify.main(["--data-dir", str(toy["data"]), "--org", "toy",
                          "--release", "2026-08", "--sample", "0"])
    finally:
        for chrom, blob in saved.items():
            release.omics_chunk("Blood", chrom).write_bytes(blob)
    assert rc == 1, "verify passed a release with a silently empty tissue"


def test_a_feature_whose_file_matches_neither_spelling_is_still_found(toy):
    """`H3PERIOD3_K27M_mutant` -> alias `H3.3 K27M mutant` -> file
    `H3.3_K27M_mutant.bedgraph`. All three differ; only their slugs agree."""
    index = chunks.feature_index(["H3PERIOD3_K27M_mutant"],
                                 {"H3PERIOD3_K27M_mutant": "H3.3 K27M mutant"})
    from chipatlas_forge.keys import slugify
    assert index[slugify("H3.3_K27M_mutant")] == "H3PERIOD3_K27M_mutant"
    assert index[slugify("H3PERIOD3_K27M_mutant")] == "H3PERIOD3_K27M_mutant"


def test_two_features_sharing_a_filename_slug_are_refused(toy):
    """Whichever won would take the other's peaks into its column."""
    with pytest.raises(SystemExit) as excinfo:
        chunks.feature_index(["A B", "A_B"])
    assert "wrong column" in str(excinfo.value)


def test_signal_files_finds_the_awkward_alias_on_disk(toy):
    release = toy["release"]
    path = release.path("signal_root") / "Histone" / "Blood" / "H3.3_K27M_mutant.bedgraph"
    path.write_text("chr1\t0\t10\t500\n")
    try:
        found = chunks.signal_files(release, "Blood", ["H3PERIOD3_K27M_mutant"],
                                    {"H3PERIOD3_K27M_mutant": "H3.3 K27M mutant"})
        assert set(found) == {"H3PERIOD3_K27M_mutant"}
    finally:
        path.unlink()


def test_verify_catches_a_feature_no_file_resolves_to(toy):
    """One unreachable column among hundreds is invisible to a per-tissue count."""
    release = toy["release"]
    avail = json.loads(release.availability().read_text())
    saved = release.availability().read_text()
    features_blob = json.loads(release.path("features").read_text())
    saved_features = release.path("features").read_text()
    try:
        # Claim a feature that has no signal file anywhere.
        avail["Blood"] = sorted(set(avail["Blood"]) | {"GHOST_ANTIGEN"})
        release.availability().write_text(json.dumps(avail))
        features_blob["features"] = features_blob["features"] + ["GHOST_ANTIGEN"]
        release.path("features").write_text(json.dumps(features_blob))
        rc = verify.main(["--data-dir", str(toy["data"]), "--org", "toy",
                          "--release", "2026-08", "--sample", "0"])
    finally:
        release.availability().write_text(saved)
        release.path("features").write_text(saved_features)
    assert rc == 1, "verify passed a release with an unreachable feature"


# --------------------------------------------------------------------------
# concurrent manifest updates


def test_a_slow_stage_does_not_erase_what_a_fast_one_recorded(tmp_path):
    """Two stages, each holding the snapshot it opened with.

    This really happened: hg38's `chunks_dna` finished at 01:22 and was gone
    from the manifest by 01:57, erased by `pairs`, which had opened the release
    a minute earlier. The data was fine -- but `require()` reads the manifest to
    decide a stage has run, and `verify`'s own entry is what gates promotion.
    """
    data = tmp_path / "data"
    layout.Release.create(data, "toy", "2026-08")

    slow = layout.Release.open(data, "toy", "2026-08")     # opens first
    fast = layout.Release.open(data, "toy", "2026-08")     # opens second
    fast.record("chunks_dna", chunks=47137)                # finishes first
    slow.record("pairs", n_pairs=1106074)                  # finishes last

    fresh = layout.Release.open(data, "toy", "2026-08")
    assert set(fresh.manifest["stages"]) == {"chunks_dna", "pairs"}
    assert fresh.manifest["stages"]["chunks_dna"]["chunks"] == 47137
    assert fresh.has("chunks_dna") and fresh.has("pairs")


def test_the_manifest_is_never_half_written(tmp_path):
    """A reader mid-write would otherwise get a JSONDecodeError."""
    data = tmp_path / "data"
    release = layout.Release.create(data, "toy", "2026-08")
    release.record("verify", checks=760779)
    path = layout.release_root(data, "toy", "2026-08") / "MANIFEST.json"
    assert json.loads(path.read_text())["stages"]["verify"]["checks"] == 760779
    leftovers = list(path.parent.glob("MANIFEST.json.tmp*"))
    assert not leftovers, "temporary manifest was left behind: %s" % leftovers
