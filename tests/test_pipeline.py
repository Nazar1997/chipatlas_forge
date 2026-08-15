"""End-to-end properties of the four stages, on a synthetic genome.

The pipeline's whole reason for existing is that it moves ~1.5 billion peaks
through a shuffle without losing any and without sorting. Both of those are
invisible in a SLURM exit code and expensive to check at scale, so they are
pinned here on data small enough to verify exhaustively:

* **conservation** -- every input peak appears exactly once across the outputs,
  in the file its accession's group says it should
* **order** -- each output BED is sorted by (chrom, start), which the pipeline
  gets for free from shard order and stable sorts, and would lose silently if
  anyone swapped in an unstable sort
* **no silent drops** -- an accession missing from the manifest is counted and
  reported, never quietly discarded

The fixture deliberately reproduces the two structural features of the real
archive that the design leans on: peaks are globally sorted by position, and
accessions are scattered throughout rather than clustered.
"""

import gzip
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipatlas_forge import collect, keys, manifest, route, shard  # noqa: E402

ORG = "hg38"
CHROMS = ["chr1", "chr2", "chrX"]
N_EXPERIMENTS = 60
N_PEAKS = 20_000


def _experiment_rows(rng):
    """(srx, org, ag_class, antigen, ct_class, celltype) for the meta CSV."""
    ag_classes = ["Histone", "TFs and others"]
    antigens = ["H3K27ac", "H3K4me3", "CTCF", "POLR2A"]
    ct_classes = ["Blood", "Liver", "Neural"]
    rows = []
    for i in range(N_EXPERIMENTS):
        prefix = ["SRX", "ERX", "DRX"][i % 3]
        rows.append((
            "%s%07d" % (prefix, 1000 + i), ORG,
            rng.choice(ag_classes), rng.choice(antigens),
            rng.choice(ct_classes), "cell_%d" % (i % 7),
        ))
    # An experiment on another assembly with a colliding accession: the manifest
    # is per-org precisely so this cannot leak into hg38's lookup.
    rows.append(("SRX0001000", "mm10", "Histone", "H3K9me3", "Liver", "other"))
    return rows


@pytest.fixture
def project(tmp_path):
    """A complete miniature project: meta/ zip plus a sorted raw archive."""
    rng = random.Random(20260815)
    root = tmp_path / "forge"
    (root / "meta").mkdir(parents=True)
    (root / "raw").mkdir(parents=True)

    rows = _experiment_rows(rng)
    header = ("Experimental ID,Genome assembly,Antigen class,Antigen,"
              "Cell type class,Cell type,Title\n")
    body = "".join(",".join(r) + ",some title\n" for r in rows)
    csv_bytes = (header + body).encode("latin-1")
    with zipfile.ZipFile(root / "meta" / "chip_atlas_experiment_list.zip", "w") as zf:
        zf.writestr("chip_atlas_experiment_list.csv", csv_bytes)

    # Peaks sorted by (chrom, start), accessions scattered -- as in the real file.
    accessions = [r[0] for r in rows if r[1] == ORG]
    peaks = []
    for chrom in CHROMS:
        starts = sorted(rng.randrange(0, 5_000_000) for _ in range(N_PEAKS // len(CHROMS)))
        for start in starts:
            peaks.append((chrom, start, start + rng.randrange(50, 500),
                          rng.choice(accessions), rng.randrange(10, 2000)))
    # Two peaks whose accession the manifest has never heard of.
    unknown = [(CHROMS[0], 10, 200, "SRX9999999", 5),
               (CHROMS[0], 20, 300, "SRX9999998", 6)]
    peaks = unknown + peaks

    raw = root / "raw" / ("allPeaks_light.%s.05.bed.gz" % ORG)
    with gzip.open(raw, "wb") as fh:
        for p in peaks:
            fh.write(("%s\t%d\t%d\t%s\t%d\n" % p).encode())

    return root, peaks, dict((r[0], r) for r in rows if r[1] == ORG)


def _run(root, chunk="64K", buckets=8):
    manifest.main(["--root", str(root), "--org", ORG])
    shard.main(["--root", str(root), "--org", ORG, "--chunk-size", chunk])
    route.main(["--root", str(root), "--org", ORG, "--shard", "all",
                "--buckets", str(buckets)])
    collect.main(["--root", str(root), "--org", ORG, "--bucket", "all",
                  "--buckets", str(buckets)])


def _read_outputs(root):
    """Every output row, tagged with the file it came from."""
    frames = []
    for path in sorted((root / "out").rglob("*.bed")):
        frame = pd.read_csv(path, sep="\t", header=None,
                            names=["chrom", "start", "end", "srx", "score"])
        frame["path"] = str(path.relative_to(root / "out"))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class TestEndToEnd:
    def test_every_peak_survives_exactly_once(self, project):
        root, peaks, _ = project
        _run(root)
        out = _read_outputs(root)

        known = [p for p in peaks if not p[3].startswith("SRX999")]
        assert len(out) == len(known), "peak count changed through the shuffle"

        as_tuples = sorted(map(tuple, out[["chrom", "start", "end", "srx", "score"]].values))
        assert as_tuples == sorted(known), "the multiset of peaks changed"

    def test_each_peak_lands_in_its_own_group_file(self, project):
        root, _, experiments = project
        _run(root)
        out = _read_outputs(root)

        for path, chunk in out.groupby("path"):
            for srx in chunk["srx"].unique():
                _, _, ag_class, antigen, ct_class, _ = experiments[srx]
                assert path == keys.group_path(ORG, ag_class, antigen, ct_class), (
                    "%s was written to %s" % (srx, path)
                )

    def test_outputs_are_sorted_without_a_sort_step(self, project):
        root, _, _ = project
        _run(root)
        files = sorted((root / "out").rglob("*.bed"))
        assert files, "no output produced"
        for path in files:
            frame = pd.read_csv(path, sep="\t", header=None,
                                names=["chrom", "start", "end", "srx", "score"])
            for chrom, chunk in frame.groupby("chrom", sort=False):
                assert chunk["start"].is_monotonic_increasing, (
                    "%s is out of order on %s -- an unstable sort crept in"
                    % (path.name, chrom)
                )

    def test_unmapped_accessions_are_counted_not_dropped_quietly(self, project):
        root, _, _ = project
        _run(root)
        stats = [pd.read_json(p, typ="series")
                 for p in sorted((root / "work" / "stats" / ORG).glob("*.json"))]
        assert sum(int(s["unmapped"]) for s in stats) == 2

    def test_result_is_identical_at_a_different_bucket_count(self, project):
        """Bucketing is a scheduling detail; it must not change the output."""
        root, _, _ = project
        _run(root, buckets=8)
        first = _read_outputs(root).sort_values(
            ["chrom", "start", "srx"]).reset_index(drop=True)

        subprocess.run(["rm", "-rf", str(root / "out"), str(root / "work")], check=True)
        _run(root, buckets=3)
        second = _read_outputs(root).sort_values(
            ["chrom", "start", "srx"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(first, second)

    def test_shards_concatenate_back_to_the_original(self, project):
        """The invariant the whole no-sort design rests on.

        If shards in index order do not reproduce the input exactly, then
        genomic order is not preserved and every output BED is quietly
        unsorted -- with no error anywhere to say so.
        """
        root, _, _ = project
        shard.main(["--root", str(root), "--org", ORG, "--chunk-size", "32K"])

        rebuilt = []
        for path in sorted((root / "work" / "shards" / ORG).glob("shard_*")):
            stream, proc = route.open_shard(path)
            rebuilt.append(stream.read())
            stream.close()
            if proc is not None:
                proc.wait()

        with gzip.open(root / "raw" / ("allPeaks_light.%s.05.bed.gz" % ORG), "rb") as fh:
            assert b"".join(rebuilt) == fh.read()

    def test_shards_never_split_a_record(self, project):
        """Every shard must parse on its own -- stage 2 reads them independently."""
        root, _, _ = project
        manifest.main(["--root", str(root), "--org", ORG])
        shard.main(["--root", str(root), "--org", ORG, "--chunk-size", "16K"])
        shards = sorted((root / "work" / "shards" / ORG).glob("shard_*"))
        assert len(shards) > 3, "chunk size did not produce several shards"
        for path in shards:
            stream, proc = route.open_shard(path)
            text = stream.read().decode()
            stream.close()
            if proc is not None:
                proc.wait()
            assert text.endswith("\n"), "%s ends mid-record" % path.name
            for line in text.splitlines():
                assert len(line.split("\t")) == 5, "%s has a torn line" % path.name


class TestMetadataSource:
    """experimentList.tab is the default source; the bundled zip is a fallback.

    The zip in the Yandex share is stamped October 2021 and misses ~3.2% of the
    accessions the peak archives cite -- ~48 million peaks at full scale. The
    live list carries 845,824 experiments against the zip's 439,593.
    """

    def _write_tab(self, meta_dir, rows):
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / manifest.LIVE_LIST, "w", encoding="latin-1") as fh:
            for row in rows:
                fh.write("\t".join(row) + "\n")

    def test_ragged_trailing_columns_do_not_shift_the_fields(self, tmp_path):
        """The free-text metadata column contains tabs of its own, so rows have
        varying field counts. A parser told to expect six columns misaligns."""
        self._write_tab(tmp_path / "meta", [
            ["SRX1", "hg38", "Histone", "H3K27ac", "Blood", "K562",
             "desc", "logs", "a title with\ttabs\tinside", "meta=1 || meta=2"],
            ["SRX2", "hg38", "Histone", "H3K4me3", "Liver", "HepG2"],
        ])
        frame = manifest.read_experiment_list(tmp_path / "meta")
        assert list(frame["srx"]) == ["SRX1", "SRX2"]
        assert list(frame["antigen"]) == ["H3K27ac", "H3K4me3"]
        assert list(frame["ct_class"]) == ["Blood", "Liver"]

    def test_rows_too_short_to_identify_are_skipped(self, tmp_path):
        self._write_tab(tmp_path / "meta", [
            ["SRX1", "hg38", "Histone", "H3K27ac", "Blood", "K562"],
            ["SRX2", "hg38", "Histone"],            # truncated, unusable
        ])
        frame = manifest.read_experiment_list(tmp_path / "meta")
        assert list(frame["srx"]) == ["SRX1"]

    def test_live_list_wins_over_the_stale_zip(self, tmp_path, project):
        """Both present: the fresher one must be chosen without being asked."""
        root, _, _ = project
        assert (root / "meta" / manifest.BUNDLED_ZIP).exists()
        self._write_tab(root / "meta", [
            ["SRXFRESH", "hg38", "Histone", "H3K9me3", "Bone", "cell"],
        ])
        frame = manifest.read_experiment_list(root / "meta")
        assert list(frame["srx"]) == ["SRXFRESH"]

        forced = manifest.read_experiment_list(root / "meta", "bundled")
        assert "SRXFRESH" not in set(forced["srx"])

    def test_cp1252_bytes_do_not_raise(self, tmp_path):
        """The real files carry smart quotes; utf-8 decoding dies on them."""
        meta = tmp_path / "meta"
        meta.mkdir()
        with open(meta / manifest.LIVE_LIST, "wb") as fh:
            fh.write(b"SRX1\thg38\tHistone\tH3K27ac\tBlood\tK562\ttitle \x91quoted\x92\n")
        frame = manifest.read_experiment_list(meta)
        assert list(frame["antigen"]) == ["H3K27ac"]


class TestGroupKeys:
    def test_slug_collisions_are_refused_not_merged(self):
        """Two antigens slugging to one path would silently share a BED."""
        frame = pd.DataFrame({
            "srx": ["SRX1", "SRX2"],
            "org": [ORG, ORG],
            "ag_class": ["Histone", "Histone"],
            "antigen": ["CD4/CD8", "CD4 CD8"],   # both slug to CD4_CD8
            "ct_class": ["Blood", "Blood"],
            "celltype": ["a", "b"],
        })
        with pytest.raises(SystemExit, match="claimed by more than one group"):
            manifest.build_for_org(frame, ORG, keys.DEFAULT_GROUP_FIELDS)

    def test_genuinely_absent_values_collapse_to_one_token(self):
        for spelling in ("", "-", "N/A", "   "):
            assert keys.normalise_field(spelling) == "NA"
        assert keys.normalise_field("H3K27ac") == "H3K27ac"

    def test_unclassified_and_no_description_stay_distinct(self):
        """They look like placeholders but are real ChIP-Atlas antigen classes,
        14,221 and 10,449 hg38 experiments -- 12.5% of the assembly. Collapsing
        them merges two categories the source deliberately keeps apart."""
        assert keys.normalise_field("Unclassified") == "Unclassified"
        assert keys.normalise_field("No description") == "No description"
        assert (keys.group_path("hg38", "Unclassified", "Unclassified", "Blood")
                != keys.group_path("hg38", "No description", "NA", "Blood"))

    def test_slugify_keeps_paths_single_segment(self):
        for name in ("CD4/CD8", "TF (isoform 2)", "α-tubulin", "a  b"):
            assert "/" not in keys.slugify(name)
            assert keys.slugify(name)
