"""The max-score track, checked against brute force at every single base.

The compressed-breakpoint algorithm is an optimisation of something with a
trivial definition -- for each base, the largest score among peaks covering it.
So the tests compute that definition directly with a dense array and demand the
fast version agree everywhere, including on randomised pile-ups. If the
compression is ever wrong it will be wrong at a boundary, which is exactly what
a dense comparison catches and a spot-check does not.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipatlas_forge.binmax import max_runs  # noqa: E402


def brute_force(starts, ends, scores, length, bin_size=1):
    """The definition, written the slow obvious way."""
    dense = np.zeros(length, dtype=np.int32)
    for s, e, v in zip(starts, ends, scores):
        dense[s:e] = np.maximum(dense[s:e], v)
    if bin_size == 1:
        return dense
    n_bins = -(-length // bin_size)
    binned = np.zeros(n_bins, dtype=np.int32)
    for b in range(n_bins):
        window = dense[b * bin_size:(b + 1) * bin_size]
        if len(window):
            binned[b] = window.max()
    return binned


def expand(run_start, run_end, run_value, length):
    """Runs back to a dense array, for comparison."""
    dense = np.zeros(length, dtype=np.int32)
    for s, e, v in zip(run_start, run_end, run_value):
        dense[s:e] = v
    return dense


class TestAgainstBruteForce:
    def test_the_examples_from_the_real_file(self):
        """The four overlapping peaks that prompted this stage."""
        starts = np.array([9915, 9919, 9921, 9924])
        ends = np.array([10410, 10288, 10441, 10295])
        scores = np.array([644, 377, 643, 478], dtype=np.int32)

        rs, re, rv = max_runs(starts, ends, scores)
        got = expand(rs, re, rv, 11000)
        want = brute_force(starts, ends, scores, 11000)
        assert np.array_equal(got, want)

        # Before any peak and after all of them there must be no signal.
        assert got[9914] == 0 and got[10441] == 0
        # The tallest peak covers 9915..10410 and nothing beats 644 there.
        assert got[9915] == 644 and got[10409] == 644
        # 10410..10441 is covered only by the 643 peak.
        assert got[10410] == 643 and got[10440] == 643

    @pytest.mark.parametrize("seed", range(12))
    def test_random_pileups_match_everywhere(self, seed):
        rng = np.random.default_rng(seed)
        n, length = 400, 5000
        starts = rng.integers(0, length - 60, size=n)
        ends = starts + rng.integers(1, 600, size=n)
        ends = np.minimum(ends, length)
        scores = rng.integers(1, 3000, size=n).astype(np.int32)

        rs, re, rv = max_runs(starts, ends, scores)
        assert np.array_equal(expand(rs, re, rv, length),
                              brute_force(starts, ends, scores, length))

    @pytest.mark.parametrize("bin_size", [1, 2, 8, 128, 1024])
    def test_binned_matches_brute_force(self, bin_size):
        rng = np.random.default_rng(7)
        n, length = 300, 8192
        starts = rng.integers(0, length - 100, size=n)
        ends = np.minimum(starts + rng.integers(1, 700, size=n), length)
        scores = rng.integers(1, 2000, size=n).astype(np.int32)

        rs, re, rv = max_runs(starts, ends, scores, bin_size)
        # Runs come back in bin units; compare on the bin grid.
        n_bins = -(-length // bin_size)
        got = expand(rs, re, rv, n_bins + 2)[:n_bins]
        want = brute_force(starts, ends, scores, length, bin_size)
        assert np.array_equal(got, want)

    def test_ties_do_not_lose_coverage(self):
        """Equal scores must still paint their full extent."""
        starts = np.array([0, 50])
        ends = np.array([100, 150])
        scores = np.array([500, 500], dtype=np.int32)
        rs, re, rv = max_runs(starts, ends, scores)
        got = expand(rs, re, rv, 200)
        assert got[:150].min() == 500 and got[150:].max() == 0


class TestRunEncoding:
    def test_adjacent_equal_runs_are_merged(self):
        """Two touching peaks of the same score are one run, not two."""
        rs, re, rv = max_runs(np.array([0, 100]), np.array([100, 200]),
                              np.array([7, 7], dtype=np.int32))
        assert list(zip(rs.tolist(), re.tolist(), rv.tolist())) == [(0, 200, 7)]

    def test_gaps_between_peaks_are_omitted(self):
        rs, re, rv = max_runs(np.array([0, 500]), np.array([100, 600]),
                              np.array([3, 9], dtype=np.int32))
        assert list(zip(rs.tolist(), re.tolist(), rv.tolist())) == [(0, 100, 3),
                                                                   (500, 600, 9)]

    def test_runs_are_sorted_and_disjoint(self):
        rng = np.random.default_rng(3)
        starts = rng.integers(0, 4000, size=500)
        ends = starts + rng.integers(1, 400, size=500)
        scores = rng.integers(1, 999, size=500).astype(np.int32)
        rs, re, rv = max_runs(starts, ends, scores)
        assert np.all(rs < re), "a run must be non-empty"
        assert np.all(re[:-1] <= rs[1:]), "runs must not overlap"
        assert np.all(np.diff(rs) > 0), "runs must be ordered"

    def test_no_run_carries_a_zero(self):
        rng = np.random.default_rng(11)
        starts = rng.integers(0, 2000, size=200)
        ends = starts + rng.integers(1, 300, size=200)
        scores = rng.integers(1, 500, size=200).astype(np.int32)
        _, _, rv = max_runs(starts, ends, scores)
        assert rv.min() > 0

    def test_empty_input(self):
        rs, re, rv = max_runs(np.empty(0, np.int64), np.empty(0, np.int64),
                              np.empty(0, np.int32))
        assert len(rs) == len(re) == len(rv) == 0

    def test_a_peak_shorter_than_one_bin_still_claims_it(self):
        """floor/ceil must never collapse a peak to nothing."""
        rs, re, rv = max_runs(np.array([1000]), np.array([1010]),
                              np.array([42], dtype=np.int32), bin_size=128)
        assert len(rs) == 1 and rv[0] == 42
        assert rs[0] == 1000 // 128 and re[0] == rs[0] + 1
