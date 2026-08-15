"""The Arrow <-> numpy bridge, and a guard against regressing onto the broken API.

On the cluster, pyarrow 20 sits next to numpy 1.26.3 and every documented
conversion between them raises -- see arrow_compat's docstring. The bridge here
is the only sanctioned crossing. The last test in this file is the important
one: it fails if anyone reaches for ``.to_numpy()`` or ``pa.array(ndarray)``
again, which works fine on a developer laptop and dies on the cluster.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipatlas_forge import arrow_compat as compat  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1] / "chipatlas_forge"


class TestBridge:
    @pytest.mark.parametrize("dtype,arrow_type", [
        (np.int32, pa.int32()), (np.int64, pa.int64()),
        (np.float32, pa.float32()), (np.uint8, pa.uint8()),
    ])
    def test_round_trip_preserves_values(self, dtype, arrow_type):
        values = np.arange(1000, dtype=dtype)
        back = compat.to_numpy(compat.to_arrow(values, arrow_type), dtype)
        assert np.array_equal(values, back)

    def test_arrow_type_is_inferred_from_the_numpy_dtype(self):
        arr = compat.to_arrow(np.arange(4, dtype=np.int32))
        assert arr.type == pa.int32()

    def test_a_slice_reads_from_its_own_offset(self):
        """A sliced Arrow array shares the parent's buffer; ignoring the offset
        silently returns the wrong rows rather than failing."""
        arr = pa.array(list(range(100)), type=pa.int32()).slice(10, 5)
        assert compat.to_numpy(arr, np.int32).tolist() == [10, 11, 12, 13, 14]

    def test_chunked_arrays_are_combined(self):
        chunked = pa.chunked_array([pa.array([1, 2], type=pa.int32()),
                                    pa.array([3, 4], type=pa.int32())])
        assert compat.to_numpy(chunked, np.int32).tolist() == [1, 2, 3, 4]

    def test_empty_is_empty_not_an_error(self):
        assert len(compat.to_numpy(pa.array([], type=pa.int32()), np.int32)) == 0

    def test_nulls_are_refused_rather_than_read_as_garbage(self):
        """Arrow leaves the values buffer undefined at null positions, so a
        silent read returns whatever was there."""
        with pytest.raises(ValueError, match="nulls"):
            compat.to_numpy(pa.array([1, None, 3], type=pa.int32()), np.int32)

    def test_take_accepts_a_bridged_permutation(self):
        table = pa.table({"v": pa.array([3, 1, 2, 0], type=pa.int32())})
        order = np.argsort(np.array([3, 1, 2, 0]), kind="stable").astype(np.int64)
        taken = table.take(compat.to_arrow(order, pa.int64()))
        assert taken.column("v").to_pylist() == [0, 1, 2, 3]

    def test_the_arrow_array_outlives_the_numpy_handle(self):
        """py_buffer must hold the reference; if it does not this reads freed
        memory, which fails intermittently rather than loudly."""
        def make():
            return compat.to_arrow(np.arange(10_000, dtype=np.int64))
        arr = make()
        assert arr.to_pylist()[:3] == [0, 1, 2]
        assert arr.to_pylist()[-1] == 9999


class TestNobodyUsesTheBrokenAPI:
    """pyarrow 20 + numpy 1.26 makes these raise; they work on a laptop."""

    # `compat.` calls are the sanctioned bridge, not the broken converter.
    FORBIDDEN = [
        (re.compile(r"(?<!compat)\.to_numpy\s*\("),
         "Array.to_numpy is ABI-broken here; use arrow_compat.to_numpy"),
        (re.compile(r"pa\.array\s*\(\s*(?!\[|\(\s*\)|list|range)[a-z_]+\s*(,|\))"),
         "pa.array(ndarray) is ABI-broken here; use arrow_compat.to_arrow"),
    ]

    def test_package_sources_avoid_them(self):
        offences = []
        for path in sorted(PACKAGE.glob("*.py")):
            if path.name == "arrow_compat.py":
                continue                      # documents them on purpose
            source = path.read_text()
            if "import pyarrow" not in source:
                # pandas' Series.to_numpy is a different method and is fine;
                # only modules that actually hold Arrow objects can trip this.
                continue
            for number, line in enumerate(source.splitlines(), 1):
                code = line.split("#", 1)[0]
                for pattern, why in self.FORBIDDEN:
                    if pattern.search(code):
                        offences.append("%s:%d  %s\n    %s"
                                        % (path.name, number, why, line.strip()))
        assert not offences, "\n".join(offences)

    def test_the_guard_would_actually_catch_a_regression(self, tmp_path):
        """A guard that cannot fail is worse than no guard."""
        bad = "import pyarrow as pa\nvalues = table.column('x').to_numpy()\n"
        hits = [why for pattern, why in self.FORBIDDEN if pattern.search(bad)]
        assert hits, "the forbidden-pattern regex no longer matches the real mistake"
