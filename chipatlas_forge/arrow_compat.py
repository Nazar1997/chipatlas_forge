"""Move data between Arrow and numpy without using pyarrow's numpy converter.

cHARISMa's ``myenv`` pairs **pyarrow 20 with numpy 1.26.3**, and pyarrow 20 is
built against numpy 2.x. The two ABIs disagree, and the entire documented
conversion API is unusable as a result -- in both directions:

    Array.to_numpy(...)      AttributeError: module 'numpy.core.multiarray'
                             has no attribute 'complexfloating'
    pa.array(ndarray)        ArrowTypeError: Did not pass numpy.dtype object
    pa.array(nd, type=...)   ArrowTypeError: Input object was not a NumPy array
    table.take(ndarray)      ArrowTypeError: Did not pass numpy.dtype object

This is the same failure the joint-training pair builder hit with torch 2.5.1
against numpy 1.26.3, where ``from_numpy``, ``as_tensor`` and ``tensor`` all
raised "expected np.ndarray (got numpy.ndarray)". Different library, identical
cause: a C extension compiled against numpy 2 loaded next to numpy 1.

The way through is the **buffer protocol**, which neither side type-checks
against the other's dtype objects. ``np.frombuffer`` reads an Arrow values
buffer, and ``pa.py_buffer`` + ``Array.from_buffers`` wraps a numpy block as an
Arrow array. Both are zero-copy, so this is not a workaround with a cost -- it
is strictly less work than the converter would have done.

Fixing the environment instead is not obviously available: numpy cannot go to
2.x without breaking torch 2.5.1+cu118, which the training code needs.

Only null-free primitive arrays are handled, which is all this pipeline
produces. Call ``fill_null`` before converting anything that might have nulls.
"""

import numpy as np
import pyarrow as pa

_ARROW_FOR_NUMPY = {
    np.dtype(np.int8): pa.int8(), np.dtype(np.int16): pa.int16(),
    np.dtype(np.int32): pa.int32(), np.dtype(np.int64): pa.int64(),
    np.dtype(np.uint8): pa.uint8(), np.dtype(np.uint16): pa.uint16(),
    np.dtype(np.uint32): pa.uint32(), np.dtype(np.uint64): pa.uint64(),
    np.dtype(np.float32): pa.float32(), np.dtype(np.float64): pa.float64(),
}


def to_arrow(values: np.ndarray, arrow_type=None) -> pa.Array:
    """Wrap a contiguous numpy block as an Arrow array, zero-copy.

    ``pa.py_buffer`` keeps a reference to ``values``, so the Arrow array owns
    its memory for as long as it is alive even if the caller drops the numpy
    handle.
    """
    values = np.ascontiguousarray(values)
    if arrow_type is None:
        arrow_type = _ARROW_FOR_NUMPY.get(values.dtype)
        if arrow_type is None:
            raise TypeError("no Arrow type registered for %s" % values.dtype)
    return pa.Array.from_buffers(arrow_type, len(values),
                                 [None, pa.py_buffer(values)])


def to_numpy(array, dtype) -> np.ndarray:
    """Read an Arrow array's values buffer as numpy, zero-copy.

    Accepts a ChunkedArray as well; chunks are combined first, which copies only
    when there is more than one of them.

    The result is a **read-only view** onto Arrow-owned memory. Copy it before
    mutating in place.
    """
    dtype = np.dtype(dtype)
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if array.null_count:
        raise ValueError(
            "array has %d nulls; the values buffer is undefined at those "
            "positions -- fill_null() before converting" % array.null_count
        )
    if len(array) == 0:
        return np.empty(0, dtype=dtype)
    return np.frombuffer(array.buffers()[1], dtype=dtype, count=len(array),
                         offset=array.offset * dtype.itemsize)
