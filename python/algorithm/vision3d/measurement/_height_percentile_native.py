"""Optional Windows backend for independent percentiles of packed uint16 depth."""
from ctypes import CDLL, c_bool, c_double, c_size_t, c_void_p
from functools import lru_cache
import os
from pathlib import Path
import platform

import numpy as np


@lru_cache(maxsize=1)
def _kernel():
    try:
        dll = CDLL(str(Path(__file__).with_name("native") / "height_percentile.dll"))
        function = dll.select_depth_samples
        function.argtypes = (c_void_p, c_size_t, c_void_p, c_size_t,
                             c_double, c_void_p, c_size_t, c_void_p)
        function.restype = c_bool
        return function
    except (AttributeError, OSError):
        return None


def select_depth_samples(raw, counts, percentile):
    if (os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}
            or os.environ.get("YINXKJ_DISABLE_RUST_HEIGHT_PERCENTILE") == "1"
            or raw.ndim != 1 or counts.ndim != 1 or not counts.size
            or raw.size < 64 or raw.dtype != np.uint16 or counts.dtype != np.int64
            or not raw.flags.c_contiguous or not raw.flags.aligned
            or not counts.flags.c_contiguous or not counts.flags.aligned):
        return None
    function = _kernel()
    if function is None:
        return None
    largest = min(int(counts.max()), raw.size)
    if largest <= 0:
        return None
    scratch = np.empty(largest, dtype=np.uint16)
    output = np.empty(len(counts), dtype=np.int64)
    if not function(raw.ctypes.data, raw.size, counts.ctypes.data, counts.size,
                    percentile, scratch.ctypes.data, scratch.size, output.ctypes.data):
        return None
    return output
