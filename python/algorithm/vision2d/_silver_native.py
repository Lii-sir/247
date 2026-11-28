"""Optional native scan for silver outside an axis-aligned boundary."""
from ctypes import CDLL, POINTER, c_size_t, c_void_p
from functools import lru_cache
from math import ceil, floor
import os
from pathlib import Path
import platform

import numpy as np


@lru_cache(maxsize=1)
def _load_kernel():
    try:
        library = CDLL(str(Path(__file__).with_name("native") / "silver_scan.dll"))
        kernel = library.scan_silver_rectangle
        kernel.argtypes = (c_void_p, c_size_t, c_size_t, c_size_t, c_size_t,
                           c_size_t, c_size_t, c_void_p, POINTER(c_size_t))
        kernel.restype = c_size_t
        return kernel
    except (AttributeError, OSError):
        return None


def rectangle_pixels(mask: np.ndarray, bounds):
    """Return the silver count and row/column indices of outside pixels."""
    if (os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}
            or os.environ.get("YINXKJ_DISABLE_RUST_SILVER") == "1"
            or not mask.flags.c_contiguous or mask.size < 65_536):
        return None
    kernel = _load_kernel()
    if kernel is None:
        return None

    height, width = mask.shape
    left, top, right, bottom = bounds
    x0, x1 = (min(width, max(0, value)) for value in (ceil(left), floor(right) + 1))
    y0, y1 = (min(height, max(0, value)) for value in (ceil(top), floor(bottom) + 1))
    indices = np.empty(mask.size - (x1 - x0) * (y1 - y0), dtype=np.intp)
    silver = c_size_t()
    outside = kernel(mask.ctypes.data, width, height, x0, x1, y0, y1, indices.ctypes.data, silver)
    rows, columns = np.divmod(indices[:outside], width)
    return silver.value, rows, columns
