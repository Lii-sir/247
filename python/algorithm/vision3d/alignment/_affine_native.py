"""Windows x64 Rust backend for OpenCV 5 nearest-neighbor scoring."""
from ctypes import CDLL, c_bool, c_int32, c_uint32, c_void_p
from functools import lru_cache
import os
from pathlib import Path
import platform

import cv2
import numpy as np


@lru_cache(maxsize=1)
def _load_library():
    try:
        library = CDLL(str(Path(__file__).with_name("native") / "affine_score.dll"))
        library.affine_has_fma.argtypes = []
        library.affine_has_fma.restype = c_bool
        kernel = library.affine_roi_mismatches
        kernel.argtypes = (c_void_p, c_int32, c_int32, c_void_p, c_int32, c_void_p, c_void_p)
        kernel.restype = c_uint32
        return kernel if library.affine_has_fma() else None
    except (AttributeError, OSError):
        return None


def make_scorer(source, target, rois):
    if (os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}
            or os.environ.get("YINXKJ_DISABLE_RUST_AFFINE") == "1"
            or cv2.__version__ != "5.0.0" or not cv2.useOptimized()
            or "*AVX2" not in cv2.getCPUFeaturesLine().split()
            or source.ndim != 2 or target.ndim != 2 or source.size == 0
            or target.size < 65536 or max(*source.shape, *target.shape) >= 32767
            or source.dtype != np.uint8 or target.dtype != np.uint8):
        return None
    target_height, target_width = target.shape
    if not all(0 <= x0 < x1 <= target_width and 0 <= y0 < y1 <= target_height
               for x0, y0, x1, y1 in rois):
        return None
    kernel = _load_library()
    if kernel is None:
        return None

    source_height, source_width = source.shape
    source_buffer = np.empty(source.size + 3, dtype=np.uint8)
    source_buffer[:-3].reshape(source.shape)[:] = source
    source_buffer[-3:] = 0  # AVX2 gathers four bytes at each valid source pixel.
    target_buffer = np.ascontiguousarray(target)
    roi_buffer = np.ascontiguousarray(rois, dtype=np.int32)
    source_ptr, target_ptr = source_buffer.ctypes.data, target_buffer.ctypes.data
    regions = [(box.ctypes.data, (x1 - x0) * (y1 - y0))
               for box, (x0, y0, x1, y1) in zip(roi_buffer, rois)]
    buffers = source_buffer, target_buffer, roi_buffer
    def score(matrix, _buffers=buffers):
        inverse = cv2.invertAffineTransform(matrix).astype(np.float32)
        if not np.all(np.abs(inverse) <= 8192):
            return None
        inverse_ptr = inverse.ctypes.data
        return tuple(kernel(source_ptr, source_width, source_height, target_ptr, target_width,
                            inverse_ptr, box_ptr) / area for box_ptr, area in regions)

    return score
