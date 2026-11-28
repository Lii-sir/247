"""Windows x64 Rust backend for OpenCV 5 nearest-neighbor scoring."""
from ctypes import CDLL, c_bool, c_int32, c_size_t, c_uint32, c_void_p
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
        if not library.affine_has_fma():
            return None
        batch = getattr(library, "affine_rois_mismatches", None)
        if batch is not None:
            batch.argtypes = (c_void_p, c_int32, c_int32, c_void_p, c_int32,
                              c_void_p, c_void_p, c_size_t, c_void_p)
            batch.restype = None
        build_blocks = getattr(library, "build_affine_block_counts", None)
        score_blocks = getattr(library, "score_affine_block_counts", None)
        blocks = None
        if build_blocks is not None and score_blocks is not None:
            build_blocks.argtypes = (c_void_p, c_size_t, c_size_t, c_void_p, c_size_t, c_void_p,
                                     c_size_t, c_void_p, c_void_p, c_size_t)
            build_blocks.restype = c_bool
            score_blocks.argtypes = (c_void_p, c_int32, c_int32, c_void_p, c_int32, c_void_p,
                                     c_void_p, c_void_p, c_void_p, c_size_t, c_void_p)
            score_blocks.restype = None
            blocks = build_blocks, score_blocks
        return kernel, batch, blocks
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
    library = _load_library()
    if library is None:
        return None
    kernel, batch, blocks = library

    source_height, source_width = source.shape
    source_buffer = np.empty(source.size + 3, dtype=np.uint8)
    source_buffer[:-3].reshape(source.shape)[:] = source
    source_buffer[-3:] = 0  # AVX2 gathers four bytes at each valid source pixel.
    target_buffer = np.ascontiguousarray(target)
    roi_buffer = np.ascontiguousarray(rois, dtype=np.int32)
    source_ptr, target_ptr = source_buffer.ctypes.data, target_buffer.ctypes.data
    regions = [(box.ctypes.data, (x1 - x0) * (y1 - y0))
               for box, (x0, y0, x1, y1) in zip(roi_buffer, rois)]
    areas = [area for _, area in regions]
    out_buffer = np.empty(len(rois), dtype=np.uint32)
    roi_ptr, out_ptr = roi_buffer.ctypes.data, out_buffer.ctypes.data
    buffers = source_buffer, target_buffer, roi_buffer, out_buffer
    score_blocks = None
    if blocks is not None and os.environ.get("YINXKJ_DISABLE_RUST_AFFINE_BLOCKS") != "1":
        build_blocks, score_blocks_candidate = blocks
        tiles = np.empty(((source_height + 3) // 4 + 1, (source_width + 7) // 8 + 1), np.uint64)
        # Match the DLL's 64-pixel blocks and keep OpenCV's scalar tail out of the table.
        vector_end = target_width // 16 * 16
        block_count = sum((y1 - y0) * ((max(0, min(x1, vector_end) - x0) + 63) // 64)
                          for x0, y0, x1, y1 in rois)
        target_ones = np.empty(block_count, np.uint8)
        tiles_ptr, ones_ptr = tiles.ctypes.data, target_ones.ctypes.data
        if build_blocks(source_ptr, source_width, source_height, target_ptr, target_width,
                        roi_ptr, len(areas), tiles_ptr, ones_ptr, block_count):
            score_blocks = score_blocks_candidate
            buffers += tiles, target_ones

    def score(matrix, _buffers=buffers):
        inverse = cv2.invertAffineTransform(matrix).astype(np.float32)
        if not np.all(np.abs(inverse) <= 8192):
            return None
        inverse_ptr = inverse.ctypes.data
        if score_blocks is not None:
            score_blocks(source_ptr, source_width, source_height, target_ptr, target_width, tiles_ptr,
                         ones_ptr, inverse_ptr, roi_ptr, len(areas), out_ptr)
            return tuple(int(count) / area for count, area in zip(out_buffer, areas))
        if batch is not None:
            batch(source_ptr, source_width, source_height, target_ptr, target_width,
                  inverse_ptr, roi_ptr, len(areas), out_ptr)
            return tuple(int(count) / area for count, area in zip(out_buffer, areas))
        return tuple(kernel(source_ptr, source_width, source_height, target_ptr, target_width,
                            inverse_ptr, box_ptr) / area for box_ptr, area in regions)

    return score
