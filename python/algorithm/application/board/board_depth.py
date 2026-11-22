"""Read a whole-board depth scan and identify its pieces."""
from dataclasses import dataclass
import time

import numpy as np

from algorithm.runtime.timing import timed_call
from algorithm.vision3d.depth import segment as segmentation
from transport.depth_input import depth_coordinate_space, input_depth_correction, validate_depth_input


@dataclass(frozen=True)
class BoardDepth:
    image: np.ndarray
    pieces: list[dict]
    correction: dict
    read_ms: float
    segmentation_ms: float


def read_board_depth(request, read_depth, config, expected_count, *, segment=None):
    if segment is None:
        segment = segmentation.segment_depth_image
    started = time.perf_counter()
    depth = timed_call("depth_read_ms", read_depth, request)
    read_ms = (time.perf_counter() - started) * 1000
    correction = input_depth_correction(request)
    started = time.perf_counter()
    pieces = timed_call("depth_segmentation_ms", segment, depth, config)
    if len(pieces) != expected_count:
        raise segmentation.PieceCountMismatchError(expected_count, len(pieces))
    return BoardDepth(depth, pieces, correction, read_ms, (time.perf_counter() - started) * 1000)


def piece_depth_geometry(request, piece_id):
    _, _, _, height = validate_depth_input(request)
    correction = input_depth_correction(request)
    piece = dict(piece_id=piece_id, y1=0, y2=height,
                 depth_coordinate_space=depth_coordinate_space(correction, piece_local=True))
    return piece, correction
