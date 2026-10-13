from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class DepthImageDecodeError(ValueError):
    pass


class PieceCountMismatchError(ValueError):
    def __init__(self, expected_count: int, detected_count: int):
        self.expected_count = expected_count
        self.detected_count = detected_count
        super().__init__(
            f"expected {expected_count} pieces, detected {detected_count}"
        )


@dataclass(frozen=True)
class SegmentationConfig:
    expected_piece_count: int = 10
    min_valid_depth: int = 0
    row_occupancy_threshold: float = 0.05
    column_occupancy_threshold: float = 0.01
    min_piece_height_ratio: float = 0.02
    max_internal_gap_height: int = 5
    padding_x: int = 0
    padding_y: int = 0

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "SegmentationConfig":
        allowed = {
            "expected_piece_count",
            "min_valid_depth",
            "row_occupancy_threshold",
            "column_occupancy_threshold",
            "min_piece_height_ratio",
            "max_internal_gap_height",
            "padding_x",
            "padding_y",
        }
        config = cls(**{key: values[key] for key in allowed if key in values})
        config.validate()
        return config

    def validate(self) -> None:
        if self.expected_piece_count <= 0:
            raise ValueError("expected_piece_count must be greater than 0")
        if not 0 < self.row_occupancy_threshold <= 1:
            raise ValueError("row_occupancy_threshold must be in (0, 1]")
        if not 0 < self.column_occupancy_threshold <= 1:
            raise ValueError("column_occupancy_threshold must be in (0, 1]")
        if not 0 < self.min_piece_height_ratio <= 1:
            raise ValueError("min_piece_height_ratio must be in (0, 1]")
        if self.max_internal_gap_height < 0:
            raise ValueError("max_internal_gap_height must be non-negative")
        if self.padding_x < 0 or self.padding_y < 0:
            raise ValueError("piece padding must be non-negative")


def decode_depth_png(data: bytes, data_format: str) -> np.ndarray:
    normalized_format = data_format.strip().lower()
    is_png = data.startswith(PNG_SIGNATURE)

    if not is_png:
        if normalized_format in {"uint16", "raw_uint16"}:
            raise DepthImageDecodeError(
                "raw uint16 data requires image_width and image_height; "
                "send encoded PNG bytes for the first-stage segmenter"
            )
        raise DepthImageDecodeError("3D input is not a PNG depth image")

    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            depth = np.asarray(image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DepthImageDecodeError(f"failed to decode PNG depth image: {exc}") from exc

    if depth.ndim != 2:
        raise DepthImageDecodeError(
            f"depth PNG must have one channel, actual shape={depth.shape}"
        )
    if depth.dtype.kind not in {"u", "i"}:
        raise DepthImageDecodeError(
            f"depth PNG must contain integer pixels, actual dtype={depth.dtype}"
        )

    return depth


def segment_depth_image(
    depth: np.ndarray,
    config: SegmentationConfig,
) -> List[Dict[str, Any]]:
    config.validate()
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("depth image must be a non-empty two-dimensional array")

    valid_mask = depth > config.min_valid_depth
    min_piece_height = max(
        1,
        int(round(depth.shape[0] * config.min_piece_height_ratio)),
    )
    row_occupancy = np.count_nonzero(valid_mask, axis=1) / depth.shape[1]
    row_ranges = _find_row_ranges(row_occupancy, config, min_piece_height)

    if len(row_ranges) != config.expected_piece_count:
        raise PieceCountMismatchError(config.expected_piece_count, len(row_ranges))

    pieces: List[Dict[str, Any]] = []

    for piece_index, (raw_y1, raw_y2) in enumerate(row_ranges, start=1):
        piece_mask = valid_mask[raw_y1:raw_y2]
        column_occupancy = np.count_nonzero(piece_mask, axis=0) / piece_mask.shape[0]
        active_columns = np.flatnonzero(
            column_occupancy >= config.column_occupancy_threshold
        )
        if active_columns.size == 0:
            raise ValueError(f"piece {piece_index} has no valid depth columns")

        raw_x1 = int(active_columns[0])
        raw_x2 = int(active_columns[-1]) + 1
        x1 = max(0, raw_x1 - config.padding_x)
        x2 = min(depth.shape[1], raw_x2 + config.padding_x)
        y1 = max(0, raw_y1 - config.padding_y)
        y2 = min(depth.shape[0], raw_y2 + config.padding_y)

        pieces.append({
            "piece_id": piece_index,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "defects": [],
        })

    return pieces


def _find_row_ranges(
    row_occupancy: np.ndarray,
    config: SegmentationConfig,
    min_piece_height: int,
) -> List[Tuple[int, int]]:
    """Find piece bands while tolerating view-dependent background occupancy.

    A changed 3D view can leave a narrow side rail visible between pieces.  In
    that case the configured threshold (historically 0.05) joins all pieces
    into one band.  Keep the configured result whenever it already has the
    expected count, then try the observed occupancy levels from low to high.
    This preserves the old view's exact boundaries and only raises the
    threshold when the view makes the initial segmentation ambiguous.
    """
    def ranges_for(threshold: float) -> List[Tuple[int, int]]:
        active_rows = _fill_short_false_runs(
            row_occupancy >= threshold,
            config.max_internal_gap_height,
        )
        return [
            (start, end)
            for start, end in _true_runs(active_rows)
            if end - start >= min_piece_height
        ]

    row_ranges = ranges_for(config.row_occupancy_threshold)
    if len(row_ranges) == config.expected_piece_count:
        return row_ranges

    # Occupancy is quantized by image width, so only observed values can
    # change the result.  Searching them avoids a hard-coded threshold tied
    # to one camera resolution or viewing angle.
    candidates = np.unique(row_occupancy[row_occupancy > config.row_occupancy_threshold])
    fallback_ranges: List[Tuple[int, int]] | None = None
    fallback_score = -1
    # Reject a threshold that creates two almost-touching bands from a
    # single piece's internal dark detail.  Among the remaining candidates,
    # prefer the widest bands (the actual parts occupy most of their pitch).
    minimum_gap = max(1, int(round(min_piece_height * 0.5)))
    for threshold in candidates:
        candidate_ranges = ranges_for(float(threshold))
        if len(candidate_ranges) == config.expected_piece_count:
            gaps = [start - previous_end
                    for (_, previous_end), (start, _) in
                    zip(candidate_ranges, candidate_ranges[1:])]
            if gaps and min(gaps) < minimum_gap:
                continue
            score = sum(end - start for start, end in candidate_ranges)
            if score > fallback_score:
                fallback_ranges = candidate_ranges
                fallback_score = score

    if fallback_ranges is not None:
        return fallback_ranges

    return row_ranges


def _true_runs(values: np.ndarray) -> List[Tuple[int, int]]:
    padded = np.concatenate((np.asarray([False]), values, np.asarray([False])))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _fill_short_false_runs(values: np.ndarray, max_run_length: int) -> np.ndarray:
    output = values.copy()
    if max_run_length == 0:
        return output

    false_runs = _true_runs(~output)
    for start, end in false_runs:
        touches_image_edge = start == 0 or end == len(output)
        if not touches_image_edge and end - start <= max_run_length:
            output[start:end] = True
    return output
