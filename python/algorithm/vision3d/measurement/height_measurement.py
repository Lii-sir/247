"""Height measurements for aligned bond, chip and wire masks."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Sequence

import numpy as np
import cv2
from algorithm.vision3d.measurement._height_percentile_native import select_depth_samples
from algorithm.vision3d.measurement.height_percentile import pick_height_samples


@dataclass(frozen=True)
class HeightConfig:
    platform_roi_offsets: tuple[tuple[float, float, float, float], ...] = ()
    invalid_values: tuple[int, ...] = (0, 65535)
    gray_offset: float = 32768.0
    vertical_resolution_mm: float = 0.00105
    height_percentile: float = 95.0
    plane_percentile: float = 35.0
    plane_min_valid_points: int = 100
    max_wire_platform_height_mm: float = 0.8
    max_solder_platform_height_mm: float = 0.8
    max_solder_chip_height_mm: float = 0.8
    chip_min_valid_points: int = 100
    thin_solder_end_ratio: float = 0.02
    thin_solder_search_dilation_px: int = 6
    platform_local_ring_px: int = 24
    semantic_exclusion_dilation_px: int = 3
    wire_measurement_erode_px: int = 0
    thin_measurement_erode_px: int = 0
    bond_measurement_erode_px: int = 0
    thin_solder_measurement_erode_px: int = 0
    reference_mode: str = "legacy"

    @classmethod
    def from_dict(cls, values: Dict[str, Any] | None) -> "HeightConfig":
        values = values or {}
        raw_aois = values.get("platform_roi_offsets", cls.platform_roi_offsets)
        if not isinstance(raw_aois, (list, tuple)):
            raise ValueError("height_3d.platform_roi_offsets must be a list")
        aois = tuple(_box(item, "height_3d.platform_roi_offsets") for item in raw_aois)
        raw_invalid = values.get("invalid_values", cls.invalid_values)
        config = cls(
            reference_mode=values.get("reference_mode", cls.reference_mode),
            platform_roi_offsets=aois,
            invalid_values=tuple(int(item) for item in raw_invalid),
            gray_offset=float(values.get("gray_offset", cls.gray_offset)),
            vertical_resolution_mm=float(
                values.get("vertical_resolution_mm", cls.vertical_resolution_mm)
            ),
            height_percentile=float(values.get("height_percentile", cls.height_percentile)),
            plane_percentile=float(values.get("plane_percentile", cls.plane_percentile)),
            plane_min_valid_points=int(
                values.get("plane_min_valid_points", cls.plane_min_valid_points)
            ),
            max_wire_platform_height_mm=float(
                values.get("max_wire_platform_height_mm", cls.max_wire_platform_height_mm)
            ),
            max_solder_platform_height_mm=float(
                values.get("max_solder_platform_height_mm", cls.max_solder_platform_height_mm)
            ),
            max_solder_chip_height_mm=float(
                values.get("max_solder_chip_height_mm", cls.max_solder_chip_height_mm)
            ),
            chip_min_valid_points=int(
                values.get("chip_min_valid_points", cls.chip_min_valid_points)
            ),
            thin_solder_end_ratio=float(
                values.get("thin_solder_end_ratio", cls.thin_solder_end_ratio)
            ),
            thin_solder_search_dilation_px=int(
                values.get("thin_solder_search_dilation_px", cls.thin_solder_search_dilation_px)
            ),
            platform_local_ring_px=int(
                values.get("platform_local_ring_px", cls.platform_local_ring_px)
            ),
            semantic_exclusion_dilation_px=int(
                values.get("semantic_exclusion_dilation_px", cls.semantic_exclusion_dilation_px)
            ),
            wire_measurement_erode_px=values.get(
                "wire_measurement_erode_px", cls.wire_measurement_erode_px),
            thin_measurement_erode_px=values.get(
                "thin_measurement_erode_px", cls.thin_measurement_erode_px),
            bond_measurement_erode_px=values.get(
                "bond_measurement_erode_px", cls.bond_measurement_erode_px),
            thin_solder_measurement_erode_px=values.get(
                "thin_solder_measurement_erode_px", cls.thin_solder_measurement_erode_px),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.reference_mode not in ("legacy", "roi_overlap"):
            raise ValueError("height_measurement_3d.reference_mode must be legacy or roi_overlap")
        for name in ("wire_measurement_erode_px", "thin_measurement_erode_px",
                     "bond_measurement_erode_px", "thin_solder_measurement_erode_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"height_measurement_3d.{name} must be a non-negative integer")
        if not 0 < self.height_percentile <= 100:
            raise ValueError("height_3d.height_percentile must be in (0, 100]")
        if not 0 < self.plane_percentile <= 100:
            raise ValueError("height_3d.plane_percentile must be in (0, 100]")
        if self.vertical_resolution_mm <= 0 or self.plane_min_valid_points <= 0:
            raise ValueError("invalid height_3d resolution or plane point count")
        if self.chip_min_valid_points <= 0:
            raise ValueError("height_3d.chip_min_valid_points must be positive")
        if not 0 < self.thin_solder_end_ratio < 0.5:
            raise ValueError("height_3d.thin_solder_end_ratio must be in (0, 0.5)")
        if min(self.thin_solder_search_dilation_px, self.platform_local_ring_px,
               self.semantic_exclusion_dilation_px) < 0:
            raise ValueError("height_3d dilation/ring parameters must be non-negative")
        if min(self.max_wire_platform_height_mm, self.max_solder_platform_height_mm,
               self.max_solder_chip_height_mm) <= 0:
            raise ValueError("height_3d maximum height thresholds must be positive")


def fit_platform_plane(
    depth: np.ndarray,
    config: HeightConfig,
    roi_mask: np.ndarray,
    exclude_mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Fit only inside perspective-mapped platform ROIs, excluding YOLO masks."""
    if np.asarray(roi_mask).shape != depth.shape:
        raise ValueError("platform roi_mask must have the same shape as depth")
    valid = (np.asarray(roi_mask) > 0) & _valid_depth(depth, config)
    excluded_count = 0
    if exclude_mask is not None:
        if np.asarray(exclude_mask).shape != depth.shape:
            raise ValueError("exclude_mask must have the same shape as depth")
        excluded_count = int(np.count_nonzero(valid & (exclude_mask > 0)))
        valid &= exclude_mask == 0
    ys, xs = np.nonzero(valid)
    plane = _fit_plane_points(depth, xs, ys, config, "wire_platform")
    plane.update(roi_area_px=int(np.count_nonzero(roi_mask)),
                 excluded_pixel_count=excluded_count, candidate_points=int(len(xs)))
    return plane


def map_platform_rois(
    circle: Sequence[float],
    offsets: Sequence[Sequence[float]],
    registration: Dict[str, Any],
    depth_shape: tuple[int, int],
) -> tuple[np.ndarray, list[list[int]], list[list[list[float]]]]:
    """Map full-image circle-relative rectangles, including those outside YOLO crop."""
    cx, cy, radius = (float(value) for value in circle)
    if not np.isfinite([cx, cy, radius]).all() or radius <= 0:
        raise ValueError("platform ROI requires a valid detected circle")
    raw = registration.get("homography_2d_to_3d")
    if raw is None:
        raw = [[registration["scale"], 0, registration["tx"]],
               [0, registration["scale"], registration["ty"]], [0, 0, 1]]
    matrix = np.asarray(raw, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or np.linalg.matrix_rank(matrix) < 3):
        raise ValueError("platform ROI requires a finite 3x3 homography")
    mask = np.zeros(depth_shape, dtype=np.uint8)
    boxes, polygons = [], []
    for roi in offsets:
        x1, y1, x2, y2 = _box(roi, "platform_roi_offsets")
        box = [round(cx + x1 * radius), round(cy + y1 * radius),
               round(cx + x2 * radius), round(cy + y2 * radius)]
        points = np.array([[[box[0], box[1]], [box[2], box[1]],
                            [box[2], box[3]], [box[0], box[3]]]], dtype=np.float64)
        denominators = np.column_stack((points[0], np.ones(4))) @ matrix[2]
        if (np.any(np.abs(denominators) < 1e-10)
                or np.min(denominators) * np.max(denominators) < 0):
            raise ValueError("platform ROI crosses the projective horizon")
        polygon = cv2.perspectiveTransform(points, matrix)[0]
        if not np.isfinite(polygon).all() or np.max(np.abs(polygon)) > 1e7:
            raise ValueError("platform ROI mapping is invalid")
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
        boxes.append(box)
        polygons.append(polygon.tolist())
    return mask, boxes, polygons


def fit_chip_plane(
    depth: np.ndarray,
    chip_mask: np.ndarray,
    config: HeightConfig,
    exclude_mask: np.ndarray | None = None,
) -> Dict[str, Any] | None:
    """Fit the chip top plane from the aligned chip segmentation mask."""
    source = np.broadcast_to(chip_mask, depth.shape)
    binary = source if source.dtype == np.uint8 else (source > 0).astype(np.uint8)
    exclusion = None
    if exclude_mask is not None:
        exclusion = np.asarray(exclude_mask)
        if exclusion.shape != depth.shape:
            raise ValueError("exclude_mask must have the same shape as depth")
    x, y, width, height = cv2.boundingRect(binary)
    if width * height < config.chip_min_valid_points:
        return None
    region = np.s_[y:y + height, x:x + width]
    mask = (binary[region] > 0) & _valid_depth(depth[region], config)
    if exclusion is not None:
        mask &= exclusion[region] == 0
    ys, xs = np.nonzero(mask)
    if len(xs) < config.chip_min_valid_points:
        return None
    return _fit_plane_points(depth, xs + x, ys + y, config, "chip", use_low_percentile=False)


def fit_local_platform_plane(
    depth: np.ndarray,
    target_mask: np.ndarray,
    exclude_mask: np.ndarray,
    config: HeightConfig,
) -> Dict[str, Any] | None:
    """Fit the platform in a ring around one solder joint.

    All aligned semantic masks are removed from the ring, so another joint,
    wire or chip on the same platform cannot become the reference surface.
    """
    depth = np.asarray(depth)
    source = np.asarray(target_mask)
    target = source if source.dtype == np.uint8 else (source > 0).astype(np.uint8)
    if target.shape != depth.shape:
        raise ValueError("target_mask must have the same shape as depth")
    exclusion = np.asarray(exclude_mask)
    if exclusion.shape != depth.shape:
        raise ValueError("exclude_mask must have the same shape as depth")
    x, y, width, height = cv2.boundingRect(target)
    if width == 0 or config.platform_local_ring_px <= 0:
        return None
    radius = int(config.platform_local_ring_px)
    inner_radius = max(1, int(config.semantic_exclusion_dilation_px))
    # Retain the complete morphology halo, including image-edge clipping.
    # Fitting still uses original piece coordinates and original row order.
    margin = max(radius, inner_radius)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(depth.shape[1], x + width + margin), min(depth.shape[0], y + height + margin)
    local_target = target[y0:y1, x0:x1]
    outer = cv2.dilate(local_target, _ellipse_kernel(radius)) > 0
    inner = cv2.dilate(local_target, _ellipse_kernel(inner_radius)) > 0
    ring = (outer & ~inner & (exclusion[y0:y1, x0:x1] == 0)
            & _valid_depth(depth[y0:y1, x0:x1], config))
    ys, xs = np.nonzero(ring)
    if len(xs) < config.plane_min_valid_points:
        return None
    try:
        return _fit_plane_points(depth, xs + x0, ys + y0, config, "local_wire_platform")
    except ValueError:
        return None


def split_thin_mask(
    mask: np.ndarray,
    end_ratio: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Split the first/last principal-axis percentage into solder masks."""
    binary = np.asarray(mask) > 0
    ys, xs = np.nonzero(binary)
    empty = np.zeros(binary.shape, dtype=np.uint8)
    if len(xs) < 3:
        return empty, []
    points = np.column_stack((xs, ys)).astype(np.float64)
    centered = points - points.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vectors[0]
    low_edge, high_edge = float(projection.min()), float(projection.max())
    end_length = (high_edge - low_edge) * end_ratio
    low, high = low_edge + end_length, high_edge - end_length
    start = empty.copy()
    end = empty.copy()
    middle = empty.copy()
    start[ys[projection <= low], xs[projection <= low]] = 255
    end[ys[projection >= high], xs[projection >= high]] = 255
    middle_pixels = (projection > low) & (projection < high)
    middle[ys[middle_pixels], xs[middle_pixels]] = 255
    solder = [item for item in (start, end) if np.any(item)]
    return middle, solder


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if radius_px <= 0 or not np.any(binary):
        return binary
    return cv2.dilate(binary, _ellipse_kernel(radius_px))


def erode_wire_mask_in_place(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Trim a newly aligned, owned uint8 mask without a full-frame copy.

    Radius is in 3D pixels. Erosion has zero background at every bbox/image
    edge. The aligned source is not shared with semantic-plane masks, so only
    the independently warped wire-height region is changed by this function.
    """
    if radius_px == 0:
        return mask
    if radius_px < 0:
        raise ValueError("wire erosion radius must be non-negative")
    x, y, width, height = cv2.boundingRect(mask)
    if width == 0:
        return mask
    local = mask[y:y+height, x:x+width]
    if min(width, height) <= 2 * radius_px:
        local[:] = 0
    else:
        local[:] = cv2.erode(local, _ellipse_kernel(radius_px),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask


def erode_measurement_mask_crop(mask: np.ndarray, radius_px: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Return an eroded uint8 bbox and its piece offset, leaving shared masks intact."""
    if radius_px == 0:
        return mask, (0, 0)
    if radius_px < 0:
        raise ValueError("measurement erosion radius must be non-negative")
    x, y, width, height = cv2.boundingRect(mask)
    if width == 0 or min(width, height) <= 2 * radius_px:
        return np.zeros((1, 1), dtype=np.uint8), (x, y)
    # cv2.erode returns a fresh local array; no full-depth mask copy or paste.
    eroded = cv2.erode(mask[y:y+height, x:x+width], _ellipse_kernel(radius_px),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return eroded, (x, y)


def _mask_points(mask, mask_offset):
    source = np.asarray(mask)
    if source.dtype == np.uint8:
        # Warp output is uint8. Find its bounds in OpenCV, then enumerate only
        # the local region instead of allocating/scanning a full-depth boolean
        # mask per wire. Row-major point order is preserved, including ties.
        left, top, width, height = cv2.boundingRect(source)
        ys, xs = np.nonzero(source[top:top+height, left:left+width])
        ys, xs = ys + top, xs + left
    else:
        ys, xs = np.nonzero(source > 0)
    ox, oy = mask_offset
    if ox or oy:
        xs, ys = xs + ox, ys + oy
    return ys, xs


def _invalid_height(reference):
    return dict(reference=reference, is_valid=False, is_ok=False,
                height_mm=None, height_diff_mm=None, point=None)


def measure_mask_height(
    depth: np.ndarray,
    mask: np.ndarray,
    plane: Dict[str, Any] | None,
    config: HeightConfig,
    reference: str,
    max_height_mm: float | None = None,
    *,
    mask_offset: tuple[int, int] = (0, 0),
    reference_height_mm: float | None = None,
) -> Dict[str, Any]:
    """Measure a percentile relative to a plane or a fixed reference height."""
    ys, xs = _mask_points(mask, mask_offset)
    valid = _valid_depth(depth[ys, xs], config)
    ys, xs = ys[valid], xs[valid]
    if len(xs) == 0 or (plane is None and reference_height_mm is None):
        return _invalid_height(reference)
    values = depth[ys, xs].astype(np.float64)
    index = int(np.argmin(np.abs(values - np.percentile(values, config.height_percentile))))
    x, y, value = int(xs[index]), int(ys[index]), float(values[index])
    height_mm = (value - config.gray_offset) * config.vertical_resolution_mm
    if reference_height_mm is None:
        plane_value = _evaluate_plane(plane, x, y)
        reference_height_mm = (plane_value - config.gray_offset) * config.vertical_resolution_mm
    difference_mm = height_mm - reference_height_mm
    return {
        "reference": reference,
        "is_valid": True,
        "is_ok": None if max_height_mm is None else bool(difference_mm < max_height_mm),
        "height_mm": float(height_mm),
        "height_diff_mm": float(difference_mm),
        "point": {"x": x, "y": y},
    }


def measure_mask_heights(depth, regions, plane, config, reference, max_height_mm, *, reference_height_mm=None):
    """Gather and reduce one sparse set of samples with a separate P95 for each mask."""
    results = [_invalid_height(reference) for _ in regions]
    if not regions or (plane is None and reference_height_mm is None):
        return results

    points = [_mask_points(mask, offset) for mask, offset in regions]
    rows = np.repeat(np.arange(len(regions)), [len(xs) for _, xs in points])
    ys = np.concatenate([ys for ys, _ in points])
    xs = np.concatenate([xs for _, xs in points])
    raw = depth[ys, xs]
    valid = _valid_depth(raw, config)
    rows, ys, xs, raw = rows[valid], ys[valid], xs[valid], raw[valid]
    counts = np.bincount(rows, minlength=len(regions))
    active = np.flatnonzero(counts)
    if not active.size:
        return results

    selected = select_depth_samples(raw, counts, config.height_percentile)
    if selected is None:
        selected = np.asarray(pick_height_samples(raw, counts, config.height_percentile), dtype=np.intp)
    points = selected[active]
    x, y = xs[points], ys[points]
    heights = (raw[points].astype(np.float64) - config.gray_offset) * config.vertical_resolution_mm
    if reference_height_mm is None:
        a, b, c = plane["coefficients"]
        reference_height_mm = (a * x + b * y + c - config.gray_offset) * config.vertical_resolution_mm
    differences = heights - reference_height_mm
    for row, cx, cy, value, diff in zip(active, x, y, heights, differences):
        results[row] = dict(reference=reference, is_valid=True,
            is_ok=None if max_height_mm is None else bool(diff < max_height_mm),
            height_mm=float(value), height_diff_mm=float(diff), point={"x": int(cx), "y": int(cy)})
    return results


def height_points_to_board(piece: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize measurement points in the submitted depth image coordinates.

    Joint detection slices rows only (depth[start:end, :]), so X is already
    full-image X. Copy nested heights: measurements and instance masks can
    share the same height object and must not receive the offset twice.
    Reference planes and mask_3d_local remain in piece-local coordinates.
    """
    input_spaces = {"board_3d", "pitch_corrected_board_3d", "piece_3d", "pitch_corrected_piece_3d"}
    if piece.get("height_point_coordinate_space") in input_spaces:
        return piece
    space = piece.get("depth_coordinate_space", "board_3d")
    row_offset = int(piece["depth_start_row"])

    def convert(item):
        height = item.get("height_3d")
        if not isinstance(height, dict):
            return item
        converted = dict(height)
        point = height.get("point")
        if isinstance(point, dict) and height.get("point_coordinate_space") not in input_spaces:
            converted["point"] = {**point, "x": int(point["x"]), "y": int(point["y"]) + row_offset}
        converted["point_coordinate_space"] = space
        return {**item, "height_3d": converted}

    result = {**piece, "height_point_coordinate_space": space}
    for field in ("wire_measurements", "solder_measurements", "masks"):
        if isinstance(piece.get(field), list):
            result[field] = [convert(item) for item in piece[field]]
    return result


def _fit_plane_points(
    depth: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    config: HeightConfig,
    name: str,
    use_low_percentile: bool = True,
) -> Dict[str, Any]:
    if len(xs) < config.plane_min_valid_points:
        raise ValueError(f"{name} plane has too few valid pixels: {len(xs)}")
    stride = max(1, len(xs) // 30000)
    xs = xs[::stride].astype(np.float64)
    ys = ys[::stride].astype(np.float64)
    values = depth[ys.astype(int), xs.astype(int)].astype(np.float64)
    keep = (
        values <= np.percentile(values, config.plane_percentile)
        if use_low_percentile
        else np.ones(values.shape, dtype=bool)
    )
    for _ in range(5):
        if np.count_nonzero(keep) < config.plane_min_valid_points:
            break
        design = np.column_stack((xs[keep], ys[keep], np.ones(np.count_nonzero(keep))))
        coefficients = np.linalg.lstsq(design, values[keep], rcond=None)[0]
        residual = values - (
            coefficients[0] * xs + coefficients[1] * ys + coefficients[2]
        )
        median = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - median))
        keep = np.abs(residual - median) <= max(50.0, 4.4478 * mad)
    if np.count_nonzero(keep) < config.plane_min_valid_points:
        raise ValueError(f"{name} plane fit retained too few pixels")
    plane = {
        "name": name,
        "coefficients": [float(value) for value in coefficients],
        "valid_points": int(np.count_nonzero(keep)),
    }
    if name == "chip" and config.reference_mode == "roi_overlap":
        # The last design matrix contains the points that produced the returned plane.
        cx, cy = map(float, design[:, :2].mean(axis=0))
        plane["fit_center_3d_local"] = {"x": cx, "y": cy}
        plane["representative_height_mm"] = (_evaluate_plane(plane, cx, cy) - config.gray_offset) * config.vertical_resolution_mm
    return plane


def _evaluate_plane(plane: Dict[str, Any], x: float, y: float) -> float:
    a, b, c = plane["coefficients"]
    return float(a * x + b * y + c)


def _valid_depth(depth: np.ndarray, config: HeightConfig) -> np.ndarray:
    return ~np.isin(depth, np.asarray(config.invalid_values, dtype=depth.dtype))


def _box(values: Any, name: str) -> tuple[float, float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{name} entries must contain four numbers")
    result = tuple(float(value) for value in values)
    x1, y1, x2, y2 = result
    if not np.isfinite(result).all() or x1 >= x2 or y1 >= y2:
        raise ValueError(f"{name} entry is invalid: {result}")
    return result


@lru_cache(maxsize=32)
def _ellipse_kernel(radius: int) -> np.ndarray:
    size = max(1, radius * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
