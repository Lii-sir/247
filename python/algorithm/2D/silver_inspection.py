"""Silver overflow and continuity checks for the streaming 2D pipeline.

The original silver-inspection project keeps the geometry independent from
YOLO and Qt.  This module keeps the same property, but accepts the already
decoded YOLO masks used by Bridge.  All masks are crop-local boolean arrays;
the caller supplies the crop box and the detected circle when a boundary is
defined in original-image or circle-relative coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


EPSILON = 1e-6


def _key(name: str) -> str:
    return " ".join(str(name).strip().casefold().replace("_", " ").split())


@dataclass(frozen=True)
class Boundary:
    """An allowed silver boundary in the local 2D crop coordinate system."""

    points: tuple[tuple[float, float], ...]
    mode: str = "polygon"

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("silver boundary points must be finite 2D coordinates")
        object.__setattr__(self, "points", tuple(tuple(map(float, p)) for p in points))
        if self.mode not in {"polygon", "line"}:
            raise ValueError("silver boundary mode must be polygon or line")
        if len(points) < 3 or (self.mode == "line" and len(points) != 3):
            raise ValueError("silver boundary polygon needs at least 3 points; line needs 3 points")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("silver boundary points cannot repeat")
        if self.mode == "line":
            distance = abs(_cross(points[1] - points[0], points[2] - points[0]))
            distance /= max(np.linalg.norm(points[1] - points[0]), EPSILON)
            if distance <= EPSILON:
                raise ValueError("the inside point of a line boundary cannot be collinear")
            return
        if abs(float(np.sum(_cross(points, np.roll(points, -1, axis=0))))) <= EPSILON:
            raise ValueError("silver boundary area is zero")
        for i in range(len(points)):
            a, b, c = points[i - 1], points[i], points[(i + 1) % len(points)]
            if abs(_cross(b - a, c - b)) <= EPSILON and np.dot(b - a, c - b) < 0:
                raise ValueError("silver boundary contains a folded edge")
            for j in range(i + 1, len(points)):
                if j == i + 1 or (i == 0 and j == len(points) - 1):
                    continue
                if _segments_intersect(
                    points[i], points[(i + 1) % len(points)],
                    points[j], points[(j + 1) % len(points)],
                ):
                    raise ValueError("silver boundary cannot self-intersect")

    def validate_image(self, shape: Sequence[int]) -> None:
        height, width = int(shape[0]), int(shape[1])
        points = np.asarray(self.points)
        if (
            np.any(points < 0)
            or np.any(points[:, 0] > width - 1)
            or np.any(points[:, 1] > height - 1)
        ):
            raise ValueError("silver boundary is outside the 2D crop")

    def signed_distance(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Return negative distances inside and positive distances outside."""
        query = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        vertices = np.asarray(self.points)
        if self.mode == "line":
            a, b, inside = vertices
            side = np.sign(_cross(b - a, inside - a))
            return -side * _cross(b - a, query - a) / np.linalg.norm(b - a)

        inside = np.zeros(len(query), dtype=bool)
        minimum = np.full(len(query), np.inf)
        x, y = query.T
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = b - a
            denominator = np.dot(edge, edge)
            projection = np.clip(((query - a) @ edge) / max(denominator, EPSILON), 0, 1)
            nearest = a + projection[:, None] * edge
            minimum = np.minimum(minimum, np.linalg.norm(query - nearest, axis=1))
            if abs(edge[1]) > EPSILON:
                crossing = ((a[1] > y) != (b[1] > y)) & (
                    x < a[0] + (y - a[1]) * edge[0] / edge[1]
                )
                inside ^= crossing
        return np.where(inside, -minimum, minimum)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
    return bool(
        abs(_cross(b - a, p - a)) <= EPSILON
        and np.all(p >= np.minimum(a, b) - EPSILON)
        and np.all(p <= np.maximum(a, b) + EPSILON)
    )


def _segments_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> bool:
    ab_c, ab_d = _cross(b - a, c - a), _cross(b - a, d - a)
    cd_a, cd_b = _cross(d - c, a - c), _cross(d - c, b - c)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return True
    return any(
        (
            _on_segment(a, b, c),
            _on_segment(a, b, d),
            _on_segment(c, d, a),
            _on_segment(c, d, b),
        )
    )


@dataclass(frozen=True)
class OverflowSettings:
    silver_class: str = "silver"
    tolerance_px: float = 0.0
    min_area_px: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.silver_class, str) or not self.silver_class.strip():
            raise ValueError("silver_class cannot be empty")
        object.__setattr__(self, "silver_class", self.silver_class.strip())
        if not np.isfinite(self.tolerance_px) or self.tolerance_px < 0:
            raise ValueError("silver overflow tolerance must be non-negative")
        if (
            isinstance(self.min_area_px, bool)
            or not isinstance(self.min_area_px, int)
            or self.min_area_px < 1
        ):
            raise ValueError("silver overflow min_area_px must be a positive integer")


def measure_chip_expansion_overflow(
    instances: Iterable[Mapping[str, Any]],
    image_shape: Sequence[int],
    settings: OverflowSettings,
    chip_class: str = "chip",
    outward_length_px: int = 20,
    min_chip_area_px: int = 100,
) -> OverflowMeasurement:
    """Measure silver outside expanded axis-aligned chip rectangles."""
    if isinstance(outward_length_px, bool) or int(outward_length_px) < 1:
        raise ValueError("chip expansion outward_length_px must be positive")
    shape = (int(image_shape[0]), int(image_shape[1]))
    chip, chip_count = _union_masks(instances, (chip_class,), shape)
    if chip_count == 0 or not chip.any():
        raise ValueError(f"no {chip_class} mask was detected")
    components_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        chip.astype(np.uint8), 8
    )
    expanded = np.zeros_like(chip)
    for label in range(1, components_count):
        x, y, width, height, area = map(int, stats[label])
        if area < int(min_chip_area_px):
            continue
        distance = int(outward_length_px)
        x1, y1 = max(0, x - distance), max(0, y - distance)
        x2 = min(shape[1], x + width + distance)
        y2 = min(shape[0], y + height + distance)
        expanded[y1:y2, x1:x2] = True
    if not expanded.any():
        raise ValueError(f"no {chip_class} component reached min_chip_area_px")
    silver, silver_count = _union_masks(instances, (settings.silver_class,), shape)
    if silver_count == 0 or not silver.any():
        raise ValueError(f"no {settings.silver_class} mask was detected")
    outside = silver & ~expanded
    candidates = outside.copy()
    count, candidate_labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8), 8
    )
    accepted = np.zeros(count, dtype=bool)
    regions: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if area >= settings.min_area_px:
            accepted[label] = True
            regions.append({
                "area_px": area,
                "box_xyxy": [x, y, x + width, y + height],
                "max_outside_distance_px": float(outward_length_px),
            })
    defects = accepted[candidate_labels]
    return OverflowMeasurement(
        int(silver.sum()), int(outside.sum()), int(candidates.sum()),
        int(defects.sum()), float(outward_length_px if outside.any() else 0.0),
        regions, silver.copy(), outside, defects,
    )


@dataclass
class OverflowMeasurement:
    silver_area_px: int
    outside_area_px: int
    candidate_area_px: int
    defect_area_px: int
    max_outside_distance_px: float
    regions: list[dict[str, Any]]
    silver_mask: np.ndarray
    outside_mask: np.ndarray
    defect_mask: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "silver_area_px": self.silver_area_px,
            "outside_area_px": self.outside_area_px,
            "candidate_area_px": self.candidate_area_px,
            "defect_area_px": self.defect_area_px,
            "defect_ratio": (
                self.defect_area_px / self.silver_area_px
                if self.silver_area_px
                else 0.0
            ),
            "max_outside_distance_px": self.max_outside_distance_px,
            "regions": self.regions,
        }


def measure_overflow(
    mask: np.ndarray, boundary: Boundary, settings: OverflowSettings
) -> OverflowMeasurement:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("silver mask must be a 2D boolean array")
    boundary.validate_image(mask.shape)
    ys, xs = np.nonzero(mask)
    distances = np.empty(len(xs), dtype=np.float64)
    for offset in range(0, len(xs), 100_000):
        stop = offset + 100_000
        distances[offset:stop] = boundary.signed_distance(
            np.column_stack((xs[offset:stop], ys[offset:stop]))
        )
    outside = np.zeros(mask.shape, dtype=bool)
    outside[ys, xs] = distances > EPSILON
    candidates = np.zeros(mask.shape, dtype=np.uint8)
    candidates[ys, xs] = distances > settings.tolerance_px + EPSILON
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    accepted = np.zeros(count, dtype=bool)
    maximum = np.zeros(count, dtype=np.float64)
    if len(ys):
        np.maximum.at(maximum, labels[ys, xs], np.maximum(distances, 0))
    regions: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if area >= settings.min_area_px:
            accepted[label] = True
            regions.append(
                {
                    "area_px": area,
                    "box_xyxy": [x, y, x + width, y + height],
                    "max_outside_distance_px": float(maximum[label]),
                }
            )
    defects = accepted[labels]
    return OverflowMeasurement(
        int(len(xs)),
        int(outside.sum()),
        int(candidates.sum()),
        int(defects.sum()),
        float(np.maximum(distances, 0).max()) if len(xs) else 0.0,
        regions,
        mask.copy(),
        outside,
        defects,
    )


@dataclass(frozen=True)
class ContinuitySettings:
    silver_class: str = "silver"
    chip_class: str = "chip"
    occlusion_classes: tuple[str, ...] = ("thin", "bond", "wire")
    outward_length_px: int = 20
    sector_count: int = 72
    min_sector_silver_px: int = 3
    min_sector_coverage: float = 0.01
    min_valid_sector_px: int = 1
    occlusion_dilation_px: int = 0
    min_chip_area_px: int = 100
    min_visible_sector_ratio: float = 0.1

    def __post_init__(self) -> None:
        for field_name in ("silver_class", "chip_class"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} cannot be empty")
            object.__setattr__(self, field_name, value.strip())
        classes = tuple(
            str(value).strip()
            for value in self.occlusion_classes
            if str(value).strip()
        )
        if len(classes) != len({_key(value) for value in classes}):
            raise ValueError("occlusion classes cannot repeat")
        object.__setattr__(self, "occlusion_classes", classes)
        if (
            isinstance(self.outward_length_px, bool)
            or not isinstance(self.outward_length_px, int)
            or self.outward_length_px < 1
        ):
            raise ValueError("outward_length_px must be a positive integer")
        if (
            isinstance(self.sector_count, bool)
            or not isinstance(self.sector_count, int)
            or not 4 <= self.sector_count <= 720
        ):
            raise ValueError("sector_count must be in 4..720")
        if (
            isinstance(self.min_sector_silver_px, bool)
            or not isinstance(self.min_sector_silver_px, int)
            or self.min_sector_silver_px < 1
        ):
            raise ValueError("min_sector_silver_px must be positive")
        if not np.isfinite(self.min_sector_coverage) or not 0 <= self.min_sector_coverage <= 1:
            raise ValueError("min_sector_coverage must be in [0, 1]")
        if (
            isinstance(self.min_valid_sector_px, bool)
            or not isinstance(self.min_valid_sector_px, int)
            or self.min_valid_sector_px < 1
        ):
            raise ValueError("min_valid_sector_px must be positive")
        if (
            isinstance(self.occlusion_dilation_px, bool)
            or not isinstance(self.occlusion_dilation_px, int)
            or self.occlusion_dilation_px < 0
        ):
            raise ValueError("occlusion_dilation_px must be non-negative")
        if (
            isinstance(self.min_chip_area_px, bool)
            or not isinstance(self.min_chip_area_px, int)
            or self.min_chip_area_px < 1
        ):
            raise ValueError("min_chip_area_px must be positive")
        if not np.isfinite(self.min_visible_sector_ratio) or not 0 <= self.min_visible_sector_ratio <= 1:
            raise ValueError("min_visible_sector_ratio must be in [0, 1]")


@dataclass
class ContinuityMeasurement:
    chip_area_px: int
    ring_area_px: int
    silver_area_px: int
    occluded_area_px: int
    valid_area_px: int
    covered_sector_count: int
    valid_sector_count: int
    ignored_sector_count: int
    missing_sector_count: int
    center_xy: tuple[float, float]
    sectors: list[dict[str, Any]]
    chips: list[dict[str, Any]]
    chip_mask: np.ndarray
    ring_mask: np.ndarray
    silver_mask: np.ndarray
    occlusion_mask: np.ndarray
    missing_mask: np.ndarray

    @property
    def is_disconnected(self) -> bool | None:
        if self.valid_sector_count == 0:
            return None
        return self.missing_sector_count > 0

    def summary(self) -> dict[str, Any]:
        return {
            "chip_area_px": self.chip_area_px,
            "ring_area_px": self.ring_area_px,
            "silver_area_px": self.silver_area_px,
            "occluded_area_px": self.occluded_area_px,
            "valid_area_px": self.valid_area_px,
            "covered_sector_count": self.covered_sector_count,
            "valid_sector_count": self.valid_sector_count,
            "ignored_sector_count": self.ignored_sector_count,
            "missing_sector_count": self.missing_sector_count,
            "coverage_ratio": (
                self.covered_sector_count / self.valid_sector_count
                if self.valid_sector_count
                else None
            ),
            "center_xy": self.center_xy,
            "sectors": self.sectors,
            "chips": self.chips,
        }


def _union_masks(
    instances: Iterable[Mapping[str, Any]], names: Iterable[str], shape: tuple[int, int]
) -> tuple[np.ndarray, int]:
    wanted = {_key(name) for name in names}
    mask = np.zeros(shape, dtype=bool)
    count = 0
    for item in instances:
        if _key(str(item.get("class_name", ""))) not in wanted:
            continue
        candidate = item.get("mask")
        if not isinstance(candidate, np.ndarray):
            continue
        candidate = np.asarray(candidate) > 0
        if candidate.shape != shape:
            raise ValueError("YOLO mask shape does not match the 2D crop")
        mask |= candidate
        count += 1
    return mask, count


def _measure_chip_component(
    component: np.ndarray,
    chip_union: np.ndarray,
    silver: np.ndarray,
    occlusion: np.ndarray,
    settings: ContinuitySettings,
    chip_index: int,
) -> dict[str, Any]:
    height, width = component.shape
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        raise ValueError(f"chip component {chip_index} has no pixels")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    distance = settings.outward_length_px
    outer_x1, outer_x2 = max(0, x1 - distance), min(width, x2 + distance)
    outer_y1, outer_y2 = max(0, y1 - distance), min(height, y2 + distance)

    inner_rectangle = np.zeros_like(component, dtype=bool)
    inner_rectangle[y1:y2, x1:x2] = True
    outer_rectangle = np.zeros_like(component, dtype=bool)
    outer_rectangle[outer_y1:outer_y2, outer_x1:outer_x2] = True
    ring = outer_rectangle & ~inner_rectangle & ~chip_union
    if not ring.any():
        raise ValueError(f"chip component {chip_index} has no valid outer ring")

    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    effective_occlusion = ring & occlusion
    silver_ring = ring & silver
    # Silver hidden by a configured wire/bond/thin mask still counts as
    # present. Only pixels with neither silver nor an occlusion mask are
    # missing continuity coverage.
    covered_mask = ring & (silver | occlusion)
    valid = ring
    # Sector angles are only needed inside this chip's expanded rectangle.
    # Computing them for the whole 5MP camera frame would allocate several
    # hundred MB for every piece in a streaming pipeline.
    local_slice = np.s_[outer_y1:outer_y2, outer_x1:outer_x2]
    local_ring = ring[local_slice]
    local_valid = valid[local_slice]
    local_silver = silver_ring[local_slice]
    local_covered = covered_mask[local_slice]
    local_occlusion = effective_occlusion[local_slice]
    yy, xx = np.indices(local_ring.shape, dtype=np.float64)
    yy += outer_y1
    xx += outer_x1
    angles = (np.arctan2(yy - center[1], xx - center[0]) + 2 * np.pi) % (2 * np.pi)
    indices = np.floor(angles / (2 * np.pi) * settings.sector_count).astype(np.int32)
    sectors: list[dict[str, Any]] = []
    local_missing_mask = np.zeros(local_ring.shape, dtype=bool)
    covered_count = valid_count = ignored_count = missing_count = 0
    for index in range(settings.sector_count):
        sector = local_ring & (indices == index)
        valid_sector = local_valid & (indices == index)
        silver_sector = local_silver & (indices == index)
        covered_sector = local_covered & (indices == index)
        occluded_sector = local_occlusion & (indices == index)
        ring_px, valid_px = int(sector.sum()), int(valid_sector.sum())
        silver_px, occluded_px = int(silver_sector.sum()), int(occluded_sector.sum())
        covered_px = int(covered_sector.sum())
        coverage = covered_px / valid_px if valid_px else None
        visible_ratio = valid_px / ring_px if ring_px else 0.0
        if valid_px < settings.min_valid_sector_px:
            status = "insufficient"
        else:
            valid_count += 1
            covered = (
                covered_px >= settings.min_sector_silver_px
                and coverage >= settings.min_sector_coverage
            )
            status = "covered" if covered else "missing"
            if covered:
                covered_count += 1
            else:
                missing_count += 1
                local_missing_mask |= valid_sector & ~covered_sector
        sectors.append(
            {
                "chip_index": chip_index,
                "index": index,
                "angle_start_deg": index * 360 / settings.sector_count,
                "angle_end_deg": (index + 1) * 360 / settings.sector_count,
                "ring_px": ring_px,
                "valid_px": valid_px,
                "silver_px": silver_px,
                "occluded_px": occluded_px,
                "covered_px": covered_px,
                "visible_ratio": visible_ratio,
                "coverage": coverage,
                "status": status,
            }
        )
    missing_mask = np.zeros(component.shape, dtype=bool)
    missing_mask[local_slice] = local_missing_mask
    return {
        "chip_mask": component,
        "ring_mask": ring,
        "silver_mask": silver_ring,
        "occlusion_mask": effective_occlusion,
        "missing_mask": missing_mask,
        "center_xy": center,
        "chip_area_px": int(component.sum()),
        "bbox_xyxy": [x1, y1, x2, y2],
        "outer_bbox_xyxy": [outer_x1, outer_y1, outer_x2, outer_y2],
        "ring_area_px": int(ring.sum()),
        "silver_area_px": int(silver_ring.sum()),
        "occluded_area_px": int(effective_occlusion.sum()),
        "valid_area_px": int(valid.sum()),
        "covered_sector_count": covered_count,
        "valid_sector_count": valid_count,
        "ignored_sector_count": ignored_count,
        "missing_sector_count": missing_count,
        "sectors": sectors,
    }


def measure_continuity(
    instances: Iterable[Mapping[str, Any]],
    image_shape: Sequence[int],
    settings: ContinuitySettings,
) -> ContinuityMeasurement:
    shape = (int(image_shape[0]), int(image_shape[1]))
    chip, chip_count = _union_masks(instances, (settings.chip_class,), shape)
    if chip_count == 0 or not chip.any():
        raise ValueError(f"no {settings.chip_class} mask was detected")
    silver, silver_count = _union_masks(instances, (settings.silver_class,), shape)
    if silver_count == 0 or not silver.any():
        raise ValueError(f"no {settings.silver_class} mask was detected")
    occlusion, _ = _union_masks(instances, settings.occlusion_classes, shape)
    if settings.occlusion_dilation_px:
        diameter = settings.occlusion_dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        occlusion = cv2.dilate(occlusion.astype(np.uint8), kernel) > 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(chip.astype(np.uint8), 8)
    components: list[dict[str, Any]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < settings.min_chip_area_px:
            continue
        components.append(
            _measure_chip_component(
                labels == label,
                chip,
                silver,
                occlusion,
                settings,
                len(components) + 1,
            )
        )
    if not components:
        raise ValueError("no chip component reached min_chip_area_px")
    chip_mask = np.logical_or.reduce([item["chip_mask"] for item in components])
    ring_mask = np.logical_or.reduce([item["ring_mask"] for item in components])
    silver_mask = np.logical_or.reduce([item["silver_mask"] for item in components])
    occlusion_mask = np.logical_or.reduce([item["occlusion_mask"] for item in components])
    missing_mask = np.logical_or.reduce([item["missing_mask"] for item in components])
    centers = np.asarray([item["center_xy"] for item in components], dtype=float)
    weights = np.asarray([item["chip_area_px"] for item in components], dtype=float)
    center = tuple(map(float, np.average(centers, axis=0, weights=weights)))
    chips = [
        {
            key: value
            for key, value in item.items()
            if key
            not in {"chip_mask", "ring_mask", "silver_mask", "occlusion_mask", "missing_mask"}
        }
        for item in components
    ]
    return ContinuityMeasurement(
        int(chip_mask.sum()),
        int(ring_mask.sum()),
        int(silver_mask.sum()),
        int(occlusion_mask.sum()),
        int((ring_mask & ~occlusion_mask).sum()),
        sum(item["covered_sector_count"] for item in components),
        sum(item["valid_sector_count"] for item in components),
        sum(item["ignored_sector_count"] for item in components),
        sum(item["missing_sector_count"] for item in components),
        center,
        [sector for item in components for sector in item["sectors"]],
        chips,
        chip_mask,
        ring_mask,
        silver_mask,
        occlusion_mask,
        missing_mask,
    )


def _points_from_config(
    boundary: Mapping[str, Any],
    *,
    crop_box: Sequence[int],
    circle: Sequence[int] | None,
) -> list[tuple[float, float]]:
    points = boundary.get("points", boundary.get("boundary_points"))
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        raise ValueError("silver boundary.points must contain at least three points")
    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("silver boundary.points must be an N x 2 array")
    coordinate_system = str(
        boundary.get("coordinate_system", boundary.get("coordinates", "local"))
    ).strip().casefold()
    x1, y1 = float(crop_box[0]), float(crop_box[1])
    if coordinate_system in {"local", "crop", "roi"}:
        converted = raw
    elif coordinate_system in {"image", "original", "full"}:
        converted = raw - np.asarray([x1, y1])
    elif coordinate_system in {"circle", "circle_relative", "normalized_circle"}:
        if circle is None or len(circle) != 3 or int(circle[2]) <= 0:
            raise ValueError("circle-relative silver boundary requires a detected circle")
        cx, cy, radius = map(float, circle)
        converted = np.column_stack((cx + raw[:, 0] * radius, cy + raw[:, 1] * radius))
        converted -= np.asarray([x1, y1])
    else:
        raise ValueError(
            "silver boundary.coordinate_system must be local, image, or circle"
        )
    return [tuple(map(float, point)) for point in converted]


def _read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(Path(path)), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"unable to read silver boundary template: {path}")
    return image


def _registered_template_points(
    image: np.ndarray,
    boundary: Mapping[str, Any],
) -> list[tuple[float, float]]:
    """Map calibrated template points into this image with SIFT + RANSAC.

    This is the headless equivalent of the external project's
    ``TemplateMatcher``. It is optional; circle-relative points avoid feature
    matching when the product has a reliable circular landmark.
    """
    template_path = boundary.get("template_path")
    points = boundary.get("template_points", boundary.get("points"))
    if not template_path or not isinstance(points, (list, tuple)):
        raise ValueError("template boundary needs template_path and template_points")
    template = _read_image(template_path)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max_side = int(boundary.get("max_image_side", 2400))

    def features(gray: np.ndarray):
        height, width = gray.shape[:2]
        scale = min(1.0, max_side / max(width, height))
        resized = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        ) if scale < 1 else gray
        detector = cv2.SIFT_create(nfeatures=int(boundary.get("max_features", 10000)))
        keypoints, descriptors = detector.detectAndCompute(resized, None)
        coordinates = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float64).reshape(-1, 2)
        coordinates *= [width / resized.shape[1], height / resized.shape[0]]
        return coordinates, descriptors

    template_points_features, template_descriptors = features(template_gray)
    target_points_features, target_descriptors = features(target_gray)
    if template_descriptors is None or target_descriptors is None:
        raise ValueError("silver boundary template has insufficient features")
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=80))
    pairs = matcher.knnMatch(template_descriptors, target_descriptors, k=2)
    ratio = float(boundary.get("ratio_threshold", 0.7))
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]
    unique: dict[int, Any] = {}
    for match in sorted(good, key=lambda item: item.distance):
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    min_matches = int(boundary.get("min_matches", 11))
    if len(good) < min_matches:
        raise ValueError(f"silver boundary template matches are insufficient: {len(good)} / {min_matches}")
    source = template_points_features[[match.queryIdx for match in good]]
    destination = target_points_features[[match.trainIdx for match in good]]
    matrix, inlier_mask = cv2.findHomography(
        source,
        destination,
        cv2.RANSAC,
        float(boundary.get("ransac_threshold", 5.0)),
    )
    if matrix is None or inlier_mask is None or not np.isfinite(matrix).all():
        raise ValueError("silver boundary template homography is invalid")
    inliers = int(inlier_mask.ravel().sum())
    min_inliers = int(boundary.get("min_inliers", 8))
    inlier_ratio = inliers / len(good)
    if inliers < min_inliers or inlier_ratio < float(boundary.get("min_inlier_ratio", 0.25)):
        raise ValueError("silver boundary template registration is unreliable")
    source_points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(source_points, matrix).reshape(-1, 2)
    if not np.isfinite(mapped).all():
        raise ValueError("silver boundary template points are invalid")
    return [tuple(map(float, point)) for point in mapped]


def build_boundary(
    config: Mapping[str, Any],
    *,
    crop_box: Sequence[int],
    circle: Sequence[int] | None,
    image_shape: Sequence[int],
    image: np.ndarray | None = None,
    base_dir: str | Path | None = None,
) -> Boundary | None:
    raw = config.get("boundary") if isinstance(config, Mapping) else None
    if not isinstance(raw, Mapping):
        return None
    calibration = raw
    calibration_path = raw.get("calibration_path")
    if calibration_path:
        path = Path(str(calibration_path)).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir).expanduser() / path
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("silver calibration file must contain an object")
        calibration = {**data, **{key: value for key, value in raw.items() if key != "calibration_path"}}
        if "template_path" in calibration and not Path(str(calibration["template_path"])).is_absolute():
            calibration["template_path"] = str(path.parent / str(calibration["template_path"]))
    if calibration.get("template_path"):
        if base_dir is not None and not Path(str(calibration["template_path"])).is_absolute():
            calibration = {
                **calibration,
                "template_path": str(Path(base_dir).expanduser() / str(calibration["template_path"])),
            }
        if image is None:
            raise ValueError("silver boundary template registration needs the source image")
        mapped = _registered_template_points(image, calibration)
        points = [
            (float(point[0]) - float(crop_box[0]), float(point[1]) - float(crop_box[1]))
            for point in mapped
        ]
    else:
        points = _points_from_config(calibration, crop_box=crop_box, circle=circle)
    boundary = Boundary(
        tuple(points),
        str(calibration.get("mode", calibration.get("boundary_mode", "polygon")))
        .strip()
        .casefold(),
    )
    boundary.validate_image(image_shape)
    return boundary


def _int_config(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    return int(value)


def _float_config(config: Mapping[str, Any], key: str, default: float) -> float:
    return float(config.get(key, default))


def overflow_settings(config: Mapping[str, Any]) -> OverflowSettings:
    return OverflowSettings(
        silver_class=str(config.get("silver_class", "silver")),
        tolerance_px=_float_config(config, "tolerance_px", 0.0),
        min_area_px=_int_config(config, "min_area_px", 1),
    )


def continuity_settings(config: Mapping[str, Any]) -> ContinuitySettings:
    classes = config.get("occlusion_classes", ("thin", "bond", "wire"))
    if isinstance(classes, str):
        classes = tuple(item.strip() for item in classes.split(",") if item.strip())
    elif isinstance(classes, (list, tuple)):
        classes = tuple(str(item) for item in classes)
    else:
        raise ValueError("silver continuity occlusion_classes must be a list or comma string")
    return ContinuitySettings(
        silver_class=str(config.get("silver_class", "silver")),
        chip_class=str(config.get("chip_class", "chip")),
        occlusion_classes=classes,
        outward_length_px=_int_config(config, "outward_length_px", 20),
        sector_count=_int_config(config, "sector_count", 72),
        min_sector_silver_px=_int_config(config, "min_sector_silver_px", 3),
        min_sector_coverage=_float_config(config, "min_sector_coverage", 0.01),
        min_valid_sector_px=_int_config(config, "min_valid_sector_px", 1),
        occlusion_dilation_px=_int_config(config, "occlusion_dilation_px", 0),
        min_chip_area_px=_int_config(config, "min_chip_area_px", 100),
        min_visible_sector_ratio=_float_config(config, "min_visible_sector_ratio", 0.1),
    )


def inspect_masks(
    instances: Iterable[Mapping[str, Any]],
    image_shape: Sequence[int],
    config: Mapping[str, Any] | None,
    *,
    crop_box: Sequence[int] = (0, 0, 0, 0),
    circle: Sequence[int] | None = None,
    image: np.ndarray | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run overflow and continuity checks on one YOLO result.

    A missing boundary or a model without silver/chip masks is represented as
    an explicit ``uncertain``/``not_configured`` status.  It is never silently
    converted into a passing result.
    """
    settings = dict(config or {})
    overflow_config = settings.get("overflow")
    continuity_config = settings.get("continuity")
    overflow_enabled = isinstance(overflow_config, Mapping)
    continuity_enabled = isinstance(continuity_config, Mapping)
    if not overflow_enabled and not continuity_enabled:
        return {
            "enabled": False,
            "silver_overflow": {"status": "disabled", "is_ok": None},
            "disconnection": {"status": "disabled", "is_ok": None},
        }

    shape = (int(image_shape[0]), int(image_shape[1]))
    boundary_error = None
    overflow_config = dict(overflow_config or {})
    continuity_config = dict(continuity_config or {})
    raw_boundary = overflow_config.get("boundary")
    chip_expansion = (
        isinstance(raw_boundary, Mapping)
        and str(raw_boundary.get("mode", "")).strip().casefold()
        in {"chip_expansion", "chip_expand", "chip"}
    )
    if chip_expansion:
        boundary = None
    else:
        try:
            boundary = build_boundary(
                overflow_config,
                crop_box=crop_box,
                circle=circle,
                image_shape=shape,
                image=image,
                base_dir=base_dir,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            boundary = None
            boundary_error = str(exc)
    instances = list(instances)

    overflow: dict[str, Any]
    if not overflow_enabled:
        overflow = {"status": "disabled", "is_ok": None}
    elif chip_expansion:
        try:
            expansion = raw_boundary or {}
            measurement = measure_chip_expansion_overflow(
                instances,
                shape,
                overflow_settings(overflow_config),
                chip_class=str(expansion.get("chip_class", "chip")),
                outward_length_px=int(expansion.get("outward_length_px", 20)),
                min_chip_area_px=int(expansion.get("min_chip_area_px", 100)),
            )
            overflow = {
                "status": "overflow" if measurement.defect_area_px else "ok",
                "is_ok": not bool(measurement.defect_area_px),
                "boundary": {
                    "mode": "chip_expansion",
                    "chip_class": str(expansion.get("chip_class", "chip")),
                    "outward_length_px": int(expansion.get("outward_length_px", 20)),
                },
                "measurement": measurement.summary(),
            }
        except ValueError as exc:
            overflow = {
                "status": "no_silver" if str(exc).startswith("no silver") else "uncertain",
                "is_ok": None,
                "message": str(exc),
                "boundary": {
                    "mode": "chip_expansion",
                    "chip_class": str((raw_boundary or {}).get("chip_class", "chip")),
                    "outward_length_px": int((raw_boundary or {}).get("outward_length_px", 20)),
                },
            }
    elif boundary is None:
        overflow = {
            "status": "uncertain" if boundary_error else "not_configured",
            "is_ok": None,
            "message": boundary_error or "silver overflow boundary is not configured",
        }
    else:
        try:
            silver_mask, silver_count = _union_masks(
                instances,
                (str(overflow_config.get("silver_class", "silver")),),
                shape,
            )
            if silver_count == 0 or not silver_mask.any():
                raise ValueError(
                    f"no {overflow_config.get('silver_class', 'silver')} mask was detected"
                )
            measurement = measure_overflow(
                silver_mask,
                boundary,
                overflow_settings(overflow_config),
            )
            overflow = {
                "status": "overflow" if measurement.defect_area_px else "ok",
                "is_ok": not bool(measurement.defect_area_px),
                "boundary": {
                    "mode": boundary.mode,
                    "points": [list(point) for point in boundary.points],
                },
                "measurement": measurement.summary(),
            }
        except ValueError as exc:
            overflow = {
                "status": "no_silver" if str(exc).startswith("no ") else "uncertain",
                "is_ok": None,
                "message": str(exc),
                "boundary": {
                    "mode": boundary.mode,
                    "points": [list(point) for point in boundary.points],
                },
            }

    if not continuity_enabled:
        continuity = {"status": "disabled", "is_ok": None}
    else:
        try:
            measurement = measure_continuity(
                instances, shape, continuity_settings(continuity_config)
            )
            disconnected = measurement.is_disconnected
            continuity = {
                "status": (
                    "uncertain"
                    if disconnected is None
                    else "disconnected"
                    if disconnected
                    else "ok"
                ),
                "is_ok": None if disconnected is None else not disconnected,
                "measurement": measurement.summary(),
            }
        except ValueError as exc:
            continuity = {"status": "uncertain", "is_ok": None, "message": str(exc)}

    return {
        "enabled": True,
        "silver_overflow": overflow,
        "disconnection": continuity,
    }


def offset_geometry(value: Any, offset: Sequence[int]) -> Any:
    """Translate diagnostic geometry from crop-local to original-image pixels."""
    if isinstance(value, dict):
        return {
            key: offset_geometry(item, offset)
            for key, item in value.items()
            if key not in {"silver_mask", "outside_mask", "defect_mask", "chip_mask", "ring_mask", "occlusion_mask", "missing_mask"}
        }
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            converted = [value[0] + offset[0], value[1] + offset[1], value[2] + offset[0], value[3] + offset[1]]
            return converted
        if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            return [value[0] + offset[0], value[1] + offset[1]]
        return [offset_geometry(item, offset) for item in value]
    return value
