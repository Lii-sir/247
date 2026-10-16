"""Frozen full-frame silver-continuity reference from the pre-20261016 implementation.

Used only by differential tests and benchmarks; never imported by the service.
"""
from __future__ import annotations
import importlib
from typing import Any, Iterable, Mapping, Sequence
import cv2
import numpy as np

silver = importlib.import_module("algorithm.2D.silver_inspection")
ContinuitySettings = silver.ContinuitySettings
ContinuityMeasurement = silver.ContinuityMeasurement
_union_masks = silver._union_masks


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
    local_silver = silver_ring[local_slice]
    local_covered = covered_mask[local_slice]
    local_occlusion = effective_occlusion[local_slice]
    # Only ring pixels contribute. Count their sectors once instead of
    # allocating and scanning five masks per sector (normally 72 times).
    ring_y, ring_x = np.nonzero(local_ring)
    yy = ring_y.astype(np.float64) + outer_y1
    xx = ring_x.astype(np.float64) + outer_x1
    angles = (np.arctan2(yy - center[1], xx - center[0]) + 2 * np.pi) % (2 * np.pi)
    indices = np.floor(angles / (2 * np.pi) * settings.sector_count).astype(np.int32)
    ring_counts = np.bincount(indices, minlength=settings.sector_count)
    silver_counts = np.bincount(
        indices[local_silver[ring_y, ring_x]], minlength=settings.sector_count
    )
    covered_counts = np.bincount(
        indices[local_covered[ring_y, ring_x]], minlength=settings.sector_count
    )
    occluded_counts = np.bincount(
        indices[local_occlusion[ring_y, ring_x]], minlength=settings.sector_count
    )
    missing_sectors = np.zeros(len(ring_counts), dtype=bool)
    sectors: list[dict[str, Any]] = []
    local_missing_mask = np.zeros(local_ring.shape, dtype=bool)
    covered_count = valid_count = ignored_count = missing_count = 0
    for index in range(settings.sector_count):
        ring_px = valid_px = int(ring_counts[index])
        silver_px, occluded_px = int(silver_counts[index]), int(occluded_counts[index])
        covered_px = int(covered_counts[index])
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
                missing_sectors[index] = True
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
    local_missing_mask[ring_y, ring_x] = (
        missing_sectors[indices] & ~local_covered[ring_y, ring_x]
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


