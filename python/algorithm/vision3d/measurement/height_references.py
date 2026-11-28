"""Classify measurement regions in 2D before fitting their reference planes."""
from dataclasses import dataclass

import cv2
import numpy as np


def platform_region(crop_box, boxes):
    """Convert the same platform boxes used for fitting to crop coordinates."""
    ox, oy = crop_box[:2]
    return [(left - ox, top - oy, right - ox, bottom - oy) for left, top, right, bottom in boxes]


def chip_region(mask, bounds=None):
    binary = np.asarray(mask, dtype=np.uint8)
    x, y, width, height = cv2.boundingRect(binary) if bounds is None else bounds
    region = np.zeros_like(binary)
    if width:
        area = np.s_[y:y + height, x:x + width]
        contours, _ = cv2.findContours(binary[area], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(region[area], contours, -1, 255, cv2.FILLED)
    return region


def reference_chips(mask, platform, regions, bounds=None):
    """Return chip IDs, () for the shared platform, or None for unknown ownership."""
    binary = mask if mask.dtype == np.uint8 else (mask > 0).astype(np.uint8)
    x, y, width, height = cv2.boundingRect(binary) if bounds is None else bounds
    ys, xs = np.nonzero(binary[y:y + height, x:x + width])
    if not xs.size:
        return None
    ys += y
    xs += x
    for left, top, right, bottom in platform:
        if ((left <= xs) & (xs <= right) & (top <= ys) & (ys <= bottom)).any():
            return ()
    overlaps = [(key, int(np.count_nonzero(region[ys, xs]))) for key, region in regions.items()]
    containing = [key for key, count in overlaps if count == xs.size]
    if containing:
        return (min(containing, key=lambda key: np.count_nonzero(regions[key])),)
    intersecting = tuple(key for key, count in overlaps if count)
    return intersecting if len(intersecting) > 1 else None if intersecting else ()


@dataclass(frozen=True)
class MeasurementReference:
    name: str
    plane: dict | None = None
    chip_id: int | None = None
    fixed_height_mm: float | None = None


def resolve_references(routes, platform_plane, chip_planes):
    """Resolve each distinct set of candidate chips once for the whole piece."""
    resolved = {None: MeasurementReference("unclassified"),
                (): MeasurementReference("wire_platform", platform_plane)}
    for ids in dict.fromkeys(routes.values()):
        if not ids:
            continue
        if len(ids) == 1:
            chip_id = ids[0]
            resolved[ids] = MeasurementReference("chip", chip_planes[chip_id], chip_id)
            continue
        candidates = [(chip_id, chip_planes[chip_id]) for chip_id in ids]
        if any(plane is None for _, plane in candidates):
            resolved[ids] = MeasurementReference("chip")
            continue
        chip_id, plane = max(candidates, key=lambda item: item[1]["representative_height_mm"])
        resolved[ids] = MeasurementReference("chip", plane, chip_id, plane["representative_height_mm"])
    return {job: resolved[ids] for job, ids in routes.items()}
