"""Check that every configured bending ROI contains a wire or thin mask."""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def check_wire_roi_presence(
    instances: Iterable[dict[str, Any]],
    roi_boxes: Sequence[Sequence[int]],
    offset: Sequence[int] = (0, 0),
) -> list[dict[str, Any]]:
    candidates = [
        item for item in instances
        if str(item.get("class_name", "")).casefold() in {"wire", "thin"}
        and isinstance(item.get("mask"), np.ndarray)
        and np.asarray(item["mask"]).ndim == 2
    ]
    ox, oy = int(offset[0]), int(offset[1])
    output = []
    by_id = {int(item.get("instance_id", 0)): item for item in candidates}
    roi_pixel_counts: list[list[tuple[int, int]]] = [[] for _ in roi_boxes]
    roi_instance_ids: list[list[int]] = [[] for _ in roi_boxes]
    for item in candidates:
        bending = item.get("bending_2d") or {}
        if "roi_id" in bending:
            # Reuse the cleaned-mask assignment and denominator of the actual
            # measurement, including invalid intersection measurements.
            roi_id = bending["roi_id"]
            if roi_id is not None and 1 <= int(roi_id) <= len(roi_boxes):
                selected = int(roi_id) - 1
                roi_pixel_counts[selected].append((int(item.get("instance_id", 0)),
                                                  int(bending["roi_mask_pixel_count"])))
                roi_instance_ids[selected].append(int(item.get("instance_id", 0)))
                continue
            if bending.get("status") != "empty_mask":
                continue
            # A rejected/empty cleaned mask still reports an unavailable
            # measurement in the ROI containing its original instance.
        mask = np.asarray(item["mask"]) > 0
        height, width = mask.shape
        overlaps = []
        for raw_box in roi_boxes:
            x1, y1, x2, y2 = (int(value) for value in raw_box)
            left, top = max(0, x1), max(0, y1)
            right, bottom = min(width, x2), min(height, y2)
            count = int(np.count_nonzero(mask[top:bottom, left:right])) if left < right and top < bottom else 0
            overlaps.append(count)
        if not overlaps or max(overlaps) <= 0:
            continue
        # Legacy/unmeasurable instances use greatest raw pixel overlap.
        # Measured instances reuse the cleaned-mask assignment above.
        selected = int(np.argmax(overlaps))
        roi_pixel_counts[selected].append((int(item.get("instance_id", 0)), overlaps[selected]))
        roi_instance_ids[selected].append(int(item.get("instance_id", 0)))

    for index, raw_box in enumerate(roi_boxes, start=1):
        x1, y1, x2, y2 = (int(value) for value in raw_box)
        assigned = roi_instance_ids[index - 1]
        mask_pixels = sum(count for _, count in roi_pixel_counts[index - 1])
        bending_checks = []
        for instance_id in assigned:
            bending = by_id[instance_id].get("bending_2d") or {}
            bending_checks.append({
                "instance_id": instance_id,
                "mask_ratio": float(bending.get("mask_ratio", 0.0)),
                "status": str(bending.get("status", "bending_unavailable")),
                "is_valid": bool(bending.get("is_valid", False)),
                "is_ok": bool(bending.get("is_valid", False) and bending.get("is_ok", False)),
            })
        has_wire = bool(assigned)
        bending_ok = has_wire and all(item["is_ok"] for item in bending_checks)
        if not has_wire:
            status = "wire_missing"
        elif bending_ok:
            status = "ok"
        elif any(not item["is_valid"] for item in bending_checks):
            status = "bending_unavailable"
        else:
            status = "bending_out_of_range"
        output.append({
            "roi_id": index,
            "box": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
            "mask_pixel_count": mask_pixels,
            "instance_ids": sorted(set(assigned)),
            "bending_checks": bending_checks,
            "is_ok": bool(bending_ok),
            "status": status,
        })
    return output
