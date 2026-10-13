"""Measure chip rotation from YOLO chip masks using a minimum-area rectangle."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


def measure_chip_alignment(
    instances: Iterable[dict[str, Any]],
    config: Mapping[str, Any] | None,
    offset: Sequence[int] = (0, 0),
) -> dict[str, Any]:
    if config is None:
        return {"enabled": False, "is_ok": None, "chips": []}
    settings = dict(config or {})

    max_angle = float(settings.get("max_angle_deg", 3.0))
    if not np.isfinite(max_angle) or not 0.0 <= max_angle <= 45.0:
        raise ValueError("chip_alignment_2d.max_angle_deg must be in [0, 45]")
    min_area = int(settings.get("min_area_px", 1))
    if min_area < 1:
        raise ValueError("chip_alignment_2d.min_area_px must be positive")

    ox, oy = int(offset[0]), int(offset[1])
    results = []
    for instance in instances:
        if str(instance.get("class_name", "")).casefold() != "chip":
            continue
        mask = np.asarray(instance.get("mask"), dtype=np.uint8)
        if mask.ndim != 2 or cv2.countNonZero(mask) < min_area:
            continue
        binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0]
        if not contours:
            continue
        points = np.concatenate(contours, axis=0)
        rotated_rect = cv2.minAreaRect(points)
        (center_x, center_y), (width, height), _ = rotated_rect
        if width <= 0 or height <= 0:
            continue
        corners = cv2.boxPoints(rotated_rect)
        edge_vectors = np.roll(corners, -1, axis=0) - corners
        lengths = np.linalg.norm(edge_vectors, axis=1)
        long_edge = edge_vectors[int(np.argmax(lengths))]
        angle = float(np.degrees(np.arctan2(long_edge[1], long_edge[0])))
        # The chip may be laid out horizontally or vertically. Report the
        # signed deviation from the nearest image axis in [-45, 45) degrees.
        deviation = float((angle + 45.0) % 90.0 - 45.0)
        aligned = abs(deviation) <= max_angle
        corners[:, 0] += ox
        corners[:, 1] += oy
        rectangle_area = float(width * height)
        mask_area = int(cv2.countNonZero(binary))
        result = {
            "instance_id": int(instance.get("instance_id", 0)),
            "class_name": "chip",
            "center_xy": [round(float(center_x + ox), 2), round(float(center_y + oy), 2)],
            "rectangle": np.rint(corners).astype(int).tolist(),
            "width_px": round(float(max(width, height)), 2),
            "height_px": round(float(min(width, height)), 2),
            "angle_deg": round(angle, 3),
            "axis_deviation_deg": round(deviation, 3),
            "max_angle_deg": max_angle,
            "mask_fill_ratio": round(mask_area / rectangle_area, 4) if rectangle_area else 0.0,
            "is_ok": bool(aligned),
            "status": "aligned" if aligned else "rotated",
        }
        instance["chip_alignment"] = result
        results.append(result)
    return {
        "enabled": True,
        "is_ok": bool(results) and all(item["is_ok"] for item in results),
        "status": "ok" if results and all(item["is_ok"] for item in results) else
                  "rotated" if results else "no_chip",
        "max_angle_deg": max_angle,
        "chips": results,
    }
