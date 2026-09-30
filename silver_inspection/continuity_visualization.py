"""Visualization for chip ring and silver continuity diagnostics."""

import cv2 as cv
import numpy as np

from .continuity_pipeline import ContinuityResult


def render_continuity_overlay(result: ContinuityResult):
    image = result.segmentation.image.copy()
    measurement = result.measurement
    if measurement is None:
        return image
    layer = image.copy()
    layer[measurement.ring_mask] = (255, 0, 255)       # magenta: check ring
    layer[measurement.silver_mask] = (70, 205, 70)     # green: detected silver
    layer[measurement.missing_mask] = (30, 30, 255)    # red: missing silver sector
    layer[measurement.occlusion_mask] = (0, 215, 255)  # yellow: ignored thin/bond; keep occlusion visible
    image = cv.addWeighted(image, 0.45, layer, 0.55, 0)
    for chip in measurement.chips:
        x1, y1, x2, y2 = chip["bbox_xyxy"]
        ox1, oy1, ox2, oy2 = chip["outer_bbox_xyxy"]
        cv.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1, cv.LINE_AA)
        cv.rectangle(image, (ox1, oy1), (ox2 - 1, oy2 - 1), (255, 190, 0), 1, cv.LINE_AA)
    cx, cy = map(int, np.rint(measurement.center_xy))
    cv.drawMarker(image, (cx, cy), (255, 255, 255), cv.MARKER_CROSS, 18, 2, cv.LINE_AA)
    text = f"{result.status.upper()}  {measurement.covered_sector_count}/{measurement.valid_sector_count} sectors"
    color = (30, 30, 255) if result.is_defect else (70, 200, 70) if result.is_defect is False else (0, 190, 255)
    cv.putText(image, text, (12, 28), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv.LINE_AA)
    return image


def render_continuity_comparison(result: ContinuityResult):
    overlay = render_continuity_overlay(result)
    pair = np.hstack((result.segmentation.image, overlay))
    header = np.full((64, pair.shape[1], 3), 35, dtype=np.uint8)
    cv.putText(header, result.message, (12, 26), cv.FONT_HERSHEY_SIMPLEX, 0.52,
               (235, 235, 235), 1, cv.LINE_AA)
    cv.putText(header, "MAGENTA: ring | WHITE: chip bbox | CYAN: expanded bbox | GREEN: silver | RED: break",
               (12, 51), cv.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1, cv.LINE_AA)
    return np.vstack((header, pair))
