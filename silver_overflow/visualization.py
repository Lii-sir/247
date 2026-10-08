"""Draw exact mapped boundaries and overflow masks without mutating input images."""

import cv2 as cv
import numpy as np

from .geometry import Boundary
from .pipeline import InspectionResult


def draw_boundary(image, boundary: Boundary):
    points = np.rint(boundary.points).astype(np.int32)
    color = (255, 210, 0)
    if boundary.mode == "polygon":
        cv.polylines(image, [points], True, color, 2, cv.LINE_AA)
    else:
        a, b, _ = np.asarray(boundary.points)
        direction = (b - a) / np.linalg.norm(b - a)
        extent = 2 * max(image.shape[:2])
        start, end = np.rint(a - direction * extent).astype(int), np.rint(a + direction * extent).astype(int)
        _, start, end = cv.clipLine((0, 0, image.shape[1], image.shape[0]), tuple(map(int, start)), tuple(map(int, end)))
        cv.line(image, start, end, color, 2, cv.LINE_AA)
    for index, (x, y) in enumerate(points, 1):
        cv.circle(image, (int(x), int(y)), 4, color, -1, cv.LINE_AA)
        label = "INSIDE" if boundary.mode == "line" and index == 3 else f"P{index}"
        cv.putText(image, label, (int(x) + 6, int(y) - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)


def render_overlay(result: InspectionResult):
    image = result.segmentation.image.copy()
    m = result.measurement
    if m is not None:
        colors = image.copy()
        colors[m.silver_mask] = (70, 200, 70)
        colors[m.outside_mask] = (0, 190, 255)
        colors[m.defect_mask] = (30, 30, 255)
        image = cv.addWeighted(image, 0.45, colors, 0.55, 0)
        for region in m.regions:
            x1, y1, x2, y2 = region["box_xyxy"]
            cv.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (30, 30, 255), 2)
    if result.boundary:
        draw_boundary(image, result.boundary)
    return image


def render_comparison(result: InspectionResult):
    pair = np.hstack((result.segmentation.image, render_overlay(result)))
    header = np.full((64, pair.shape[1], 3), 35, dtype=np.uint8)
    label = f"{result.status.upper()} | silver instances: {result.silver_count}"
    if result.measurement:
        label += f" | defect: {result.measurement.defect_area_px} px | max: {result.measurement.max_outside_distance_px:.2f} px"
    color = (30, 50, 255) if result.is_defect else (70, 200, 70) if result.is_defect is False else (0, 190, 255)
    cv.putText(header, label, (12, 26), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv.LINE_AA)
    cv.putText(header, "ORIGINAL / RESULT   cyan: boundary | green: silver | amber: ignored overflow | red: defect",
               (12, 51), cv.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv.LINE_AA)
    return np.vstack((header, pair))
