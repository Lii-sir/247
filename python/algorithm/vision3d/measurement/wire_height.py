"""3D wire-height flow: thin mask components to centerlines and find P95 points."""
from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


DEFAULTS = {
    "substrate_aois": [[0.02, 0.00, 0.18, 0.35], [0.02, 0.65, 0.18, 1.00]],
    "wire_x_range": [300, 400],
    "mask_threshold_gray": 80.0,
    "mask_percentile": 65.0,
    "min_component_area_px": 40,
    "depth_percentile": 95.0,
    "invalid_gray_values": [0, 65535],
    "plane_initial_percentile": 40.0,
    "plane_min_valid_points": 200,
    "plane_sample_stride": 4,
    "plane_fit_iterations": 5,
    "plane_residual_threshold_gray": 140.0,
    "plane_mad_multiplier": 4.4478,
    "mask_close_iterations": 1,
    "mask_open_iterations": 1,
    "gray_offset": 32768.0,
    "vertical_resolution_mm": 0.00105,
    "max_wire_substrate_height_mm": 0.8,
}

_config = dict(DEFAULTS)


def configure(values: Dict[str, Any] | None) -> None:
    global _config
    _config = {**DEFAULTS, **(values or {})}


def _roi(values, width: int, height: int):
    x1, y1, x2, y2 = (float(value) for value in values)
    return round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)


def _fit_substrate(depth: np.ndarray, roi) -> float:
    """Fit the substrate surface and return its mean gray value in the ROI."""
    height, width = depth.shape
    x1, y1, x2, y2 = _roi(roi, width, height)
    patch = depth[y1:y2, x1:x2]
    valid = ~np.isin(patch, np.asarray(_config["invalid_gray_values"], dtype=patch.dtype))
    yy, xx = np.nonzero(valid)
    xx = xx + x1
    yy = yy + y1
    z = depth[yy, xx].astype(np.float64)
    if z.size < int(_config["plane_min_valid_points"]):
        raise ValueError("substrate ROI has too few valid pixels")
    sample = np.arange(z.size)[::int(_config["plane_sample_stride"])]
    x, y, z = xx[sample].astype(float), yy[sample].astype(float), z[sample]
    keep = z <= np.percentile(z, float(_config["plane_initial_percentile"]))
    for _ in range(int(_config["plane_fit_iterations"])):
        design = np.column_stack((x[keep], y[keep], np.ones(np.count_nonzero(keep))))
        coef = np.linalg.lstsq(design, z[keep], rcond=None)[0]
        residual = z - (coef[0] * x + coef[1] * y + coef[2])
        mad = np.median(np.abs(residual[keep] - np.median(residual[keep])))
        keep = np.abs(residual) <= max(float(_config["plane_residual_threshold_gray"]),
                                       float(_config["plane_mad_multiplier"]) * mad)
    grid_y, grid_x = np.indices((y2 - y1, x2 - x1), dtype=np.float64)
    predicted = coef[0] * (grid_x + x1) + coef[1] * (grid_y + y1) + coef[2]
    return float(np.mean(predicted))


def _wire_mask(piece: np.ndarray, substrate_gray: float) -> np.ndarray:
    valid = ~np.isin(piece, np.asarray(_config["invalid_gray_values"], dtype=piece.dtype))
    residual = piece.astype(np.float64) - substrate_gray
    threshold = max(float(_config["mask_threshold_gray"]),
                    float(np.percentile(residual[valid], float(_config["mask_percentile"]))))
    mask = (valid & (residual >= threshold)).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                            iterations=int(_config["mask_close_iterations"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                            iterations=int(_config["mask_open_iterations"]))
    return mask.astype(bool)


def _thin_to_centerline(mask: np.ndarray) -> np.ndarray:
    """Use Zhang-Suen morphological thinning to obtain a one-pixel centerline."""
    image = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:]
            p5, p6, p7 = image[2:, 2:], image[2:, 1:-1], image[2:, :-2]
            p8, p9 = image[1:-1, :-2], image[:-2, :-2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            for first, second in ((p3, p4), (p4, p5), (p5, p6), (p6, p7),
                                  (p7, p8), (p8, p9), (p9, p2)):
                transitions += ((first == 0) & (second == 1)).astype(np.uint8)
            if step == 0:
                preserve_1 = p2 * p4 * p6 == 0
                preserve_2 = p4 * p6 * p8 == 0
            else:
                preserve_1 = p2 * p4 * p8 == 0
                preserve_2 = p2 * p6 * p8 == 0
            remove = ((image[1:-1, 1:-1] == 1) & (neighbors >= 2) &
                      (neighbors <= 6) & (transitions == 1) & preserve_1 & preserve_2)
            if np.any(remove):
                image[1:-1, 1:-1][remove] = 0
                changed = True
    return image[1:-1, 1:-1].astype(bool)


def build_full_mask(depth: np.ndarray, pieces: list[Dict[str, Any]]):
    """Build the same full-depth wire mask used by the experiment."""
    full_mask = np.zeros(depth.shape, dtype=bool)
    substrate_gray_by_piece = {}
    for piece in pieces:
        x1, y1, x2, y2 = (int(piece[key]) for key in ("x1", "y1", "x2", "y2"))
        local = depth[y1:y2, x1:x2]
        substrate_grays = [_fit_substrate(local, roi) for roi in _config["substrate_aois"]]
        substrate_gray = float(np.mean(substrate_grays))
        piece_id = int(piece["piece_id"])
        substrate_gray_by_piece[piece_id] = substrate_gray
        full_mask[y1:y2, x1:x2] = _wire_mask(local, substrate_gray)
    return full_mask, substrate_gray_by_piece


def measure(depth: np.ndarray, pieces: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Crop mask by X and return one centerline depth-P95 point per component."""
    mask, substrate_grays = build_full_mask(depth, pieces)
    x1, x2 = (int(value) for value in _config["wire_x_range"])
    x1, x2 = max(0, x1), min(depth.shape[1], x2)
    cropped = np.zeros_like(mask)
    cropped[:, x1:x2] = mask[:, x1:x2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cropped.astype(np.uint8), 8)
    minimum = int(_config["min_component_area_px"])
    percentile = float(_config["depth_percentile"])
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum:
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels[top:top + height, left:left + width] == label
        ys, xs = np.where(_thin_to_centerline(component))
        xs, ys = xs + left, ys + top
        values = depth[ys, xs]
        valid = ~np.isin(values, np.asarray(_config["invalid_gray_values"], dtype=values.dtype))
        if not np.any(valid):
            continue
        values, xs, ys = values[valid], xs[valid], ys[valid]
        percentile_value = float(np.percentile(values, percentile))
        index = int(np.argmin(np.abs(values.astype(np.float64) - percentile_value)))
        components.append({"mask_area_px": area, "centerline_pixel_count": int(values.size),
                           "depth_percentile": percentile,
                           "depth_percentile_value": percentile_value,
                           "percentile_point_depth": int(values[index]),
                           "highest_point_x": int(xs[index]), "highest_point_y": int(ys[index])})

    output = []
    for piece in pieces:
        piece_id = int(piece["piece_id"])
        y1, y2 = int(piece["y1"]), int(piece["y2"])
        local = [item for item in components if y1 <= item["highest_point_y"] < y2]
        local.sort(key=lambda item: (item["highest_point_y"], item["highest_point_x"]))
        substrate_gray = substrate_grays[piece_id]
        resolution = float(_config["vertical_resolution_mm"])
        offset = float(_config["gray_offset"])
        substrate_height = (substrate_gray - offset) * resolution
        point_heights = [(item["percentile_point_depth"] - offset) * resolution for item in local]
        differences = [height - substrate_height for height in point_heights]
        is_ok = bool(local) and all(
            value < float(_config["max_wire_substrate_height_mm"]) for value in differences
        )
        points = [{"height_mm": height, "height_diff_mm": difference,
                   "x": item["highest_point_x"], "y": item["highest_point_y"]}
                  for item, height, difference in zip(local, point_heights, differences)]
        output.append({"piece_id": piece_id, "is_ok": is_ok,
                       "substrate_height_mm": substrate_height, "points": points})
    return output
