"""按工件类型检测目标圆并生成热力图/score 使用的 ignore mask。"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def read_rgb(path: Path) -> np.ndarray:
    """读取并处理 EXIF 方向，返回 RGB uint8 数组。"""
    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB"))


def load_configs(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("圆检测配置必须是 JSON 对象。")
    return payload


def category_config(configs: dict, category: str) -> dict:
    value = configs.get(category, configs.get("default", {}))
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"工件类型 {category} 的圆检测配置必须是对象。")
    return value


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值。")
    return value


def _roi_pixels(shape: tuple[int, int], roi: list[float] | tuple[float, ...]) -> tuple[int, int, int, int]:
    height, width = shape
    if len(roi) != 4:
        raise ValueError("roi 必须是归一化的 [x, y, w, h]。")
    x, y, w, h = (_finite(item, "roi") for item in roi)
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 - x and 0 < h <= 1 - y):
        raise ValueError("roi 必须满足 0<=x,y，且 x+w、y+h 不超过 1。")
    x0, y0 = int(round(x * width)), int(round(y * height))
    x1, y1 = int(round((x + w) * width)), int(round((y + h) * height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("roi 在当前图片上为空。")
    return x0, y0, x1, y1


def _edge_score(gray: np.ndarray, x: float, y: float, radius: float) -> float:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    samples = []
    for delta in (-2.0, 0.0, 2.0):
        px = np.rint(x + (radius + delta) * np.cos(angles)).astype(np.int32)
        py = np.rint(y + (radius + delta) * np.sin(angles)).astype(np.int32)
        valid = (px >= 0) & (py >= 0) & (px < gray.shape[1]) & (py < gray.shape[0])
        samples.append(float(np.mean(gradient[py[valid], px[valid]])) if valid.any() else 0.0)
    return float(np.mean(samples))


def _circle_from_three_points(points: np.ndarray) -> tuple[float, float, float] | None:
    """由三个点计算圆；近似共线时返回 None。"""
    (x1, y1), (x2, y2), (x3, y3) = points.astype(np.float64)
    denominator = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(denominator) < 1e-8:
        return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    center_x = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / denominator
    center_y = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / denominator
    radius = math.hypot(x1 - center_x, y1 - center_y)
    if not all(math.isfinite(value) for value in (center_x, center_y, radius)):
        return None
    return center_x, center_y, radius


def _refine_circle(points: np.ndarray) -> tuple[float, float, float] | None:
    """用最小二乘在 RANSAC 内点上精修圆。"""
    if len(points) < 3:
        return None
    values = points.astype(np.float64)
    matrix = np.column_stack((2.0 * values[:, 0], 2.0 * values[:, 1], np.ones(len(values))))
    target = values[:, 0] ** 2 + values[:, 1] ** 2
    try:
        center_x, center_y, constant = np.linalg.lstsq(matrix, target, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    radius_squared = constant + center_x * center_x + center_y * center_y
    if radius_squared <= 0:
        return None
    radius = math.sqrt(float(radius_squared))
    if not all(math.isfinite(value) for value in (center_x, center_y, radius)):
        return None
    return float(center_x), float(center_y), float(radius)


def _circle_support(
    points: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    tolerance: float,
    angle_bins: int = 36,
) -> tuple[np.ndarray, float, float]:
    distances = np.hypot(points[:, 0] - center_x, points[:, 1] - center_y)
    inliers = np.abs(distances - radius) <= tolerance
    inlier_ratio = float(np.mean(inliers)) if len(points) else 0.0
    if not np.any(inliers):
        return inliers, inlier_ratio, 0.0
    angles = np.arctan2(points[inliers, 1] - center_y, points[inliers, 0] - center_x)
    indices = np.floor((angles + np.pi) * angle_bins / (2.0 * np.pi)).astype(np.int32)
    indices = np.clip(indices, 0, angle_bins - 1)
    coverage = float(len(np.unique(indices)) / angle_bins)
    return inliers, inlier_ratio, coverage


def _ransac_circle(
    contour: np.ndarray,
    min_radius: float,
    max_radius: float,
    params: dict,
    seed: int,
) -> dict | None:
    """从轮廓点稳健拟合圆，并返回内点率与角度覆盖率。"""
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < 6:
        return None
    maximum_points = max(24, int(params.get("ransac_max_points", 720)))
    rng = np.random.default_rng(seed)
    if len(points) > maximum_points:
        points = points[rng.choice(len(points), maximum_points, replace=False)]
    iterations = max(20, int(params.get("ransac_iterations", 180)))
    tolerance_ratio = float(params.get("ransac_tolerance_ratio", 0.03))
    best: tuple[float, int, tuple[float, float, float], np.ndarray] | None = None
    for _ in range(iterations):
        model = _circle_from_three_points(points[rng.choice(len(points), 3, replace=False)])
        if model is None:
            continue
        center_x, center_y, radius = model
        if not min_radius <= radius <= max_radius:
            continue
        tolerance = max(1.5, tolerance_ratio * radius)
        inliers, _, coverage = _circle_support(points, center_x, center_y, radius, tolerance)
        count = int(np.count_nonzero(inliers))
        objective = count * (0.5 + 0.5 * coverage)
        if best is None or objective > best[0]:
            best = (objective, count, model, inliers)
    if best is None:
        return None
    refined = _refine_circle(points[best[3]]) or best[2]
    center_x, center_y, radius = refined
    if not min_radius <= radius <= max_radius:
        return None
    tolerance = max(1.5, tolerance_ratio * radius)
    _, inlier_ratio, coverage = _circle_support(
        points, center_x, center_y, radius, tolerance
    )
    if inlier_ratio < float(params.get("min_ransac_inlier_ratio", 0.45)):
        return None
    if coverage < float(params.get("min_angular_coverage", 0.50)):
        return None
    return {
        "center_x": center_x,
        "center_y": center_y,
        "radius": radius,
        "ransac_inlier_ratio": inlier_ratio,
        "angular_coverage": coverage,
    }


def _dark_contrast(gray: np.ndarray, center_x: float, center_y: float, radius: float) -> float:
    """估计圆内相对外环的暗度；正值表示圆内更暗。"""
    height, width = gray.shape
    inner = np.zeros((height, width), dtype=np.uint8)
    outer = np.zeros((height, width), dtype=np.uint8)
    center = (int(round(center_x)), int(round(center_y)))
    cv2.circle(inner, center, max(1, int(round(0.72 * radius))), 255, thickness=-1)
    cv2.circle(outer, center, max(2, int(round(1.25 * radius))), 255, thickness=-1)
    cv2.circle(outer, center, max(1, int(round(1.05 * radius))), 0, thickness=-1)
    inside_values = gray[inner > 0]
    outside_values = gray[outer > 0]
    if not len(inside_values) or not len(outside_values):
        return 0.0
    return float(np.mean(outside_values) - np.mean(inside_values))


def _group_candidates(candidates: list[dict]) -> list[list[dict]]:
    """将圆心接近的候选合并为同心圆组。"""
    groups: list[list[dict]] = []
    for candidate in sorted(candidates, key=lambda item: item["radius"]):
        attached = []
        for index, group in enumerate(groups):
            reference = max(group, key=lambda item: item.get("candidate_score", item["edge_score"]))
            distance = math.hypot(
                candidate["center_x"] - reference["center_x"],
                candidate["center_y"] - reference["center_y"],
            )
            if distance <= 0.35 * max(candidate["radius"], reference["radius"]):
                attached.append(index)
        if not attached:
            groups.append([candidate])
        else:
            first = attached[0]
            groups[first].append(candidate)
            for index in reversed(attached[1:]):
                groups[first].extend(groups.pop(index))
    return groups


def _dark_binary(gray: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """生成暗区域二值图及其可序列化参数记录。"""
    offset = _finite(params.get("dark_threshold_offset", 0.0), "dark_threshold_offset")
    morph_kernel = max(1, int(params.get("morph_kernel", 5)) | 1)
    otsu_value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    threshold_value = int(np.clip(otsu_value + offset, 0, 255))
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary, {
        "otsu_threshold": float(otsu_value),
        "dark_threshold": threshold_value,
        "morph_kernel": morph_kernel,
    }


def dark_region_mask(image_rgb: np.ndarray, params: dict) -> np.ndarray:
    """返回与实际轮廓检测一致的原图尺寸暗区诊断 mask。"""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("输入图片必须是 RGB 三通道数组。")
    x0, y0, x1, y1 = _roi_pixels(
        image_rgb.shape[:2], params.get("roi", [0.45, 0.05, 0.50, 0.90])
    )
    crop = cv2.cvtColor(image_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(crop, int(params.get("blur_kernel", 5)) | 1)
    binary, _ = _dark_binary(gray, params)
    result = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    result[y0:y1, x0:x1] = binary
    return result


def _dark_contour_candidates(
    gray: np.ndarray,
    min_radius: int,
    max_radius: int,
    params: dict,
) -> tuple[list[dict], dict]:
    """暗区域分割后，使用轮廓约束和 RANSAC 生成圆候选。"""
    binary, details = _dark_binary(gray, params)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max(1, int(params.get("max_contours", 40)))]
    candidates = []
    rejected = {"too_small": 0, "shape": 0, "ransac": 0, "outside_roi": 0, "score": 0}
    min_axis_ratio = _finite(params.get("min_axis_ratio", 0.65), "min_axis_ratio")
    min_candidate_score = _finite(params.get("min_contour_score", 0.40), "min_contour_score")
    min_inlier_ratio = _finite(
        params.get("min_ransac_inlier_ratio", 0.45), "min_ransac_inlier_ratio"
    )
    min_coverage = _finite(params.get("min_angular_coverage", 0.50), "min_angular_coverage")
    tolerance_ratio = _finite(
        params.get("ransac_tolerance_ratio", 0.03), "ransac_tolerance_ratio"
    )
    if not 0 < min_axis_ratio <= 1:
        raise ValueError("min_axis_ratio 必须位于 (0, 1]。")
    if not 0 <= min_candidate_score <= 1:
        raise ValueError("min_contour_score 必须位于 [0, 1]。")
    if not 0 <= min_inlier_ratio <= 1 or not 0 <= min_coverage <= 1:
        raise ValueError("RANSAC 内点率和角度覆盖率必须位于 [0, 1]。")
    if tolerance_ratio <= 0:
        raise ValueError("ransac_tolerance_ratio 必须大于 0。")
    minimum_area = math.pi * min_radius * min_radius * float(params.get("min_contour_area_factor", 0.20))
    for contour_index, contour in enumerate(contours):
        area = float(abs(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        if area < minimum_area or len(contour) < 6 or perimeter <= 0:
            rejected["too_small"] += 1
            continue
        ellipse = cv2.fitEllipse(contour)
        (_, _), (diameter_a, diameter_b), angle = ellipse
        major, minor = max(diameter_a, diameter_b), min(diameter_a, diameter_b)
        if major <= 0 or minor / major < min_axis_ratio:
            rejected["shape"] += 1
            continue
        fitted = _ransac_circle(
            contour, min_radius, max_radius, params,
            seed=int(params.get("ransac_seed", 2026)) + contour_index,
        )
        if fitted is None:
            rejected["ransac"] += 1
            continue
        center_x, center_y, radius = fitted["center_x"], fitted["center_y"], fitted["radius"]
        if params.get("require_circle_inside_roi", True) and not (
            center_x - radius >= 0
            and center_y - radius >= 0
            and center_x + radius < gray.shape[1]
            and center_y + radius < gray.shape[0]
        ):
            rejected["outside_roi"] += 1
            continue
        circularity = float(np.clip(4.0 * math.pi * area / (perimeter * perimeter), 0.0, 1.0))
        axis_ratio = float(minor / major)
        contrast = _dark_contrast(gray, center_x, center_y, radius)
        edge_score = _edge_score(gray, center_x, center_y, radius)
        contrast_score = float(np.clip(contrast / 50.0, 0.0, 1.0))
        edge_normalized = float(edge_score / (edge_score + 50.0)) if edge_score > 0 else 0.0
        candidate_score = (
            0.24 * fitted["ransac_inlier_ratio"]
            + 0.22 * fitted["angular_coverage"]
            + 0.16 * circularity
            + 0.14 * axis_ratio
            + 0.14 * contrast_score
            + 0.10 * edge_normalized
        )
        if candidate_score < min_candidate_score:
            rejected["score"] += 1
            continue
        candidates.append({
            "center_x": int(round(center_x)),
            "center_y": int(round(center_y)),
            "radius": int(round(radius)),
            "edge_score": edge_score,
            "candidate_score": float(candidate_score),
            "detector": "dark_contour_ransac",
            "axis_ratio": axis_ratio,
            "circularity": circularity,
            "dark_contrast": contrast,
            "ransac_inlier_ratio": fitted["ransac_inlier_ratio"],
            "angular_coverage": fitted["angular_coverage"],
            "fitted_ellipse": {
                "diameter_major": float(major),
                "diameter_minor": float(minor),
                "angle": float(angle),
            },
        })
    return candidates, {
        **details,
        "contour_count": len(contours),
        "rejected_contours": rejected,
    }


def _hough_candidates(
    gray: np.ndarray,
    min_radius: int,
    max_radius: int,
    params: dict,
) -> tuple[list[dict], int]:
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=float(params.get("dp", 1.2)),
        minDist=max(10.0, float(params.get("min_dist_ratio", 0.12)) * min(gray.shape[:2])),
        param1=float(params.get("param1", 100.0)),
        param2=float(params.get("param2", 22.0)),
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    candidates, discarded_outside_roi = [], 0
    if circles is not None:
        for center_x, center_y, radius in np.rint(circles[0]).astype(np.int32):
            if radius <= 0:
                continue
            if params.get("require_circle_inside_roi", True) and not (
                center_x - radius >= 0
                and center_y - radius >= 0
                and center_x + radius < gray.shape[1]
                and center_y + radius < gray.shape[0]
            ):
                discarded_outside_roi += 1
                continue
            edge_score = _edge_score(gray, float(center_x), float(center_y), float(radius))
            candidates.append({
                "center_x": int(center_x),
                "center_y": int(center_y),
                "radius": int(radius),
                "edge_score": edge_score,
                "candidate_score": float(edge_score / (edge_score + 50.0)) if edge_score > 0 else 0.0,
                "detector": "hough",
            })
    return candidates, discarded_outside_roi


def _sample_circle_values(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radii: np.ndarray,
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按多个半径采样圆周，返回 ``半径×角度`` 数值和有效位置。"""
    radius_grid = np.asarray(radii, dtype=np.float32)[:, None]
    x = np.rint(center_x + radius_grid * np.cos(angles)[None, :]).astype(np.int32)
    y = np.rint(center_y + radius_grid * np.sin(angles)[None, :]).astype(np.int32)
    valid = (x >= 0) & (y >= 0) & (x < image.shape[1]) & (y < image.shape[0])
    values = np.zeros(x.shape, dtype=np.float32)
    values[valid] = image[y[valid], x[valid]].astype(np.float32)
    return values, valid


def _black_ring_metrics(
    gray: np.ndarray,
    binary: np.ndarray,
    candidate: dict,
    outer: dict,
    params: dict,
) -> dict:
    """评价候选圆外侧是否存在黑色环带；不约束内外圆圆心关系。"""
    center_x = float(candidate["center_x"])
    center_y = float(candidate["center_y"])
    radius = float(candidate["radius"])
    outer_radius = float(outer["radius"])
    samples = max(72, int(params.get("radial_samples", 360)))
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False, dtype=np.float32)
    width_ratio = _finite(params.get("black_ring_width_ratio", 0.06), "black_ring_width_ratio")
    if not 0 < width_ratio < 1:
        raise ValueError("black_ring_width_ratio 必须位于 (0, 1)。")
    ring_width = max(3.0, width_ratio * outer_radius)
    start = max(1.5, float(params.get("black_ring_start_ratio", 0.02)) * radius)
    ring_radii = np.linspace(radius + start, radius + ring_width, 5, dtype=np.float32)
    ring_values, ring_valid = _sample_circle_values(
        binary, center_x, center_y, ring_radii, angles
    )
    radial_fraction = np.divide(
        np.count_nonzero((ring_values > 0) & ring_valid, axis=0),
        np.maximum(1, np.count_nonzero(ring_valid, axis=0)),
    )
    supported = radial_fraction >= float(params.get("black_ring_radial_fraction", 0.50))
    valid_angles = np.any(ring_valid, axis=0)
    black_coverage = float(np.mean(supported[valid_angles])) if valid_angles.any() else 0.0

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    edge_values, edge_valid = _sample_circle_values(
        gradient,
        center_x,
        center_y,
        np.asarray([max(1.0, radius - 2.0), radius, radius + 2.0], dtype=np.float32),
        angles,
    )
    edge_peak = np.max(np.where(edge_valid, edge_values, 0.0), axis=0)
    edge_threshold = max(
        8.0,
        float(params.get("inner_edge_threshold", np.percentile(gradient, 65))),
    )
    edge_angles = np.any(edge_valid, axis=0)
    edge_coverage = float(np.mean(edge_peak[edge_angles] >= edge_threshold)) if edge_angles.any() else 0.0

    delta = max(2.0, 0.04 * radius)
    side_values, side_valid = _sample_circle_values(
        gray,
        center_x,
        center_y,
        np.asarray([max(1.0, radius - delta), radius + delta], dtype=np.float32),
        angles,
    )
    side_ok = side_valid[0] & side_valid[1]
    boundary_contrast = (
        float(np.median(np.abs(side_values[1, side_ok] - side_values[0, side_ok])))
        if side_ok.any()
        else 0.0
    )
    edge_score = float(candidate.get("edge_score", _edge_score(gray, center_x, center_y, radius)))
    edge_normalized = edge_score / (edge_score + 50.0) if edge_score > 0 else 0.0
    fit_coverage = float(candidate.get("angular_coverage", edge_coverage))
    contrast_score = float(np.clip(boundary_contrast / 40.0, 0.0, 1.0))
    score = (
        0.50 * black_coverage
        + 0.25 * edge_coverage
        + 0.10 * contrast_score
        + 0.10 * fit_coverage
        + 0.05 * edge_normalized
    )
    return {
        "black_ring_coverage": black_coverage,
        "edge_coverage": edge_coverage,
        "boundary_contrast": boundary_contrast,
        "ring_width": ring_width,
        "candidate_score": float(score),
        "edge_score": edge_score,
    }


def _deduplicate_circles(candidates: list[dict]) -> list[dict]:
    """合并不同候选生成器产生的近似重复圆。"""
    unique: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item.get("candidate_score", 0.0), reverse=True):
        duplicate = False
        for kept in unique:
            radius_scale = max(4.0, float(candidate["radius"]), float(kept["radius"]))
            center_distance = math.hypot(
                float(candidate["center_x"]) - float(kept["center_x"]),
                float(candidate["center_y"]) - float(kept["center_y"]),
            )
            radius_distance = abs(float(candidate["radius"]) - float(kept["radius"]))
            if center_distance <= 0.08 * radius_scale and radius_distance <= 0.12 * radius_scale:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique


def _outer_inner_ring_candidates(
    gray: np.ndarray,
    min_radius: int,
    max_radius: int,
    params: dict,
) -> tuple[list[dict], dict]:
    """先找外圆，再依靠黑色环带寻找允许大幅偏心的内圆。"""
    outer_params = {
        **params,
        "min_contour_score": float(params.get("outer_min_contour_score", 0.30)),
        "min_ransac_inlier_ratio": float(params.get("outer_min_ransac_inlier_ratio", 0.35)),
        "min_angular_coverage": float(params.get("outer_min_angular_coverage", 0.45)),
    }
    outer_contours, contour_details = _dark_contour_candidates(
        gray, min_radius, max_radius, outer_params
    )
    outer_hough, discarded = _hough_candidates(gray, min_radius, max_radius, params)
    outer_pool = _deduplicate_circles(outer_contours + outer_hough)
    if not outer_pool:
        raise ValueError("ROI 内未检测到外侧大圆。")
    largest_outer_radius = max(float(item["radius"]) for item in outer_pool)
    outer = max(
        outer_pool,
        key=lambda item: (
            float(item.get("candidate_score", 0.0))
            + 0.35 * float(item["radius"]) / largest_outer_radius
        ),
    ).copy()
    outer["role"] = "outer_search_boundary"

    inner_min_ratio = _finite(
        params.get("inner_radius_min_ratio", 0.15), "inner_radius_min_ratio"
    )
    inner_max_ratio = _finite(
        params.get("inner_radius_max_ratio", 0.75), "inner_radius_max_ratio"
    )
    if not 0 < inner_min_ratio < inner_max_ratio < 1:
        raise ValueError("内圆半径比例必须满足 0 < min < max < 1。")
    inner_min = max(2, int(round(inner_min_ratio * float(outer["radius"]))))
    inner_max = max(inner_min + 1, int(round(inner_max_ratio * float(outer["radius"]))))

    binary, binary_details = _dark_binary(gray, params)
    inner_params = {
        **params,
        "min_ransac_inlier_ratio": float(params.get("min_inner_ransac_inlier_ratio", 0.25)),
        "min_angular_coverage": float(params.get("min_inner_angular_coverage", 0.35)),
        "ransac_tolerance_ratio": float(params.get("inner_ransac_tolerance_ratio", 0.04)),
    }
    contour_sets: list[tuple[str, list[np.ndarray]]] = []
    dark_contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_sets.append(("dark_inner_boundary", dark_contours))
    high = float(params.get("param1", 100.0))
    edges = cv2.Canny(gray, max(1.0, high * 0.45), high)
    edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_sets.append(("edge_arc", edge_contours))

    raw_candidates: list[dict] = []
    max_contours = max(10, int(params.get("max_inner_contours", 80)))
    seed = int(params.get("ransac_seed", 2026)) + 10000
    for source, contours in contour_sets:
        for index, contour in enumerate(
            sorted(contours, key=lambda item: cv2.arcLength(item, True), reverse=True)[:max_contours]
        ):
            if len(contour) < 6:
                continue
            fitted = _ransac_circle(
                contour, inner_min, inner_max, inner_params, seed + index
            )
            if fitted is None:
                continue
            raw_candidates.append({
                **fitted,
                "center_x": int(round(fitted["center_x"])),
                "center_y": int(round(fitted["center_y"])),
                "radius": int(round(fitted["radius"])),
                "detector": source,
            })
        seed += max_contours
    hough_candidates, _ = _hough_candidates(
        gray, inner_min, inner_max, {**params, "require_circle_inside_roi": False}
    )
    raw_candidates.extend(hough_candidates)

    accepted: list[dict] = []
    rejected = {"outside_outer": 0, "black_ring": 0, "arc_coverage": 0}
    containment_margin = max(0.0, float(params.get("inner_containment_margin_ratio", 0.01))) * float(outer["radius"])
    min_black_coverage = _finite(
        params.get("min_black_ring_coverage", 0.45), "min_black_ring_coverage"
    )
    min_arc_coverage = _finite(
        params.get("min_inner_angular_coverage", 0.35), "min_inner_angular_coverage"
    )
    if not 0 <= min_black_coverage <= 1 or not 0 <= min_arc_coverage <= 1:
        raise ValueError("黑环覆盖率和内圆圆弧覆盖率必须位于 [0, 1]。")
    for candidate in raw_candidates:
        center_distance = math.hypot(
            float(candidate["center_x"]) - float(outer["center_x"]),
            float(candidate["center_y"]) - float(outer["center_y"]),
        )
        if center_distance + float(candidate["radius"]) + containment_margin >= float(outer["radius"]):
            rejected["outside_outer"] += 1
            continue
        metrics = _black_ring_metrics(gray, binary, candidate, outer, params)
        candidate = {**candidate, **metrics, "role": "inner_mask_circle"}
        if candidate["black_ring_coverage"] < min_black_coverage:
            rejected["black_ring"] += 1
            continue
        if max(candidate.get("angular_coverage", 0.0), candidate["edge_coverage"]) < min_arc_coverage:
            rejected["arc_coverage"] += 1
            continue
        accepted.append(candidate)
    accepted = _deduplicate_circles(accepted)
    accepted.sort(key=lambda item: item["candidate_score"], reverse=True)
    if not accepted:
        raise ValueError("已检测到外圆，但没有找到具有足够黑色环带支持的内圆。")
    return accepted, {
        **binary_details,
        "outer_circle": outer,
        "outer_candidate_count": len(outer_pool),
        "inner_raw_candidate_count": len(raw_candidates),
        "inner_rejected": rejected,
        "inner_radius_range": [inner_min, inner_max],
        "discarded_outside_roi": discarded,
        "outer_contour_details": contour_details,
    }


def detect_candidates(image_rgb: np.ndarray, params: dict) -> dict:
    """返回 ROI 和候选圆；坐标均为原图坐标。"""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("输入图片必须是 RGB 三通道数组。")
    if params.get("enabled", True) is False:
        return {"enabled": False, "candidates": [], "groups": [], "roi": None}
    roi = params.get("roi", [0.45, 0.05, 0.50, 0.90])
    x0, y0, x1, y1 = _roi_pixels(image_rgb.shape[:2], roi)
    crop = cv2.cvtColor(image_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(crop, int(params.get("blur_kernel", 5)) | 1)
    min_dim = min(gray.shape[:2])
    min_radius = max(2, int(round(float(params.get("min_radius_ratio", 0.05)) * min_dim)))
    max_radius = max(min_radius + 1, int(round(float(params.get("max_radius_ratio", 0.80)) * min_dim)))
    method = str(params.get("detection_method", "hybrid")).strip().lower()
    if method not in {"hybrid", "dark_contour", "hough", "outer_inner_ring"}:
        raise ValueError(
            "detection_method 只能是 hybrid、dark_contour、hough 或 outer_inner_ring。"
        )
    contour_details: dict = {}
    discarded_outside_roi = 0
    candidates: list[dict] = []
    detector_used = method
    outer_circle = None
    if method == "outer_inner_ring":
        candidates, contour_details = _outer_inner_ring_candidates(
            gray, min_radius, max_radius, params
        )
        outer_circle = contour_details.get("outer_circle")
        detector_used = "outer_inner_black_ring"
    elif method in {"hybrid", "dark_contour"}:
        candidates, contour_details = _dark_contour_candidates(
            gray, min_radius, max_radius, params
        )
        detector_used = "dark_contour_ransac"
    hough_fallback_used = False
    if method == "hough" or (method == "hybrid" and not candidates):
        candidates, discarded_outside_roi = _hough_candidates(
            gray, min_radius, max_radius, params
        )
        detector_used = "hough"
        hough_fallback_used = method == "hybrid"
    # 内部检测均使用 ROI 局部坐标，返回结果统一转换成原图坐标。
    for candidate in candidates:
        candidate["center_x"] = int(candidate["center_x"] + x0)
        candidate["center_y"] = int(candidate["center_y"] + y0)
    if outer_circle is not None:
        outer_circle["center_x"] = int(outer_circle["center_x"] + x0)
        outer_circle["center_y"] = int(outer_circle["center_y"] + y0)
        groups = [[candidate] for candidate in candidates]
    else:
        groups = _group_candidates(candidates)
    return {
        "enabled": True,
        "roi": [x0, y0, x1 - x0, y1 - y0],
        "candidates": candidates,
        "groups": groups,
        "requested_method": method,
        "detector_used": detector_used,
        "hough_fallback_used": hough_fallback_used,
        "contour_details": contour_details,
        "discarded_outside_roi": discarded_outside_roi,
        "min_radius": min_radius,
        "max_radius": max_radius,
        "outer_circle": outer_circle,
    }


def select_circle(detection: dict, params: dict) -> dict:
    if not detection.get("enabled", True):
        return {"enabled": False}
    groups = detection.get("groups", [])
    if not groups:
        raise ValueError("ROI 内未检测到圆候选。")
    if detection.get("requested_method") == "outer_inner_ring":
        chosen = max(
            detection.get("candidates", []),
            key=lambda item: item.get("candidate_score", 0.0),
        )
        return {
            **chosen,
            "circle_target": "inner_black_ring",
            "group_target": "black_ring_score",
            "group_size": 1,
            "candidate_count": len(detection.get("candidates", [])),
            "roi": detection["roi"],
        }
    group_mode = params.get("group_target", "largest")
    if group_mode == "largest":
        group = max(groups, key=lambda items: max(item["radius"] for item in items))
    elif group_mode == "strongest":
        group = max(groups, key=lambda items: max(item["edge_score"] for item in items))
    elif group_mode == "best_score":
        group = max(groups, key=lambda items: max(item.get("candidate_score", 0.0) for item in items))
    else:
        raise ValueError("group_target 只能是 largest、strongest 或 best_score。")
    target = params.get("circle_target", "outer")
    if target == "inner":
        chosen = min(group, key=lambda item: (item["radius"], -item["edge_score"]))
    elif target == "outer":
        chosen = max(group, key=lambda item: (item["radius"], item["edge_score"]))
    elif target == "best_contrast":
        chosen = max(group, key=lambda item: item["edge_score"])
    else:
        raise ValueError("circle_target 只能是 inner、outer 或 best_contrast。")
    return {
        **chosen,
        "circle_target": target,
        "group_target": group_mode,
        "group_size": len(group),
        "candidate_count": len(detection.get("candidates", [])),
        "roi": detection["roi"],
    }


def detect_circle(image_rgb: np.ndarray, params: dict) -> dict:
    started = time.perf_counter()
    detection = detect_candidates(image_rgb, params)
    selected = select_circle(detection, params) if detection.get("enabled", True) else {"enabled": False}
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {"detection": detection, "selected": selected, "detection_ms": elapsed_ms}


def make_mask(shape: tuple[int, int], selected: dict, params: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not selected.get("enabled", True):
        return mask
    radius = effective_mask_radius(selected, params)
    cv2.circle(mask, (selected["center_x"], selected["center_y"]), radius, 255, thickness=-1)
    return mask


def load_mask_image(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """读取默认二值 mask，并按原图尺寸最近邻缩放；非零区域为忽略区。"""
    data = np.fromfile(path, dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取默认 mask 图片：{path}")
    height, width = shape
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def mask_from_circle_record(shape: tuple[int, int], circle: dict) -> np.ndarray:
    """根据已保存的动态圆结果或默认 mask 记录生成原图尺寸 mask。"""
    params = circle.get("params", {})
    result = circle.get("result", {})
    selected = result.get("selected", {})
    if selected.get("enabled", True):
        return make_mask(shape, selected, params)
    fallback = circle.get("fallback_mask") or params.get("fallback_mask") or params.get("default_mask")
    if fallback:
        return load_mask_image(Path(fallback), shape)
    return np.zeros(shape, dtype=np.uint8)


def detect_circle_or_fallback(image_rgb: np.ndarray, params: dict, *, base_dir: Path | None = None) -> dict:
    """检测圆；失败时使用配置的 fallback_mask/default_mask。"""
    started = time.perf_counter()
    try:
        return detect_circle(image_rgb, params)
    except ValueError as error:
        fallback = params.get("fallback_mask") or params.get("default_mask")
        if not fallback:
            raise
        fallback_path = Path(fallback)
        if not fallback_path.is_absolute() and base_dir is not None:
            fallback_path = (base_dir / fallback_path).resolve()
        # 预先读取并校验，避免训练开始后才发现默认 mask 无法使用。
        load_mask_image(fallback_path, image_rgb.shape[:2])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "detection": {
                "enabled": False,
                "fallback_used": True,
                "error": str(error),
                "fallback_mask": str(fallback_path),
                "candidates": [],
                "groups": [],
                "roi": None,
            },
            "selected": {"enabled": False},
            "fallback_mask": str(fallback_path),
            "detection_ms": elapsed_ms,
        }


def default_mask_record(image_rgb: np.ndarray, params: dict, *, base_dir: Path | None = None) -> dict:
    """为一张图片建立“始终使用默认 mask”的记录，并预先校验 mask。"""
    fallback = params.get("default_mask") or params.get("fallback_mask")
    if not fallback:
        raise ValueError("未配置 default_mask；当前模式要求所有图片使用默认 mask。")
    mask_path = Path(fallback)
    if not mask_path.is_absolute() and base_dir is not None:
        mask_path = (base_dir / mask_path).resolve()
    load_mask_image(mask_path, image_rgb.shape[:2])
    return {
        "params": {**params, "default_mask": str(mask_path)},
        "result": {
            "detection": {
                "enabled": False,
                "fallback_used": True,
                "mode": "default_mask",
                "fallback_mask": str(mask_path),
                "candidates": [],
                "groups": [],
                "roi": None,
            },
            "selected": {"enabled": False},
            "fallback_mask": str(mask_path),
            "detection_ms": 0.0,
        },
        "fallback_mask": str(mask_path),
    }


def effective_mask_radius(selected: dict, params: dict) -> int:
    return max(1, int(round(selected["radius"] * float(params.get("mask_radius_scale", 1.0)))) + int(params.get("mask_margin", 2)))


def preprocess_record(path: Path, params: dict, image_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """读取原图并返回未填充的网络输入和对应 ignore mask。"""
    image_rgb = read_rgb(path)
    result = detect_circle(image_rgb, params)
    selected = result["selected"]
    mask = make_mask(image_rgb.shape[:2], selected, params)
    image = np.asarray(Image.fromarray(image_rgb).resize((image_size, image_size), Image.Resampling.BILINEAR))
    resized_mask = np.asarray(Image.fromarray(mask).resize((image_size, image_size), Image.Resampling.NEAREST)) > 0
    metadata = {"params": params, **selected, "mask_radius": effective_mask_radius(selected, params) if selected.get("enabled", True) else 0}
    return image, resized_mask, {"result": result, "selected": metadata}


def mask_score(anomaly_map, ignore_mask):
    """仅在非 mask 区域取最大值；ignore_mask 为 True 的位置不参与。"""
    import torch

    ignore_mask = expanded_ignore_mask(anomaly_map, ignore_mask)
    return anomaly_map.masked_fill(ignore_mask, -torch.inf).amax(dim=(-2, -1))


def expanded_ignore_mask(anomaly_map, ignore_mask):
    """将任意常用形状的 ignore mask 转成与异常图一致的 bool Tensor。"""
    import torch

    if ignore_mask is None:
        return torch.zeros_like(anomaly_map, dtype=torch.bool)
    if not torch.is_tensor(ignore_mask):
        ignore_mask = torch.as_tensor(ignore_mask, device=anomaly_map.device)
    else:
        ignore_mask = ignore_mask.to(anomaly_map.device)
    ignore_mask = ignore_mask.bool()
    if ignore_mask.ndim > anomaly_map.ndim:
        raise ValueError(
            f"ignore_mask 维数 {ignore_mask.ndim} 不能超过异常图维数 "
            f"{anomaly_map.ndim}。"
        )
    # EfficientAD 异常图为 B×C×H×W。常见 mask 分别是 H×W、B×H×W
    # 和 B×1×H×W：二维 mask 应在前面补 batch/channel 维，三维 mask
    # 应只在 batch 后插入 channel 维。不能统一 unsqueeze(1)，否则 H×W
    # 会错误地变成 H×1×1×W。
    if ignore_mask.ndim == 2 and anomaly_map.ndim >= 2:
        while ignore_mask.ndim < anomaly_map.ndim:
            ignore_mask = ignore_mask.unsqueeze(0)
    elif ignore_mask.ndim == anomaly_map.ndim - 1 and anomaly_map.ndim >= 3:
        ignore_mask = ignore_mask.unsqueeze(1)
    else:
        while ignore_mask.ndim < anomaly_map.ndim:
            ignore_mask = ignore_mask.unsqueeze(0)
    try:
        return ignore_mask.expand_as(anomaly_map)
    except RuntimeError as error:
        raise ValueError(
            f"ignore_mask 形状 {tuple(ignore_mask.shape)} 无法匹配异常图 {tuple(anomaly_map.shape)}。"
        ) from error


def mask_aware_average_pool(anomaly_map, ignore_mask, *, pool_kernel: int = 21):
    """执行 mask-aware 局部平均池化并返回 ``(pooled_map, valid_mask)``。

    使用有效像素计数修正 mask 边界处的池化，避免圆 mask 内的异常值泄漏到
    外部，也避免简单填零造成边界均值被人为压低。Score 与定位必须复用本函数。
    """
    import torch.nn.functional as F

    pool_kernel = int(pool_kernel)
    if pool_kernel < 1 or pool_kernel % 2 == 0:
        raise ValueError("pool_kernel 必须是正奇数。")
    ignore_mask = expanded_ignore_mask(anomaly_map, ignore_mask)
    valid = ~ignore_mask
    valid_float = valid.to(anomaly_map.dtype)
    padding = pool_kernel // 2
    pooled_sum = F.avg_pool2d(
        anomaly_map * valid_float,
        kernel_size=pool_kernel,
        stride=1,
        padding=padding,
        divisor_override=1,
    )
    pooled_count = F.avg_pool2d(
        valid_float,
        kernel_size=pool_kernel,
        stride=1,
        padding=padding,
        divisor_override=1,
    )
    smoothed = pooled_sum / pooled_count.clamp_min(1)
    # mask 中心位置本身始终无效；仅在其窗口内存在有效像素并不能使其参与 Score。
    smoothed = smoothed.masked_fill(~valid, 0)
    return smoothed, valid


def pooled_topk_score(anomaly_map, ignore_mask, *, pool_kernel: int = 21, topk_ratio: float = 0.001):
    """有效区域 mask-aware 局部均值池化后，计算最高一部分位置的均值。"""
    import torch

    topk_ratio = float(topk_ratio)
    if not 0 < topk_ratio <= 1:
        raise ValueError("topk_ratio 必须位于 (0, 1]。")
    smoothed, valid = mask_aware_average_pool(
        anomaly_map, ignore_mask, pool_kernel=pool_kernel
    )
    scores = []
    for index in range(anomaly_map.shape[0]):
        values = smoothed[index][valid[index]]
        if values.numel() == 0:
            raise ValueError("mask 后没有可用于计算整图分数的有效像素。")
        count = max(1, int(math.ceil(values.numel() * topk_ratio)))
        scores.append(torch.topk(values, count, largest=True, sorted=False).values.mean())
    return torch.stack(scores)


def multiscale_topk_scores(
    anomaly_map,
    ignore_mask,
    *,
    pool_kernels: tuple[int, ...] | list[int] = (1, 7, 21),
    topk_ratio: float = 0.001,
):
    """计算多个池化尺度的 Top-K 分数，返回 ``{kernel: tensor}``。"""
    kernels = tuple(int(kernel) for kernel in pool_kernels)
    if not kernels:
        raise ValueError("pool_kernels 不能为空。")
    if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
        raise ValueError("pool_kernels 必须全部是正奇数。")
    return {
        str(kernel): pooled_topk_score(
            anomaly_map, ignore_mask, pool_kernel=kernel, topk_ratio=topk_ratio
        )
        for kernel in kernels
    }


def overlay_diagnostics(image_rgb: np.ndarray, result: dict, params: dict) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for index, candidate in enumerate(result["detection"].get("candidates", [])):
        color = (180, 180, 180)
        cv2.circle(canvas, (candidate["center_x"], candidate["center_y"]), candidate["radius"], color, 1)
    selected = result["selected"]
    if selected.get("enabled", True):
        outer = result["detection"].get("outer_circle")
        if outer is not None:
            cv2.circle(
                canvas,
                (outer["center_x"], outer["center_y"]),
                outer["radius"],
                (255, 255, 0),
                2,
            )
            cv2.drawMarker(
                canvas,
                (outer["center_x"], outer["center_y"]),
                (255, 255, 0),
                cv2.MARKER_CROSS,
                10,
                1,
            )
        mask = make_mask(image_rgb.shape[:2], selected, params)
        mask_radius = effective_mask_radius(selected, params)
        cv2.circle(canvas, (selected["center_x"], selected["center_y"]), selected["radius"], (0, 255, 0), 2)
        cv2.circle(canvas, (selected["center_x"], selected["center_y"]), mask_radius, (0, 0, 255), 2)
        cv2.drawMarker(canvas, (selected["center_x"], selected["center_y"]), (255, 0, 0), cv2.MARKER_CROSS, 14, 2)
        x, y, w, h = result["detection"]["roi"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 128, 0), 2)
        cv2.putText(canvas, f"target={selected['circle_target']} r={selected['radius']} mask_r={mask_radius}",
                    (max(5, selected["center_x"] - selected["radius"]), max(20, selected["center_y"] - selected["radius"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        if "black_ring_coverage" in selected:
            cv2.putText(
                canvas,
                f"black_ring={selected['black_ring_coverage']:.3f} edge={selected['edge_coverage']:.3f}",
                (
                    max(5, selected["center_x"] - selected["radius"]),
                    min(image_rgb.shape[0] - 8, selected["center_y"] + selected["radius"] + 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
