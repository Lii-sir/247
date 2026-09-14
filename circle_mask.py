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


def detect_candidates(image_rgb: np.ndarray, params: dict) -> dict:
    """返回 ROI、候选圆和同心圆分组；坐标均为原图坐标。"""
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
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=float(params.get("dp", 1.2)),
        minDist=max(10.0, float(params.get("min_dist_ratio", 0.12)) * min_dim),
        param1=float(params.get("param1", 100.0)),
        param2=float(params.get("param2", 22.0)),
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    candidates = []
    discarded_outside_roi = 0
    if circles is not None:
        for cx, cy, radius in np.rint(circles[0]).astype(np.int32):
            if radius <= 0:
                continue
            # ROI 是候选圆的完整搜索区域：圆的外接框必须完全落在 ROI 内，
            # 不能只检查圆心，否则 ROI 边缘的圆弧也可能被当成目标圆。
            if params.get("require_circle_inside_roi", True) and not (
                cx - radius >= 0
                and cy - radius >= 0
                and cx + radius < gray.shape[1]
                and cy + radius < gray.shape[0]
            ):
                discarded_outside_roi += 1
                continue
            candidates.append({
                "center_x": int(cx + x0),
                "center_y": int(cy + y0),
                "radius": int(radius),
                "edge_score": _edge_score(gray, float(cx), float(cy), float(radius)),
            })
    # Union candidates with nearly identical centers into concentric groups.
    groups: list[list[dict]] = []
    for candidate in sorted(candidates, key=lambda item: item["radius"]):
        attached = []
        for index, group in enumerate(groups):
            reference = max(group, key=lambda item: item["edge_score"])
            distance = math.hypot(candidate["center_x"] - reference["center_x"], candidate["center_y"] - reference["center_y"])
            # Hough 的同心候选圆心会因纹理和遮挡产生偏移，不能只用很小的
            # minDist；这里按较大圆半径给出一定容差，再在组内选择 inner/outer。
            if distance <= 0.35 * max(candidate["radius"], reference["radius"]):
                attached.append(index)
        if not attached:
            groups.append([candidate])
        else:
            first = attached[0]
            groups[first].append(candidate)
            for index in reversed(attached[1:]):
                groups[first].extend(groups.pop(index))
    return {
        "enabled": True,
        "roi": [x0, y0, x1 - x0, y1 - y0],
        "candidates": candidates,
        "groups": groups,
        "discarded_outside_roi": discarded_outside_roi,
        "min_radius": min_radius,
        "max_radius": max_radius,
    }


def select_circle(detection: dict, params: dict) -> dict:
    if not detection.get("enabled", True):
        return {"enabled": False}
    groups = detection.get("groups", [])
    if not groups:
        raise ValueError("ROI 内未检测到圆候选。")
    group_mode = params.get("group_target", "largest")
    if group_mode == "largest":
        group = max(groups, key=lambda items: max(item["radius"] for item in items))
    elif group_mode == "strongest":
        group = max(groups, key=lambda items: max(item["edge_score"] for item in items))
    else:
        raise ValueError("group_target 只能是 largest 或 strongest。")
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

    if not torch.is_tensor(ignore_mask):
        ignore_mask = torch.as_tensor(ignore_mask, device=anomaly_map.device)
    else:
        ignore_mask = ignore_mask.to(anomaly_map.device)
    ignore_mask = ignore_mask.bool()
    while ignore_mask.ndim < anomaly_map.ndim:
        ignore_mask = ignore_mask.unsqueeze(1)
    ignore_mask = ignore_mask.expand_as(anomaly_map)
    return anomaly_map.masked_fill(ignore_mask, -torch.inf).amax(dim=(-2, -1))


def pooled_topk_score(anomaly_map, ignore_mask, *, pool_kernel: int = 21, topk_ratio: float = 0.001):
    """有效区域局部均值池化后，计算最高一部分位置的均值。

    使用有效像素计数修正 mask 边界处的池化，避免圆 mask 内的异常值泄漏到
    外部，也避免简单填零造成边界均值被人为压低。
    """
    import torch
    import torch.nn.functional as F

    pool_kernel = int(pool_kernel)
    topk_ratio = float(topk_ratio)
    if pool_kernel < 1 or pool_kernel % 2 == 0:
        raise ValueError("pool_kernel 必须是正奇数。")
    if not 0 < topk_ratio <= 1:
        raise ValueError("topk_ratio 必须位于 (0, 1]。")
    if not torch.is_tensor(ignore_mask):
        ignore_mask = torch.as_tensor(ignore_mask, device=anomaly_map.device)
    else:
        ignore_mask = ignore_mask.to(anomaly_map.device)
    ignore_mask = ignore_mask.bool()
    while ignore_mask.ndim < anomaly_map.ndim:
        ignore_mask = ignore_mask.unsqueeze(1)
    ignore_mask = ignore_mask.expand_as(anomaly_map)
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
    scores = []
    for index in range(anomaly_map.shape[0]):
        values = smoothed[index][valid[index] & (pooled_count[index] > 0)]
        if values.numel() == 0:
            raise ValueError("mask 后没有可用于计算整图分数的有效像素。")
        count = max(1, int(math.ceil(values.numel() * topk_ratio)))
        scores.append(torch.topk(values, count, largest=True, sorted=False).values.mean())
    return torch.stack(scores)


def overlay_diagnostics(image_rgb: np.ndarray, result: dict, params: dict) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for index, candidate in enumerate(result["detection"].get("candidates", [])):
        color = (180, 180, 180)
        cv2.circle(canvas, (candidate["center_x"], candidate["center_y"]), candidate["radius"], color, 1)
    selected = result["selected"]
    if selected.get("enabled", True):
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
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
