"""由整图 Score 定义生成一致的空间响应、连通域和异常框。"""

from __future__ import annotations

import math
from numbers import Real
import cv2
import numpy as np

from circle_mask import expanded_ignore_mask, mask_aware_average_pool


DEFAULT_LOCALIZATION_PARAMS = {
    # 以有效异常图面积为基准，768×768 时约保留 12 像素以上的区域。
    "min_area_ratio": 0.00002,
    "morph_kernel": 3,
    "open_iterations": 1,
    "close_iterations": 1,
    "merge_iou": 0.15,
    # 小尺度框被大尺度框包含时，IoU 可能很低；用较小框的覆盖率识别同一缺陷。
    "merge_containment": 0.80,
    "merge_distance_ratio": 0.005,
    "padding_ratio": 0.003,
    "fallback_size_ratio": 0.02,
    # 多尺度融合响应已经无量纲，使用独立于原始异常图的固定显示下限。
    "normalized_display_max": 3.0,
}


def normalize_localization_params(overrides: dict | None = None) -> dict:
    """合并并校验定位参数，供训练、评估和单图预测共同使用。"""
    params = {**DEFAULT_LOCALIZATION_PARAMS, **(overrides or {})}
    float_ranges = {
        "min_area_ratio": (0.0, 1.0, True),
        "merge_iou": (0.0, 1.0, True),
        "merge_containment": (0.0, 1.0, True),
        "merge_distance_ratio": (0.0, 1.0, True),
        "padding_ratio": (0.0, 0.5, True),
        "fallback_size_ratio": (0.0, 1.0, False),
        "normalized_display_max": (0.0, math.inf, False),
    }
    for name, (lower, upper, lower_inclusive) in float_ranges.items():
        value = params.get(name)
        if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} 必须是有限数值。")
        value = float(value)
        lower_ok = value >= lower if lower_inclusive else value > lower
        if not math.isfinite(value) or not lower_ok or value > upper:
            bracket = "[" if lower_inclusive else "("
            raise ValueError(f"{name} 必须位于 {bracket}{lower}, {upper}]。")
        params[name] = value

    morph_kernel = params.get("morph_kernel")
    if not isinstance(morph_kernel, int) or isinstance(morph_kernel, bool):
        raise ValueError("morph_kernel 必须是正奇数。")
    if morph_kernel < 1 or morph_kernel % 2 == 0:
        raise ValueError("morph_kernel 必须是正奇数。")
    params["morph_kernel"] = morph_kernel
    for name in ("open_iterations", "close_iterations"):
        value = params.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} 必须是非负整数。")
    return params


def _single_map(anomaly_map):
    """规范化为单张、单通道异常图；当前可视化一次只处理一张图。"""
    import torch

    if not torch.is_tensor(anomaly_map):
        anomaly_map = torch.as_tensor(anomaly_map)
    if anomaly_map.ndim == 2:
        anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)
    elif anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.unsqueeze(0)
    if anomaly_map.ndim != 4 or anomaly_map.shape[0] != 1 or anomaly_map.shape[1] != 1:
        raise ValueError(
            "定位要求 anomaly_map 为单张单通道的 H×W、1×H×W 或 1×1×H×W。"
        )
    if not torch.isfinite(anomaly_map).all():
        raise ValueError("定位异常图不能包含 NaN 或 Inf。")
    return anomaly_map


def _numpy_map(value) -> np.ndarray:
    # 必须复制：CPU Tensor 的 .numpy() 与原 Tensor 共享内存，定位阶段应用 mask
    # 不应反向修改 prediction.anomaly_map。
    return value.detach().cpu().numpy().squeeze().astype(np.float32, copy=True)


def _clean_binary(binary: np.ndarray, valid: np.ndarray, params: dict) -> np.ndarray:
    """执行形态学去噪/连接，之后重新应用有效区以防跨越圆形 mask。"""
    result = np.asarray(binary, dtype=np.uint8) * 255
    kernel_size = params["morph_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if params["open_iterations"]:
        result = cv2.morphologyEx(
            result, cv2.MORPH_OPEN, kernel, iterations=params["open_iterations"]
        )
    if params["close_iterations"]:
        result = cv2.morphologyEx(
            result, cv2.MORPH_CLOSE, kernel, iterations=params["close_iterations"]
        )
    return (result > 0) & valid


def _component_boxes(
    binary: np.ndarray,
    response: np.ndarray,
    valid: np.ndarray,
    params: dict,
    *,
    source_scale: int,
) -> list[dict]:
    """从处理后的二值图提取外接框，坐标采用异常图空间的半开区间。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    min_area = max(1, int(math.ceil(int(valid.sum()) * params["min_area_ratio"])))
    height, width = binary.shape
    padding = int(round(max(height, width) * params["padding_ratio"]))
    boxes = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        w = int(stats[component, cv2.CC_STAT_WIDTH])
        h = int(stats[component, cv2.CC_STAT_HEIGHT])
        component_values = response[labels == component]
        boxes.append({
            "x0": max(0, x - padding),
            "y0": max(0, y - padding),
            "x1": min(width, x + w + padding),
            "y1": min(height, y + h + padding),
            "area": area,
            "peak": float(component_values.max()),
            "mean": float(component_values.mean()),
            "source_scales": [int(source_scale)],
        })
    return boxes


def _intersection_and_areas(first: dict, second: dict) -> tuple[int, int, int]:
    """返回交集面积及两个框面积，供 IoU 和包含率复用。"""
    x0, y0 = max(first["x0"], second["x0"]), max(first["y0"], second["y0"])
    x1, y1 = min(first["x1"], second["x1"]), min(first["y1"], second["y1"])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    first_area = (first["x1"] - first["x0"]) * (first["y1"] - first["y0"])
    second_area = (second["x1"] - second["x0"]) * (second["y1"] - second["y0"])
    return intersection, first_area, second_area


def _box_gap(first: dict, second: dict) -> float:
    horizontal = max(0, max(first["x0"], second["x0"]) - min(first["x1"], second["x1"]))
    vertical = max(0, max(first["y0"], second["y0"]) - min(first["y1"], second["y1"]))
    return math.hypot(horizontal, vertical)


def _merge_two_boxes(first: dict, second: dict) -> dict:
    total_area = first["area"] + second["area"]
    return {
        "x0": min(first["x0"], second["x0"]),
        "y0": min(first["y0"], second["y0"]),
        "x1": max(first["x1"], second["x1"]),
        "y1": max(first["y1"], second["y1"]),
        "area": total_area,
        "peak": max(first["peak"], second["peak"]),
        "mean": (
            first["mean"] * first["area"] + second["mean"] * second["area"]
        ) / total_area,
        "source_scales": sorted(set(first["source_scales"] + second["source_scales"])),
    }


def merge_boxes(boxes: list[dict], shape: tuple[int, int], params: dict) -> list[dict]:
    """迭代合并重叠或相邻框，适用于单尺度碎片和多尺度重复框。"""
    merged = [dict(box) for box in boxes]
    max_gap = math.hypot(*shape) * params["merge_distance_ratio"]
    changed = True
    while changed:
        changed = False
        for first_index in range(len(merged)):
            for second_index in range(first_index + 1, len(merged)):
                first, second = merged[first_index], merged[second_index]
                intersection, first_area, second_area = _intersection_and_areas(first, second)
                union = first_area + second_area - intersection
                overlap = intersection / union if union else 0.0
                containment = (
                    intersection / min(first_area, second_area)
                    if min(first_area, second_area) > 0 else 0.0
                )
                overlap_match = params["merge_iou"] > 0 and overlap >= params["merge_iou"]
                containment_match = (
                    params["merge_containment"] > 0
                    and containment >= params["merge_containment"]
                )
                # 距离规则只负责没有面积重叠的临近/接触框；低 IoU 重叠框仍由
                # IoU/包含率控制，避免距离条件令重叠参数失效。
                distance_match = overlap == 0 and _box_gap(first, second) <= max_gap
                if overlap_match or containment_match or distance_match:
                    merged[first_index] = _merge_two_boxes(first, second)
                    merged.pop(second_index)
                    changed = True
                    break
            if changed:
                break
    return sorted(merged, key=lambda box: (-box["peak"], box["y0"], box["x0"]))


def _fallback_box(response: np.ndarray, valid: np.ndarray, params: dict, scales: list[int]) -> dict:
    """整图为异常但后处理无框时，以最强有效响应生成一个最小兜底框。"""
    values = np.where(valid, response, -np.inf)
    if not np.isfinite(values).any():
        raise ValueError("mask 后没有可用于定位的有效像素。")
    valid_values = values[valid]
    if np.all(valid_values == valid_values[0]):
        # 完全平坦的响应没有“最强位置”。np.argmax 会固定返回第一个有效点，
        # 往往造成左上角伪定位；此时选择最接近图像中心的有效像素更中性。
        coordinates = np.argwhere(valid)
        center = (np.asarray(values.shape, dtype=np.float64) - 1.0) / 2.0
        distance = np.square(coordinates - center).sum(axis=1)
        y_value, x_value = coordinates[int(np.argmin(distance))]
        fallback_position = "valid_region_center"
    else:
        y_value, x_value = np.unravel_index(int(np.argmax(values)), values.shape)
        fallback_position = "strongest_response"
    y, x = int(y_value), int(x_value)
    height, width = values.shape
    size = max(3, int(math.ceil(min(height, width) * params["fallback_size_ratio"])))
    half = size // 2
    return {
        "x0": max(0, x - half), "y0": max(0, y - half),
        "x1": min(width, x - half + size), "y1": min(height, y - half + size),
        "area": 1, "peak": float(values[y, x]), "mean": float(values[y, x]),
        "source_scales": sorted(set(int(scale) for scale in scales)),
        "fallback": True,
        "fallback_position": fallback_position,
    }


def _scale_payload(
    pooled_map,
    valid: np.ndarray,
    *,
    kernel: int,
    raw_score: float,
    normalized_score: float | None,
    active: bool,
    spatial_threshold: float,
    params: dict,
) -> dict:
    pooled = _numpy_map(pooled_map)
    pooled[~valid] = 0
    threshold_binary = (
        (pooled.astype(np.float64) > spatial_threshold) & valid
        if active else np.zeros_like(valid)
    )
    processed_binary = _clean_binary(threshold_binary, valid, params)
    boxes = _component_boxes(
        processed_binary, pooled, valid, params, source_scale=kernel
    ) if active else []
    return {
        "kernel": int(kernel),
        "map": pooled,
        "raw_score": float(raw_score),
        "normalized_score": None if normalized_score is None else float(normalized_score),
        "active": bool(active),
        "spatial_threshold": float(spatial_threshold),
        "pooled_max": float(pooled[valid].max()),
        "binary_map": threshold_binary,
        "processed_binary_map": processed_binary,
        "boxes": boxes,
    }


def build_localization(
    anomaly_map,
    ignore_mask,
    *,
    score_mode: str,
    score_method: dict,
    score_values: dict,
    threshold: float,
    raw_display_max: float,
    params: dict | None = None,
) -> dict:
    """按照当前整图 Score 定义建立用于画框的空间响应。

    ``top`` 使用原始异常图；``pool_topk`` 使用完全相同的 mask-aware 池化图；
    ``multiscale_pool`` 只融合整图尺度分数超过阈值的尺度。
    """
    params = normalize_localization_params(params)
    threshold = float(threshold)
    score = float(score_values["score"])
    if not all(math.isfinite(value) for value in (threshold, score, raw_display_max)):
        raise ValueError("定位使用的 score、threshold 和 display_max 必须是有限数值。")
    tensor = _single_map(anomaly_map)
    ignore = expanded_ignore_mask(tensor, ignore_mask)
    valid = _numpy_map(~ignore).astype(bool)
    if not valid.any():
        raise ValueError("mask 后没有可用于定位的有效像素。")
    raw_map = _numpy_map(tensor)
    raw_map[~valid] = 0
    predicted_anomaly = score > threshold
    # threshold_for_target_recall 使用严格大于号；边界为 0 时分类阈值可能是
    # nextafter(0, -inf)。异常响应通常非负，画框时直接使用这个负数会把所有
    # 零响应像素选中。分类仍沿用原阈值，空间定位统一把阈值下限限制为 0。
    localization_threshold = max(0.0, threshold)

    if score_mode == "top":
        threshold_binary = (
            (raw_map.astype(np.float64) > localization_threshold) & valid
            if predicted_anomaly else np.zeros_like(valid)
        )
        processed = _clean_binary(threshold_binary, valid, params)
        boxes = _component_boxes(processed, raw_map, valid, params, source_scale=1)
        response, response_threshold = raw_map, localization_threshold
        response_title = "Raw response M"
        scales = []
        display_max = raw_display_max
    elif score_mode == "pool_topk":
        kernel = int(score_method.get("pool_kernel", 21))
        pooled_map, _ = mask_aware_average_pool(tensor, ignore, pool_kernel=kernel)
        scale = _scale_payload(
            pooled_map, valid, kernel=kernel,
            raw_score=score, normalized_score=None, active=predicted_anomaly,
            spatial_threshold=localization_threshold, params=params,
        )
        response = scale["map"]
        threshold_binary = scale["binary_map"]
        processed = scale["processed_binary_map"]
        boxes = scale["boxes"]
        response_threshold = localization_threshold
        response_title = f"Pool {kernel}×{kernel}"
        scales = [scale]
        # 原始异常图和单尺度池化图量纲一致，主图严格共用固定色阶。
        display_max = raw_display_max
    elif score_mode == "multiscale_pool":
        # threshold_for_target_recall 为适配严格大于号，边界为 0 时会返回
        # nextafter(0, -inf)。归一化空间 L>=0，直接使用该极小负数会令整图
        # 都满足 L>T；定位阈值因此以 0 为下限，分类阈值本身保持不变。
        scales = []
        active_normalized_maps = []
        all_boxes = []
        threshold_binary = np.zeros_like(valid)
        processed = np.zeros_like(valid)
        for kernel in score_method["pool_kernels"]:
            kernel = int(kernel)
            reference = score_method["normalization"][str(kernel)]
            median = float(reference["median"])
            denominator = float(reference["denominator"])
            if (not math.isfinite(median) or not math.isfinite(denominator)
                    or denominator <= 0):
                raise ValueError(f"尺度 {kernel} 的定位归一化参数无效。")
            raw_score = float(score_values[f"score_kernel_{kernel}"])
            normalized_score = float(score_values[f"score_normalized_kernel_{kernel}"])
            active = predicted_anomaly and normalized_score > threshold
            pooled_map, _ = mask_aware_average_pool(tensor, ignore, pool_kernel=kernel)
            normalized_map = np.maximum(0, (_numpy_map(pooled_map) - median) / denominator)
            normalized_map[~valid] = 0
            spatial_threshold = median + localization_threshold * denominator
            scale = _scale_payload(
                pooled_map, valid, kernel=kernel, raw_score=raw_score,
                normalized_score=normalized_score, active=active,
                spatial_threshold=spatial_threshold, params=params,
            )
            scale["normalized_map"] = normalized_map.astype(np.float32, copy=False)
            # 多尺度定位定义是 L_k > Threshold。尤其当整图阈值为负数时，
            # 不能用未截断的反算原始阈值替代，否则不再等价。
            scale["binary_map"] = (
                (scale["normalized_map"].astype(np.float64) > localization_threshold) & valid
                if active else np.zeros_like(valid)
            )
            scale["processed_binary_map"] = _clean_binary(
                scale["binary_map"], valid, params
            )
            scale["boxes"] = _component_boxes(
                scale["processed_binary_map"], scale["normalized_map"], valid,
                params, source_scale=kernel,
            ) if active else []
            scales.append(scale)
            if active:
                active_normalized_maps.append(scale["normalized_map"])
                threshold_binary |= scale["binary_map"]
                processed |= scale["processed_binary_map"]
                all_boxes.extend(scale["boxes"])
        response = (
            np.maximum.reduce(active_normalized_maps)
            if active_normalized_maps else np.zeros_like(raw_map)
        )
        boxes = merge_boxes(all_boxes, raw_map.shape, params)
        response_threshold = localization_threshold
        response_title = "Active multiscale normalized fusion L"
        display_max = max(
            params["normalized_display_max"], max(0.0, threshold) * 1.25
        )
    else:
        raise ValueError(f"不支持的定位 score_mode：{score_mode}")

    boxes = merge_boxes(boxes, raw_map.shape, params)
    if predicted_anomaly and not boxes:
        active_scales = [scale["kernel"] for scale in scales if scale["active"]] or [1]
        fallback_response = response
        fallback_source = "localization_response"
        if score_mode == "multiscale_pool" and float(response[valid].max()) <= 0:
            # 归一化响应全部为 0 时没有唯一最大位置，改用原始异常图寻找最强点，
            # 避免 np.argmax 固定把兜底框放在左上角。
            fallback_response = raw_map
            fallback_source = "raw_anomaly_map"
        fallback = _fallback_box(fallback_response, valid, params, active_scales)
        fallback["fallback_source"] = fallback_source
        boxes = [fallback]
    return {
        "score_mode": score_mode,
        "predicted_anomaly": predicted_anomaly,
        "score": score,
        "threshold": threshold,
        "raw_map": raw_map,
        "valid_mask": valid,
        "response_map": response.astype(np.float32, copy=False),
        "response_threshold": float(response_threshold),
        "threshold_adjusted_for_localization": bool(response_threshold != threshold),
        "response_title": response_title,
        "response_display_max": float(display_max),
        "binary_map": threshold_binary.astype(bool),
        "processed_binary_map": processed.astype(bool),
        "boxes": boxes,
        "scales": scales,
        "params": params,
    }


def localization_summary(localization: dict) -> dict:
    """去除大型数组，返回可写入 prediction.json 的定位摘要。"""
    def json_native(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: json_native(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_native(item) for item in value]
        return value

    scale_summaries = []
    for scale in localization.get("scales", []):
        scale_summaries.append({
            key: value
            for key, value in scale.items()
            if key not in {"map", "normalized_map", "binary_map", "processed_binary_map"}
        })
    summary = {
        key: value
        for key, value in localization.items()
        if key not in {
            "raw_map", "valid_mask", "response_map", "binary_map",
            "processed_binary_map", "scales",
        }
    } | {"scales": scale_summaries}
    return json_native(summary)
