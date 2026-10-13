from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# 寻找基板右侧圆孔作为基准定位
def find_circle(image: np.ndarray, fixed_radius: int | None = None,
                radius_tolerance: float = 0.03,
                search_roi: tuple[int, int, int, int] | None = None) -> tuple[int, int, int]:
    """Find the circular anchor inside the calibrated search ROI."""
    height, width = image.shape[:2]
    if search_roi is None:
        raise ValueError("必须配置圆搜索区域 circle_search_roi")
    search_x, search_y, search_x2, search_y2 = search_roi
    if not (0 <= search_x < search_x2 <= width and 0 <= search_y < search_y2 <= height):
        raise ValueError(
            f"圆搜索区域超出图像范围: {search_roi}, image_shape={image.shape[:2]}"
        )
    search_image = image[search_y:search_y2, search_x:search_x2]
    search_height, search_width = search_image.shape[:2]
    expected_center = (search_width / 2, search_height / 2)
    center_scales = (search_width / 2, search_height / 2)

    gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    search = cv2.GaussianBlur(gray, (9, 9), 2)
    candidates = []
    if fixed_radius:
        min_fixed = max(1, round(fixed_radius * (1 - radius_tolerance)))
        max_fixed = round(fixed_radius * (1 + radius_tolerance))
    for param2, min_scale, max_scale in ((35, 0.10, 0.23), (28, 0.08, 0.30), (22, 0.06, 0.35)):
        if fixed_radius:
            min_radius, max_radius = min_fixed, max_fixed
        else:
            min_radius = max(20, int(height * min_scale))
            max_radius = max(21, int(height * max_scale))
        circles = cv2.HoughCircles(search, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=max(80, int(height * 0.10)),
                                   param1=100, param2=param2,
                                   minRadius=min_radius, maxRadius=max_radius)
        if circles is not None:
            candidates.extend(np.rint(circles[0]).astype(int))
        if candidates:
            break
    if not candidates:
        raise RuntimeError("未检测到大圆 请检查圆检测参数或输入图像")
    candidates = np.asarray(candidates)
    expected_radius = fixed_radius or height * 0.15
    expected = np.array([*expected_center, expected_radius])
    scales = np.array([*center_scales, expected_radius * 0.35])
    cx, cy, radius = min(candidates, key=lambda item: np.linalg.norm((item - expected) / scales))
    return int(cx + search_x), int(cy + search_y), int(radius)

# 根据配置文件中预设的偏移量裁剪ROI区域
def crop_from_config(image: np.ndarray, circle: tuple[int, int, int],
                     config: dict[str, Any]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    cx, cy, radius = circle
    offsets = config["roi_offsets"]
    x1, y1 = round(cx + offsets[0] * radius), round(cy + offsets[1] * radius)
    x2, y2 = round(cx + offsets[2] * radius), round(cy + offsets[3] * radius)
    height, width = image.shape[:2]
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(1, min(width, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(1, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"计算出的 ROI 无效: {(x1, y1, x2, y2)}")
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)
