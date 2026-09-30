"""只绘图：复用分割配色和找点标记，不进行推理。"""

import cv2 as cv
import numpy as np

from part_segmentation.models import Segment, SegmentationResult
from part_segmentation.visualization import render_overlay
from point_matcher.core import annotate_image

from .models import TransferResult


def render_views(result: TransferResult, alpha: float = 0.45, show_masks: bool = True,
                 show_points: bool = True) -> tuple[np.ndarray, np.ndarray]:
    a = result.source.image.copy()
    b = result.target_image.copy()
    if show_masks:
        a = render_overlay(result.source, alpha=alpha, show_labels=False)
        # 不重新编号标签，表格始终用 source_id；这里仅绘制同类同色的掩膜。
        segments = tuple(Segment(item.class_id, item.class_name, item.source_confidence,
                                 item.box, item.mask) for item in result.instances if item.box is not None)
        mapped = SegmentationResult(result.target_path, result.target_image, segments, 0)
        b = render_overlay(mapped, alpha=alpha, show_labels=False)
    if show_points:
        a = annotate_image(a, result.match_a)
        b = annotate_image(b, result.match_b)
    return a, b


def render_comparison(result: TransferResult) -> np.ndarray:
    a, b = render_views(result)
    # 比较图仅用于展示；实际 PNG 掩膜仍使用 B1 原始分辨率。
    height = min(1600, max(a.shape[0], b.shape[0]))
    panels = []
    for image, title in ((a, "A1 | YOLO segmentation"), (b, "B1 | transferred masks (not YOLO)")):
        width = max(1, round(image.shape[1] * height / image.shape[0]))
        resized = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
        panel = np.full((height + 60, max(width, 520), 3), (30, 26, 22), dtype=np.uint8)
        panel[60:, :width] = resized
        cv.putText(panel, title, (12, 38), cv.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)

