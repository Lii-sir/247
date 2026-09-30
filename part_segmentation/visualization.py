"""纯绘图函数：不加载模型、不写文件、不依赖 Qt。所有颜色均为 BGR。"""

import cv2 as cv
import numpy as np

from .models import SegmentationResult

# 同一类别在所有图片中保持同色。
PALETTE = ((60, 190, 255), (230, 170, 40), (120, 210, 100), (210, 100, 220))


def class_color(class_id: int) -> tuple[int, int, int]:
    return PALETTE[class_id % len(PALETTE)]


def render_overlay(
    result: SegmentationResult,
    alpha: float = 0.45,
    show_boxes: bool = False,
    show_labels: bool = True,
) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError("掩膜透明度必须在 0 和 1 之间")
    output = result.image.copy()
    height, width = output.shape[:2]
    thickness = max(1, round(max(height, width) / 1000))
    # 低置信度先绘制，重叠区域由高置信度实例覆盖；只混合一次。
    color_layer = result.image.copy()
    for segment in sorted(result.segments, key=lambda item: item.confidence):
        if segment.mask.shape != (height, width):
            raise ValueError("掩膜尺寸必须与原图一致")
        color_layer[segment.mask] = class_color(segment.class_id)
    output = cv.addWeighted(output, 1 - alpha, color_layer, alpha, 0)
    for index, segment in enumerate(result.segments, start=1):
        color = class_color(segment.class_id)
        contours, _ = cv.findContours(segment.mask.astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(output, contours, -1, color, thickness)
        x1, y1, x2, y2 = (int(round(value)) for value in segment.box)
        if show_boxes:
            cv.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        if show_labels:
            label = f"#{index} {segment.class_name} {segment.confidence:.2f}"
            scale = max(0.45, max(height, width) / 2400)
            (tw, th), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
            x = max(0, min(x1, width - tw - 8))
            y = max(th + 8, min(y1, height - baseline - 4))
            cv.rectangle(output, (x, y - th - 6), (x + tw + 6, y + baseline), color, -1)
            cv.putText(output, label, (x + 3, y - 3), cv.FONT_HERSHEY_SIMPLEX,
                       scale, (20, 25, 30), thickness, cv.LINE_AA)
    return output


def render_comparison(result: SegmentationResult, **options) -> np.ndarray:
    """原图与叠加图并排，保持原始分辨率；上方附英文标题和类别图例。"""
    overlay = render_overlay(result, **options)
    height, width = result.image.shape[:2]
    header = max(80, round(width * 0.06))
    canvas = np.full((height + header, width * 2, 3), (30, 26, 22), dtype=np.uint8)
    canvas[header:, :width] = result.image
    canvas[header:, width:] = overlay
    scale = max(0.6, width / 1800)
    cv.putText(canvas, "ORIGINAL", (20, int(header * 0.4)), cv.FONT_HERSHEY_SIMPLEX,
               scale, (235, 235, 235), 2, cv.LINE_AA)
    title = f"SEGMENTATION | {len(result.segments)} instances"
    cv.putText(canvas, title, (width + 20, int(header * 0.4)), cv.FONT_HERSHEY_SIMPLEX,
               scale, (235, 235, 235), 2, cv.LINE_AA)
    names = {segment.class_id: segment.class_name for segment in result.segments}
    x = width + 20
    for class_id, name in sorted(names.items()):
        count = sum(segment.class_id == class_id for segment in result.segments)
        label = f"{name}: {count}   "
        cv.putText(canvas, label, (x, int(header * 0.82)), cv.FONT_HERSHEY_SIMPLEX,
                   scale * 0.75, class_color(class_id), 2, cv.LINE_AA)
        x += cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, scale * 0.75, 2)[0][0]
    return canvas

