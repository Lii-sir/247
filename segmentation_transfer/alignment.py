"""原图对齐检查的纯绘图层：将 B1 重采样到 A1 坐标，不重新估计映射。"""

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .geometry import normalize_homography


@dataclass(frozen=True)
class AlignmentPreview:
    image_a: np.ndarray
    warped_b: np.ndarray
    valid: np.ndarray  # B1 对该 A1 像素有完整插值支持；不是由像素颜色判定。


def prepare_alignment(image_a: np.ndarray, image_b: np.ndarray, matrix_a1_to_b1) -> AlignmentPreview:
    """输入为 A1→B1，实际使用其逆矩阵把 B1 投影到 A1 原图画布。"""
    for image in (image_a, image_b):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError("对齐预览需要非空 uint8 BGR 原图")
    inverse = normalize_homography(np.linalg.inv(normalize_homography(matrix_a1_to_b1)))
    height, width = image_a.shape[:2]
    warped = cv.warpPerspective(image_b, inverse, (width, height), flags=cv.INTER_LINEAR,
                                borderMode=cv.BORDER_CONSTANT, borderValue=0)
    support = cv.warpPerspective(np.ones(image_b.shape[:2], dtype=np.float32), inverse,
                                 (width, height), flags=cv.INTER_LINEAR,
                                 borderMode=cv.BORDER_CONSTANT, borderValue=0)
    # 不混合插值邻域越界的边缘像素，避免黑色填充造成伪轮廓。
    return AlignmentPreview(image_a, warped, support >= 1 - 1e-6)


def blend_alignment(preview: AlignmentPreview, opacity: float = .5) -> np.ndarray:
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("B1 不透明度必须在 0 和 1 之间")
    output = cv.addWeighted(preview.image_a, 1 - opacity, preview.warped_b, opacity, 0)
    output[~preview.valid] = preview.image_a[~preview.valid]
    return output
