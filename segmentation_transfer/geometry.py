"""纯几何：对应点拟合、矩阵组合与二值掩膜变换，无文件/模型/GUI 依赖。"""

import math

import cv2 as cv
import numpy as np

from .models import HomographyFit, MappingSettings


def normalize_homography(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("映射矩阵必须是有限的 3×3 矩阵")
    scale = np.max(np.abs(matrix))
    if scale == 0:
        raise ValueError("映射矩阵不可逆")
    matrix = matrix / scale
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("映射矩阵不可逆或接近奇异")
    return matrix / matrix[2, 2] if abs(matrix[2, 2]) > 1e-12 else matrix


def project_points(points, matrix) -> np.ndarray:
    matrix = normalize_homography(matrix)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("点坐标必须为有限的 N×2 数组")
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError("映射点接近无穷远，无法使用该透视变换")
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(projected).all():
        raise ValueError("映射点坐标无效")
    return projected


def validate_points(points, image_shape, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError(f"{name}至少需要 4 个二维点")
    height, width = image_shape[:2]
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > [width - 1, height - 1]):
        raise ValueError(f"{name}选点必须在模板原图范围内，且不能包含 NaN/无穷大")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError(f"{name}存在重复点，请重新选点")
    spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    area = cv.contourArea(cv.convexHull(points.astype(np.float32)))
    if spread[1] < spread[0] * 1e-3 or area < max(16, width * height * 0.0001):
        raise ValueError(f"{name}选点共线、过于集中或接近共线，请在目标区域周围分散选点")
    return points


def validate_domain(matrix, image_shape) -> None:
    """拒绝穿过源图像的透视无穷远线，避免掩膜被翻转/拉成伪影。"""
    matrix = normalize_homography(matrix)
    height, width = image_shape[:2]
    corners = np.array([[0, 0, 1], [width - 1, 0, 1],
                        [width - 1, height - 1, 1], [0, height - 1, 1]])
    denominator = corners @ matrix[2]
    if not (np.all(denominator > 1e-9) or np.all(denominator < -1e-9)):
        raise ValueError("透视无穷远线穿过源图像，无法安全映射整个分割结果")


def fit_template_mapping(points_a, points_b, shape_a, shape_b,
                         settings: MappingSettings | None = None) -> HomographyFit:
    settings = settings or MappingSettings()
    a = validate_points(points_a, shape_a, "A 模板")
    b = validate_points(points_b, shape_b, "B 模板")
    if len(a) != len(b):
        raise ValueError("A/B 模板点数必须相等，按编号一一对应")
    matrix, inliers = cv.findHomography(a, b, cv.RANSAC, settings.ransac_threshold)
    if matrix is None or inliers is None:
        raise ValueError("人工对应点无法建立单应性映射，请检查点序与分布")
    matrix = normalize_homography(matrix)
    errors = np.linalg.norm(project_points(a, matrix) - b, axis=1)
    # 重拟合后的残差也需满足阈值，不能只信任初始 RANSAC 标记。
    inliers = inliers.ravel().astype(bool) & (errors <= settings.ransac_threshold)
    required = max(4, math.ceil(len(a) * settings.min_inlier_ratio))
    if int(inliers.sum()) < required:
        raise ValueError(f"模板对应关系不可靠：{inliers.sum()}/{len(a)} 对内点，至少需要 {required} 对")
    validate_points(a[inliers], shape_a, "A 模板内点")
    validate_points(b[inliers], shape_b, "B 模板内点")
    validate_domain(matrix, shape_a)
    return HomographyFit(matrix, inliers, errors)


def compose_image_mapping(template_a_to_a1, template_a_to_b, template_b_to_b1) -> np.ndarray:
    """列向量约定：H(A1→B1) = H(B→B1) @ H(A→B) @ inverse(H(A→A1))。"""
    ha = normalize_homography(template_a_to_a1)
    hab = normalize_homography(template_a_to_b)
    hb = normalize_homography(template_b_to_b1)
    return normalize_homography(hb @ hab @ np.linalg.inv(ha))


def warp_binary_mask(mask: np.ndarray, matrix, target_shape) -> np.ndarray:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("源掩膜必须是二维 bool 数组")
    height, width = target_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("目标图像尺寸无效")
    # 矩阵是 source→destination；不设置 WARP_INVERSE_MAP。二值掩膜只用最近邻。
    return cv.warpPerspective(mask.astype(np.uint8), normalize_homography(matrix), (width, height),
                              flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0).astype(bool)

