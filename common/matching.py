"""SIFT registration and point transfer, independent of the graphical interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2 as cv
import numpy as np

from common.image_io import read_image


class MatchError(ValueError):
    """An image cannot be registered reliably."""


@dataclass(frozen=True)
class MatchSettings:
    ratio_threshold: float = 0.7
    ransac_threshold: float = 5.0
    min_matches: int = 11
    min_inliers: int = 8
    min_inlier_ratio: float = 0.25
    max_features: int = 10000
    max_image_side: int = 2400

    def __post_init__(self):
        if not 0 < self.ratio_threshold < 1:
            raise ValueError("匹配比值必须在 0 和 1 之间")
        if self.ransac_threshold <= 0 or not np.isfinite(self.ransac_threshold):
            raise ValueError("RANSAC 阈值必须是正数")
        if self.min_matches < 4 or self.min_inliers < 4:
            raise ValueError("最少匹配点和内点数不得小于 4")
        if not 0 < self.min_inlier_ratio <= 1:
            raise ValueError("内点比例必须在 0 和 1 之间")
        if self.max_features < 4 or self.max_image_side < 64:
            raise ValueError("特征数或处理尺寸过小")


@dataclass
class MappedPoint:
    point_id: int
    template_x: float
    template_y: float
    target_x: float
    target_y: float
    inside_image: bool


@dataclass
class MatchResult:
    image_path: str
    status: str
    message: str
    points: list[MappedPoint] = field(default_factory=list)
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    median_error: float | None = None
    homography: list[list[float]] | None = None
    template_outline: list[list[float]] = field(default_factory=list)
    image_size: tuple[int, int] | None = None


def transform_points(points: Sequence, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise MatchError("变换矩阵异常，点被投影到无穷远")
    result = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(result).all():
        raise MatchError("点坐标计算失败")
    return result


class TemplateMatcher:
    """Extract template descriptors once, then reuse them for every target."""

    def __init__(self, template: np.ndarray, points: Sequence, settings: MatchSettings | None = None):
        self.settings = settings or MatchSettings()
        self.height, self.width = template.shape[:2]
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not len(self.points):
            raise MatchError("请先在模板图上选择至少一个点")
        if not np.isfinite(self.points).all() or np.any(self.points < 0):
            raise MatchError("模板点坐标无效")
        if np.any(self.points[:, 0] > self.width - 1) or np.any(self.points[:, 1] > self.height - 1):
            raise MatchError("选点超出模板图片范围")
        self.sift = cv.SIFT_create(nfeatures=self.settings.max_features)
        self.keypoints, self.descriptors = self._features(template)
        if self.descriptors is None or len(self.descriptors) < self.settings.min_matches:
            raise MatchError("模板特征不足：请选择包含纹理、文字或清晰边缘的模板图")

    def _features(self, image: np.ndarray):
        height, width = image.shape[:2]
        if image.ndim == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
        scale = min(1.0, self.settings.max_image_side / max(width, height))
        if scale < 1:
            gray = cv.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv.INTER_AREA)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        # Restore original pixels independently in x/y to account for rounded resizing.
        coordinates = np.array([k.pt for k in keypoints], dtype=np.float64).reshape(-1, 2)
        coordinates *= [width / gray.shape[1], height / gray.shape[0]]
        return coordinates, descriptors

    def match_path(self, path: str | Path) -> MatchResult:
        try:
            return self.match(read_image(path), str(path))
        except (MatchError, cv.error, OSError, ValueError) as exc:
            return MatchResult(str(path), "error", str(exc))

    def match(self, target: np.ndarray, image_path: str = "") -> MatchResult:
        target_points, descriptors = self._features(target)
        if descriptors is None or len(descriptors) < 2:
            raise MatchError("目标图特征不足，无法找到模板")
        flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=80))
        pairs = flann.knnMatch(self.descriptors, descriptors, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < self.settings.ratio_threshold * pair[1].distance]
        # A target keypoint must not be counted repeatedly as independent evidence.
        unique = {}
        for match in sorted(good, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        good = list(unique.values())
        if len(good) < self.settings.min_matches:
            raise MatchError(f"可靠匹配点不足：{len(good)} / {self.settings.min_matches}")
        source = self.keypoints[[m.queryIdx for m in good]]
        destination = target_points[[m.trainIdx for m in good]]
        matrix, mask = cv.findHomography(source, destination, cv.RANSAC, self.settings.ransac_threshold)
        if matrix is None or mask is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
            raise MatchError("无法估计有效的透视变换；请更换模板或目标图")
        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        fraction = inliers / len(good)
        if inliers < self.settings.min_inliers or fraction < self.settings.min_inlier_ratio:
            raise MatchError(f"匹配不可靠：{inliers} 个内点，内点占比 {fraction:.0%}")
        area = cv.contourArea(cv.convexHull(source[inlier_mask].astype(np.float32)))
        if area < max(16, self.width * self.height * 0.001):
            raise MatchError("匹配特征过于集中或接近共线，无法可靠映射选点")
        corners = np.array([[0, 0], [self.width - 1, 0], [self.width - 1, self.height - 1], [0, self.height - 1]])
        denominator = np.column_stack((corners, np.ones(4))) @ matrix[2]
        if not (np.all(denominator > 1e-9) or np.all(denominator < -1e-9)):
            raise MatchError("模板区域出现无效的透视翻转")
        outline = transform_points(corners, matrix)
        contour = outline.astype(np.float32)
        height, width = target.shape[:2]
        if (not cv.isContourConvex(contour) or cv.contourArea(contour) < 16
                or np.abs(outline).max() > max(width, height) * 100):
            raise MatchError("估计的模板区域异常，已拒绝本次匹配")
        projected = transform_points(self.points, matrix)
        mapped = [MappedPoint(
            i + 1, float(original[0]), float(original[1]), float(point[0]), float(point[1]),
            bool(0 <= point[0] < width and 0 <= point[1] < height),
        ) for i, (original, point) in enumerate(zip(self.points, projected))]
        outside = sum(not point.inside_image for point in mapped)
        status = "ok" if outside == 0 else "outside" if outside == len(mapped) else "partial"
        message = "定位成功" if outside == 0 else f"已匹配模板，但 {outside} 个选点落在目标图外"
        errors = np.linalg.norm(transform_points(source[inlier_mask], matrix) - destination[inlier_mask], axis=1)
        return MatchResult(
            image_path, status, message, mapped, len(good), inliers, fraction,
            float(np.median(errors)), matrix.tolist(), outline.tolist(), (width, height),
        )


def annotate_image(image: np.ndarray, result: MatchResult) -> np.ndarray:
    annotated = image.copy()
    thickness = max(2, round(min(image.shape[:2]) / 500))
    if result.template_outline:
        polygon = np.rint(result.template_outline).astype(np.int32)
        cv.polylines(annotated, [polygon], True, (255, 190, 40), thickness, cv.LINE_AA)
    for point in result.points:
        if not point.inside_image:
            continue
        x, y = round(point.target_x), round(point.target_y)
        cv.drawMarker(annotated, (x, y), (0, 225, 255), cv.MARKER_CROSS, thickness * 10, thickness, cv.LINE_AA)
        cv.circle(annotated, (x, y), thickness * 5, (0, 225, 255), thickness, cv.LINE_AA)
        cv.putText(annotated, f"P{point.point_id}", (x + thickness * 6, y - thickness * 4), cv.FONT_HERSHEY_SIMPLEX, thickness * 0.35, (0, 225, 255), thickness, cv.LINE_AA)
    return annotated
