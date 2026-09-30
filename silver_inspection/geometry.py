"""Boundary geometry in original image pixels; no Qt or YOLO dependency."""

from dataclasses import dataclass

import cv2 as cv
import numpy as np

EPSILON = 1e-6


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _on_segment(a, b, p):
    return abs(_cross(b - a, p - a)) <= EPSILON and bool(
        np.all(p >= np.minimum(a, b) - EPSILON)
        and np.all(p <= np.maximum(a, b) + EPSILON))


def _intersects(a, b, c, d):
    ab_c, ab_d = _cross(b - a, c - a), _cross(b - a, d - a)
    cd_a, cd_b = _cross(d - c, a - c), _cross(d - c, b - c)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return True
    return any((_on_segment(a, b, c), _on_segment(a, b, d),
                _on_segment(c, d, a), _on_segment(c, d, b)))


@dataclass(frozen=True)
class Boundary:
    # polygon: vertices in perimeter order; line: endpoints followed by an inside point.
    points: tuple[tuple[float, float], ...]
    mode: str = "polygon"

    def __post_init__(self):
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("边界点必须是有限的二维坐标")
        object.__setattr__(self, "points", tuple(tuple(map(float, p)) for p in points))
        if self.mode not in {"polygon", "line"}:
            raise ValueError("边界模式必须是 polygon 或 line")
        if len(points) < 3 or (self.mode == "line" and len(points) != 3):
            raise ValueError("闭合区域至少需要 3 点；直线需要 2 个端点和 1 个内侧点")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("边界点不能重复，无需再次点击首点闭合")
        if self.mode == "line":
            distance = abs(_cross(points[1] - points[0], points[2] - points[0])) / np.linalg.norm(points[1] - points[0])
            if distance <= EPSILON:
                raise ValueError("内侧点不能位于边界直线上")
            return
        if abs(float(np.sum(_cross(points, np.roll(points, -1, axis=0))))) <= EPSILON:
            raise ValueError("闭合边界面积为零")
        for i in range(len(points)):
            a, b, c = points[i - 1], points[i], points[(i + 1) % len(points)]
            if abs(_cross(b - a, c - b)) <= EPSILON and np.dot(b - a, c - b) < 0:
                raise ValueError("边界包含重叠或折返边")
            for j in range(i + 1, len(points)):
                if j == i + 1 or (i == 0 and j == len(points) - 1):
                    continue
                if _intersects(points[i], points[(i + 1) % len(points)],
                               points[j], points[(j + 1) % len(points)]):
                    raise ValueError("边界不能自交，请按轮廓顺序选点")

    def validate_image(self, shape):
        height, width = shape[:2]
        points = np.asarray(self.points)
        if np.any(points < 0) or np.any(points[:, 0] > width - 1) or np.any(points[:, 1] > height - 1):
            raise ValueError("边界点或内侧点超出图片范围，不能可靠判定")

    def signed_distance(self, points):
        """Positive outside, negative inside, zero on the exact floating-point boundary."""
        query = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        vertices = np.asarray(self.points)
        if self.mode == "line":
            a, b, inside = vertices
            side = np.sign(_cross(b - a, inside - a))
            return -side * _cross(b - a, query - a) / np.linalg.norm(b - a)
        inside = np.zeros(len(query), dtype=bool)
        minimum = np.full(len(query), np.inf)
        x, y = query.T
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = b - a
            projection = np.clip(((query - a) @ edge) / np.dot(edge, edge), 0, 1)
            nearest = a + projection[:, None] * edge
            minimum = np.minimum(minimum, np.linalg.norm(query - nearest, axis=1))
            if abs(edge[1]) > 0:
                crossing = ((a[1] > y) != (b[1] > y)) & (x < a[0] + (y - a[1]) * edge[0] / edge[1])
                inside ^= crossing
        return np.where(inside, -minimum, minimum)


@dataclass(frozen=True)
class OverflowSettings:
    silver_class: str = "silver"
    tolerance_px: float = 0.0
    min_area_px: int = 1

    def __post_init__(self):
        if not isinstance(self.silver_class, str) or not self.silver_class.strip():
            raise ValueError("银浆类别名不能为空")
        object.__setattr__(self, "silver_class", self.silver_class.strip())
        if not np.isfinite(self.tolerance_px) or self.tolerance_px < 0:
            raise ValueError("越界容差必须是非负有限数，单位为目标原图像素")
        if isinstance(self.min_area_px, bool) or not isinstance(self.min_area_px, int) or self.min_area_px < 1:
            raise ValueError("最小越界连通域面积必须是正整数")


@dataclass
class OverflowMeasurement:
    silver_area_px: int
    outside_area_px: int
    candidate_area_px: int
    defect_area_px: int
    max_outside_distance_px: float
    regions: list[dict]
    silver_mask: np.ndarray
    outside_mask: np.ndarray
    defect_mask: np.ndarray

    def summary(self):
        return {
            "silver_area_px": self.silver_area_px,
            "outside_area_px": self.outside_area_px,
            "candidate_area_px": self.candidate_area_px,
            "defect_area_px": self.defect_area_px,
            "defect_ratio": self.defect_area_px / self.silver_area_px if self.silver_area_px else 0.0,
            "max_outside_distance_px": self.max_outside_distance_px,
            "regions": self.regions,
        }


def measure_overflow(mask, boundary: Boundary, settings: OverflowSettings) -> OverflowMeasurement:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("银浆掩膜必须是原图尺寸的二维布尔数组")
    boundary.validate_image(mask.shape)
    ys, xs = np.nonzero(mask)
    # Chunk the query to avoid multi-million-pixel floating-point temporaries.
    distances = np.empty(len(xs), dtype=np.float64)
    for offset in range(0, len(xs), 100_000):
        stop = offset + 100_000
        distances[offset:stop] = boundary.signed_distance(np.column_stack((xs[offset:stop], ys[offset:stop])))
    outside = np.zeros(mask.shape, dtype=bool)
    outside[ys, xs] = distances > EPSILON
    candidates = np.zeros(mask.shape, dtype=np.uint8)
    candidates[ys, xs] = distances > settings.tolerance_px + EPSILON
    count, labels, stats, _ = cv.connectedComponentsWithStats(candidates, connectivity=8)
    accepted = np.zeros(count, dtype=bool)
    maximum = np.zeros(count)
    np.maximum.at(maximum, labels[ys, xs], np.maximum(distances, 0))
    regions = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if area >= settings.min_area_px:
            accepted[label] = True
            regions.append({"area_px": area, "box_xyxy": [x, y, x + width, y + height],
                            "max_outside_distance_px": float(maximum[label])})
    defects = accepted[labels]
    return OverflowMeasurement(len(xs), int(outside.sum()), int(candidates.sum()), int(defects.sum()),
                               float(np.maximum(distances, 0).max()) if len(xs) else 0.0,
                               regions, mask.copy(), outside, defects)
