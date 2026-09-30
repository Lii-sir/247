"""流程数据对象。下层算法不需要知道本模块的存在。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from part_segmentation.models import SegmentationResult
from point_matcher.core import MatchResult


@dataclass(frozen=True)
class Calibration:
    template_a: Path
    template_b: Path
    # 同一下标代表一对人工对应点，坐标为解码后的模板原图像素。
    points_a: tuple[tuple[float, float], ...]
    points_b: tuple[tuple[float, float], ...]
    # 与对应点同下标；None 为普通点，正数为圆半径（原图像素）。圆心即 points。
    # 空 tuple 兼容旧标定。几何算法只读取 points，不使用圆边缘生成额外对应点。
    radii_a: tuple[float | None, ...] = ()
    radii_b: tuple[float | None, ...] = ()

    def __post_init__(self):
        for points, radii in ((self.points_a, self.radii_a), (self.points_b, self.radii_b)):
            if radii and len(radii) != len(points):
                raise ValueError("圆标注数量必须与对应点数量一致")
            if any(r is not None and (not np.isfinite(r) or r <= 0) for r in radii):
                raise ValueError("圆半径必须是有限的正数，普通点使用 null")


@dataclass(frozen=True)
class MappingSettings:
    ransac_threshold: float = 3.0  # B 模板像素
    min_inlier_ratio: float = 0.6

    def __post_init__(self):
        if not np.isfinite(self.ransac_threshold) or self.ransac_threshold <= 0:
            raise ValueError("对应点误差阈值必须为正数")
        if not 0 < self.min_inlier_ratio <= 1:
            raise ValueError("对应点最小内点比例必须在 (0, 1] 范围内")


@dataclass(frozen=True)
class HomographyFit:
    matrix: np.ndarray
    inliers: np.ndarray
    errors: np.ndarray  # 所有人工点在 B 模板中的前向重投影误差


@dataclass(frozen=True)
class TransferredInstance:
    source_id: int  # A1 分割结果中的编号，从 1 开始；空掩膜也保留编号
    class_id: int
    class_name: str
    source_confidence: float  # 仅 A1 模型置信度，不代表 B1 映射精度
    mask: np.ndarray  # B1 原图尺寸，bool
    box: tuple[int, int, int, int] | None  # xyxy，右下角为排他边界；空掩膜为 None
    source_area: int
    source_coverage: float  # 源掩膜中落入 B1 视野的像素比例（最近邻近似）
    status: str  # ok / clipped / empty

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class TransferResult:
    calibration: Calibration
    fit: HomographyFit
    match_a: MatchResult
    match_b: MatchResult
    matrix_a1_to_b1: np.ndarray
    point_errors_b1: np.ndarray  # 链式映射点与 B 模板投影点的差；不是独立精度测量
    source: SegmentationResult
    target_path: Path
    target_image: np.ndarray
    instances: tuple[TransferredInstance, ...]
    warnings: tuple[str, ...]

