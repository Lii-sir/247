"""推理、绘制和界面之间共享的数据结构，不依赖 YOLO 或 Qt。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SegmentationSettings:
    confidence: float = 0.25
    iou: float = 0.7
    image_size: int = 640
    device: str = "0"

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("置信度必须在 0 和 1 之间")
        if not 0 < self.iou <= 1:
            raise ValueError("IoU 必须大于 0 且不超过 1")
        if self.image_size < 32 or self.image_size % 32:
            raise ValueError("推理尺寸必须是至少 32 的 32 倍数")
        if not self.device.strip():
            raise ValueError("推理设备不能为空，例如 cpu 或 0")


@dataclass(frozen=True)
class Segment:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]
    mask: np.ndarray  # bool，尺寸与原图相同；不经过轮廓重建，保留孔洞。

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class SegmentationResult:
    image_path: Path
    image: np.ndarray  # uint8 BGR，原图像素坐标
    segments: tuple[Segment, ...]
    elapsed_ms: float  # 图片读取 + 推理 + 转换；不含模型首次加载

