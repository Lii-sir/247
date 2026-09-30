"""Segmentation and chip-surrounding silver continuity orchestration."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from part_segmentation.inference import PartSegmenter
from part_segmentation.models import SegmentationResult, SegmentationSettings

from .continuity import ContinuityMeasurement, ContinuitySettings, measure_continuity, _key


@dataclass
class ContinuityResult:
    segmentation: SegmentationResult
    status: str
    message: str
    measurement: ContinuityMeasurement | None = None
    elapsed_ms: float = 0.0

    @property
    def is_defect(self):
        return {"ok": False, "disconnected": True}.get(self.status)

    def summary(self):
        return {
            "image": str(self.segmentation.image_path),
            "status": self.status,
            "message": self.message,
            "is_defect": self.is_defect,
            "elapsed_ms": self.elapsed_ms,
            "measurement": self.measurement.summary() if self.measurement else None,
        }


def evaluate_continuity(segmentation: SegmentationResult, settings: ContinuitySettings) -> ContinuityResult:
    try:
        measurement = measure_continuity(segmentation.segments, segmentation.image.shape, settings)
    except ValueError as exc:
        return ContinuityResult(segmentation, "uncertain", str(exc))
    if measurement.is_disconnected is None:
        status = "uncertain"
        message = "有效检查扇区为 0，无法判断银浆是否连续"
    elif measurement.is_disconnected:
        status = "disconnected"
        message = (f"发现 {measurement.missing_sector_count} 个银浆缺失扇区，"
                   f"覆盖 {measurement.covered_sector_count}/{measurement.valid_sector_count} 个有效扇区")
    else:
        status = "ok"
        message = (f"银浆连续，覆盖 {measurement.covered_sector_count}/{measurement.valid_sector_count} 个有效扇区；"
                   f"忽略 {measurement.ignored_sector_count} 个遮挡/无效扇区")
    return ContinuityResult(segmentation, status, message, measurement)


class SilverContinuityInspector:
    def __init__(self, weights, settings: ContinuitySettings | None = None,
                 segmentation_settings: SegmentationSettings | None = None, segmenter=None):
        self.weights = Path(weights).expanduser().resolve()
        self.settings = settings or ContinuitySettings()
        self.segmentation_settings = segmentation_settings or SegmentationSettings()
        self.segmenter = segmenter if segmenter is not None else PartSegmenter(self.weights)
        names = self.segmenter.class_names
        normalized = {_key(name) for name in names}
        required = (_key(self.settings.silver_class), _key(self.settings.chip_class))
        missing = [name for name in required if name not in normalized]
        if missing:
            raise ValueError(f"权重缺少必要类别：{', '.join(missing)}；实际类别：{', '.join(names)}")

    def inspect(self, path) -> ContinuityResult:
        start = perf_counter()
        segmentation = self.segmenter.predict(path, self.segmentation_settings)
        result = evaluate_continuity(segmentation, self.settings)
        result.elapsed_ms = (perf_counter() - start) * 1000
        return result
