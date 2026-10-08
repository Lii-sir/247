"""Upper-level orchestration of the two existing, independent algorithm packages."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2 as cv
import numpy as np

from common.image_io import read_image
from common.segmentation.inference import PartSegmenter
from common.segmentation.models import SegmentationResult, SegmentationSettings
from common.matching import MatchResult, MatchSettings, TemplateMatcher
from .calibration import Calibration
from .geometry import Boundary, OverflowMeasurement, OverflowSettings, measure_overflow


@dataclass
class InspectionResult:
    segmentation: SegmentationResult
    match: MatchResult
    status: str
    message: str
    silver_count: int
    boundary: Boundary | None = None
    measurement: OverflowMeasurement | None = None
    elapsed_ms: float = 0.0

    @property
    def is_defect(self):
        return {"ok": False, "overflow": True}.get(self.status)

    def summary(self):
        return {
            "image": str(self.segmentation.image_path), "status": self.status,
            "message": self.message, "is_defect": self.is_defect,
            "silver_count": self.silver_count, "elapsed_ms": self.elapsed_ms,
            "match": asdict(self.match),
            "boundary_mode": self.boundary.mode if self.boundary else None,
            "target_boundary_points": self.boundary.points if self.boundary else None,
            "measurement": self.measurement.summary() if self.measurement else None,
        }


def evaluate(segmentation: SegmentationResult, match: MatchResult, boundary: Boundary,
             settings: OverflowSettings) -> InspectionResult:
    silver = [s for s in segmentation.segments if s.class_name.strip().casefold() == settings.silver_class.casefold()]
    result = InspectionResult(segmentation, match, "uncertain", "", len(silver))
    if match.status != "ok" or len(match.points) != len(boundary.points):
        result.message = f"匹配失败或边界不完整，不能判定：{match.message}"
        return result
    try:
        if not all(point.inside_image for point in match.points):
            raise ValueError("映射边界存在图外点")
        mapped = Boundary([(p.target_x, p.target_y) for p in match.points], boundary.mode)
        mapped.validate_image(segmentation.image.shape)
        result.boundary = mapped
        if not silver:
            result.status = "no_silver"
            result.message = f"未检出 {settings.silver_class}，不能判为合格，请检查类别、置信度和图片"
            return result
        mask = np.zeros(segmentation.image.shape[:2], dtype=bool)
        for segment in silver:
            if segment.mask.shape != mask.shape or segment.mask.dtype != np.bool_:
                raise ValueError("银浆掩膜不是原图尺寸的布尔数组")
            mask |= segment.mask  # Union avoids double-counting overlapping instances.
        if not mask.any():
            result.status, result.message = "no_silver", "银浆掩膜为空，不能判为合格"
            return result
        result.measurement = measure_overflow(mask, mapped, settings)
        result.status = "overflow" if result.measurement.defect_area_px else "ok"
        result.message = (f"发现 {len(result.measurement.regions)} 处溢出，面积 {result.measurement.defect_area_px} px"
                          if result.is_defect else "在当前容差和最小面积设置下未发现溢出")
    except (ValueError, cv.error) as exc:
        result.status, result.message = "uncertain", str(exc)
    return result


class SilverInspector:
    def __init__(self, weights, calibration: Calibration, settings: OverflowSettings | None = None,
                 segmentation_settings: SegmentationSettings | None = None,
                 match_settings: MatchSettings | None = None, segmenter=None):
        self.weights = Path(weights).expanduser().resolve()
        self.calibration = calibration
        self.settings = settings or OverflowSettings()
        self.segmentation_settings = segmentation_settings or SegmentationSettings()
        self.match_settings = match_settings or MatchSettings()
        self.segmenter = segmenter if segmenter is not None else PartSegmenter(self.weights)
        names = self.segmenter.class_names
        if self.settings.silver_class.casefold() not in {name.strip().casefold() for name in names}:
            raise ValueError(f"权重不含银浆类别 '{self.settings.silver_class}'；实际类别：{', '.join(names)}。"
                             "请更换含 silver 的权重，或确认真实银浆类别后显式设置 silver-class；不会自动把 bond 当作 silver。")
        template = read_image(calibration.template_path)
        calibration.boundary.validate_image(template.shape)
        self.matcher = TemplateMatcher(template, calibration.boundary.points, self.match_settings)

    def inspect(self, path) -> InspectionResult:
        start = perf_counter()
        segmentation = self.segmenter.predict(path, self.segmentation_settings)
        try:
            match = self.matcher.match(segmentation.image, str(segmentation.image_path))
        except (ValueError, cv.error) as exc:
            match = MatchResult(str(segmentation.image_path), "error", str(exc))
        result = evaluate(segmentation, match, self.calibration.boundary, self.settings)
        result.elapsed_ms = (perf_counter() - start) * 1000
        return result
