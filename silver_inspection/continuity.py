"""Chip-surrounding silver continuity measurement.

Each chip is checked from its axis-aligned bounding rectangle. The rectangle is
expanded outward by a configurable number of pixels and the expanded rectangle
minus the original rectangle is sampled by angle. Thin/bond pixels are removed
from the denominator because they can hide silver in the image.
"""

from dataclasses import dataclass

import cv2 as cv
import numpy as np


EPSILON = 1e-6


def _key(name: str) -> str:
    return " ".join(str(name).strip().casefold().replace("_", " ").split())


@dataclass(frozen=True)
class ContinuitySettings:
    silver_class: str = "silver"
    chip_class: str = "chip"
    occlusion_classes: tuple[str, ...] = ("thin", "bond")
    outward_length_px: int = 20
    sector_count: int = 72
    min_sector_silver_px: int = 3
    min_sector_coverage: float = 0.01
    min_valid_sector_px: int = 1
    occlusion_dilation_px: int = 0
    min_chip_area_px: int = 100
    min_visible_sector_ratio: float = 0.1

    def __post_init__(self):
        for field_name in ("silver_class", "chip_class"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} 不能为空")
            object.__setattr__(self, field_name, value.strip())
        classes = tuple(str(value).strip() for value in self.occlusion_classes if str(value).strip())
        if len(classes) != len(set(_key(value) for value in classes)):
            raise ValueError("遮挡类别不能重复")
        object.__setattr__(self, "occlusion_classes", classes)
        if isinstance(self.outward_length_px, bool) or not isinstance(self.outward_length_px, int) or self.outward_length_px < 1:
            raise ValueError("芯片外扩长度必须是正整数像素")
        if isinstance(self.sector_count, bool) or not isinstance(self.sector_count, int) or not 4 <= self.sector_count <= 720:
            raise ValueError("角度扇区数量必须在 4 到 720 之间")
        if isinstance(self.min_sector_silver_px, bool) or not isinstance(self.min_sector_silver_px, int) or self.min_sector_silver_px < 1:
            raise ValueError("每个扇区的最小银浆像素数必须是正整数")
        if not np.isfinite(self.min_sector_coverage) or not 0 <= self.min_sector_coverage <= 1:
            raise ValueError("扇区银浆覆盖率必须在 0 到 1 之间")
        if isinstance(self.min_valid_sector_px, bool) or not isinstance(self.min_valid_sector_px, int) or self.min_valid_sector_px < 1:
            raise ValueError("扇区最小有效像素数必须是正整数")
        if isinstance(self.occlusion_dilation_px, bool) or not isinstance(self.occlusion_dilation_px, int) or self.occlusion_dilation_px < 0:
            raise ValueError("遮挡膨胀像素必须是非负整数")
        if isinstance(self.min_chip_area_px, bool) or not isinstance(self.min_chip_area_px, int) or self.min_chip_area_px < 1:
            raise ValueError("芯片最小连通域面积必须是正整数")
        if not np.isfinite(self.min_visible_sector_ratio) or not 0 <= self.min_visible_sector_ratio <= 1:
            raise ValueError("扇区最小可见比例必须在 0 到 1 之间")


@dataclass
class ContinuityMeasurement:
    chip_area_px: int
    ring_area_px: int
    silver_area_px: int
    occluded_area_px: int
    valid_area_px: int
    covered_sector_count: int
    valid_sector_count: int
    ignored_sector_count: int
    missing_sector_count: int
    center_xy: tuple[float, float]
    sectors: list[dict]
    chips: list[dict]
    chip_mask: np.ndarray
    ring_mask: np.ndarray
    silver_mask: np.ndarray
    occlusion_mask: np.ndarray
    missing_mask: np.ndarray

    @property
    def is_disconnected(self) -> bool | None:
        if self.valid_sector_count == 0:
            return None
        return self.missing_sector_count > 0

    def summary(self):
        return {
            "chip_area_px": self.chip_area_px,
            "ring_area_px": self.ring_area_px,
            "silver_area_px": self.silver_area_px,
            "occluded_area_px": self.occluded_area_px,
            "valid_area_px": self.valid_area_px,
            "covered_sector_count": self.covered_sector_count,
            "valid_sector_count": self.valid_sector_count,
            "ignored_sector_count": self.ignored_sector_count,
            "missing_sector_count": self.missing_sector_count,
            "coverage_ratio": self.covered_sector_count / self.valid_sector_count if self.valid_sector_count else None,
            "center_xy": self.center_xy,
            "sectors": self.sectors,
            "chips": self.chips,
        }


def _union_masks(segments, names, shape):
    wanted = {_key(name) for name in names}
    mask = np.zeros(shape, dtype=bool)
    count = 0
    for segment in segments:
        if _key(segment.class_name) not in wanted:
            continue
        if segment.mask.shape != shape or segment.mask.dtype != np.bool_:
            raise ValueError(f"类别 {segment.class_name} 的掩膜不是原图尺寸的布尔数组")
        mask |= segment.mask
        count += 1
    return mask, count


def _measure_chip_component(component, chip_union, silver, occlusion, settings, chip_index):
    height, width = component.shape
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        raise ValueError(f"第 {chip_index} 个 chip 没有有效像素")

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    distance = settings.outward_length_px
    outer_x1 = max(0, x1 - distance)
    outer_x2 = min(width, x2 + distance)
    outer_y1 = max(0, y1 - distance)
    outer_y2 = min(height, y2 + distance)

    inner_rectangle = np.zeros_like(component, dtype=bool)
    inner_rectangle[y1:y2, x1:x2] = True
    outer_rectangle = np.zeros_like(component, dtype=bool)
    outer_rectangle[outer_y1:outer_y2, outer_x1:outer_x2] = True
    ring = outer_rectangle & ~inner_rectangle & ~chip_union
    if not ring.any():
        raise ValueError(f"第 {chip_index} 个 chip 外扩后没有形成有效检查环带")

    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    effective_occlusion = ring & occlusion
    valid = ring & ~occlusion
    silver_ring = ring & silver
    effective_silver = silver_ring & ~occlusion
    yy, xx = np.indices(component.shape, dtype=np.float64)
    angles = (np.arctan2(yy - center[1], xx - center[0]) + 2 * np.pi) % (2 * np.pi)
    indices = np.floor(angles / (2 * np.pi) * settings.sector_count).astype(np.int32)
    sectors = []
    missing_mask = np.zeros(component.shape, dtype=bool)
    covered_count = valid_count = ignored_count = missing_count = 0
    for index in range(settings.sector_count):
        sector = ring & (indices == index)
        valid_sector = valid & (indices == index)
        silver_sector = effective_silver & (indices == index)
        occluded_sector = effective_occlusion & (indices == index)
        ring_px = int(sector.sum())
        valid_px = int(valid_sector.sum())
        silver_px = int(silver_sector.sum())
        occluded_px = int(occluded_sector.sum())
        coverage = silver_px / valid_px if valid_px else None
        visible_ratio = valid_px / ring_px if ring_px else 0.0
        mostly_occluded = occluded_px > 0 and visible_ratio < settings.min_visible_sector_ratio
        if valid_px < settings.min_valid_sector_px or mostly_occluded:
            status = "ignored" if occluded_px else "insufficient"
            if status == "ignored":
                ignored_count += 1
        else:
            valid_count += 1
            covered = silver_px >= settings.min_sector_silver_px and coverage >= settings.min_sector_coverage
            status = "covered" if covered else "missing"
            if covered:
                covered_count += 1
            else:
                missing_count += 1
                # Only visible, non-silver pixels are missing. Occluded pixels
                # must remain excluded from both the decision and the red mask.
                missing_mask |= valid_sector & ~silver_sector
        sectors.append({
            "chip_index": chip_index,
            "index": index,
            "angle_start_deg": index * 360 / settings.sector_count,
            "angle_end_deg": (index + 1) * 360 / settings.sector_count,
            "ring_px": ring_px,
            "valid_px": valid_px,
            "silver_px": silver_px,
            "occluded_px": occluded_px,
            "visible_ratio": visible_ratio,
            "coverage": coverage,
            "status": status,
        })
    return {
        "chip_mask": component, "ring_mask": ring, "silver_mask": silver_ring,
        "occlusion_mask": effective_occlusion, "missing_mask": missing_mask,
        "center_xy": center, "chip_area_px": int(component.sum()),
        "bbox_xyxy": [x1, y1, x2, y2],
        "outer_bbox_xyxy": [outer_x1, outer_y1, outer_x2, outer_y2],
        "ring_area_px": int(ring.sum()), "silver_area_px": int(silver_ring.sum()),
        "occluded_area_px": int(effective_occlusion.sum()), "valid_area_px": int(valid.sum()),
        "covered_sector_count": covered_count, "valid_sector_count": valid_count,
        "ignored_sector_count": ignored_count, "missing_sector_count": missing_count,
        "sectors": sectors,
    }


def measure_continuity(segments, image_shape, settings: ContinuitySettings) -> ContinuityMeasurement:
    shape = tuple(image_shape[:2])
    chip, chip_count = _union_masks(segments, (settings.chip_class,), shape)
    if chip_count == 0 or not chip.any():
        raise ValueError(f"未检测到 {settings.chip_class} 掩膜")
    silver, silver_count = _union_masks(segments, (settings.silver_class,), shape)
    if silver_count == 0 or not silver.any():
        raise ValueError(f"未检测到 {settings.silver_class} 掩膜")
    occlusion, _ = _union_masks(segments, settings.occlusion_classes, shape)
    if settings.occlusion_dilation_px:
        diameter = settings.occlusion_dilation_px * 2 + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (diameter, diameter))
        occlusion = cv.dilate(occlusion.astype(np.uint8), kernel) > 0

    count, labels, stats, centroids = cv.connectedComponentsWithStats(chip.astype(np.uint8), 8)
    components = []
    for label in range(1, count):
        area = int(stats[label, cv.CC_STAT_AREA])
        if area < settings.min_chip_area_px:
            continue
        component = labels == label
        components.append(_measure_chip_component(component, chip, silver, occlusion, settings, len(components) + 1))
    if not components:
        raise ValueError("未检测到达到最小面积的 chip 连通域")
    chip_mask = np.logical_or.reduce([item["chip_mask"] for item in components])
    ring_mask = np.logical_or.reduce([item["ring_mask"] for item in components])
    silver_mask = np.logical_or.reduce([item["silver_mask"] for item in components])
    occlusion_mask = np.logical_or.reduce([item["occlusion_mask"] for item in components])
    missing_mask = np.logical_or.reduce([item["missing_mask"] for item in components])
    centers = np.asarray([item["center_xy"] for item in components], dtype=float)
    weights = np.asarray([item["chip_area_px"] for item in components], dtype=float)
    center = tuple(map(float, np.average(centers, axis=0, weights=weights)))
    chips = [{key: value for key, value in item.items() if key not in {"chip_mask", "ring_mask", "silver_mask",
                                                                         "occlusion_mask", "missing_mask"}}
             for item in components]
    return ContinuityMeasurement(
        int(chip_mask.sum()), int(ring_mask.sum()), int(silver_mask.sum()), int(occlusion_mask.sum()),
        int((ring_mask & ~occlusion_mask).sum()), sum(item["covered_sector_count"] for item in components),
        sum(item["valid_sector_count"] for item in components), sum(item["ignored_sector_count"] for item in components),
        sum(item["missing_sector_count"] for item in components), center,
        [sector for item in components for sector in item["sectors"]], chips,
        chip_mask, ring_mask, silver_mask, occlusion_mask, missing_mask,
    )
