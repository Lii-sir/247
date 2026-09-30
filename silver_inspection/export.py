"""Headless batch inspection with explicit unknown/error statuses."""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from part_segmentation.image_io import collect_images, write_image
from .pipeline import SilverInspector
from .visualization import render_comparison, render_overlay


def export_batch(weights, source, output, calibration, settings, segmentation_settings,
                 recursive=False, calibration_path=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    paths = collect_images(source, recursive)
    if source.is_dir() and output.is_relative_to(source):
        raise ValueError("输出目录不能位于输入图片目录内")
    protected = {Path(weights).resolve(), calibration.template_path.resolve(), *paths}
    if calibration_path is not None:
        protected.add(Path(calibration_path).resolve())
    destinations = [output / "summary.json"]
    for path in paths:
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        destinations.extend(output / relative.parent / f"{relative.name}.{suffix}"
                            for suffix in ("overlay.png", "comparison.jpg", "overflow.png"))
    if any(path.resolve() in protected for path in destinations):
        raise ValueError("输出将覆盖输入图片、模板、标定或权重，请另选目录")
    inspector = SilverInspector(weights, calibration, settings, segmentation_settings)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        try:
            result = inspector.inspect(path)
            record = result.summary()
            base = output / relative.parent / relative.name
            write_image(str(base) + ".overlay.png", render_overlay(result))
            write_image(str(base) + ".comparison.jpg", render_comparison(result))
            # Always write a mask, including unknown results; status, not mask alone, is authoritative.
            import numpy as np
            mask = (result.measurement.defect_mask.astype(np.uint8) * 255 if result.measurement
                    else np.zeros(result.segmentation.image.shape[:2], dtype=np.uint8))
            write_image(str(base) + ".overflow.png", mask)
        except Exception as exc:
            record = {"image": str(path), "status": "error", "is_defect": None, "message": str(exc)}
        records.append(record)
        print(f"[{index}/{len(paths)}] {relative}: {record['status']} - {record['message']}")
    report = {
        "schema_version": 1,
        "coordinate_system": "original target image pixels; origin top-left; x right; y down; pixel centers at integer coordinates",
        "weights": str(Path(weights).resolve()), "calibration": calibration.summary(),
        "overflow_settings": asdict(settings), "segmentation_settings": asdict(segmentation_settings),
        "match_settings": asdict(inspector.match_settings),
        "counts": dict(Counter(r["status"] for r in records)), "images": records,
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return int(any(r["status"] not in {"ok", "overflow"} for r in records))
