"""Headless batch export for silver continuity inspection."""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from part_segmentation.image_io import collect_images, write_image

from .continuity import ContinuitySettings
from .continuity_pipeline import SilverContinuityInspector
from .continuity_visualization import render_continuity_comparison, render_continuity_overlay


def export_continuity_batch(weights, source, output, settings, segmentation_settings, recursive=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    paths = collect_images(source, recursive)
    if source.is_dir() and output.is_relative_to(source):
        raise ValueError("输出目录不能位于输入图片目录内")
    protected = {Path(weights).resolve(), *paths}
    destinations = [output / "summary.json"]
    for path in paths:
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        base = output / relative.parent / relative.name
        destinations.extend((Path(str(base) + ".continuity.png"), Path(str(base) + ".comparison.jpg")))
    if any(destination.resolve() in protected for destination in destinations):
        raise ValueError("输出将覆盖输入图片或权重，请另选目录")
    inspector = SilverContinuityInspector(weights, settings, segmentation_settings)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        base = output / relative.parent / relative.name
        try:
            result = inspector.inspect(path)
            record = result.summary()
            write_image(str(base) + ".continuity.png", render_continuity_overlay(result))
            write_image(str(base) + ".comparison.jpg", render_continuity_comparison(result))
        except Exception as exc:
            record = {"image": str(path), "status": "error", "is_defect": None, "message": str(exc)}
        records.append(record)
        print(f"[{index}/{len(paths)}] {relative}: {record['status']} - {record['message']}")
    report = {
        "schema_version": 1,
        "coordinate_system": "original image pixels; origin top-left; x right; y down",
        "weights": str(Path(weights).resolve()),
        "continuity_settings": asdict(settings),
        "segmentation_settings": asdict(segmentation_settings),
        "counts": dict(Counter(record["status"] for record in records)),
        "images": records,
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return int(any(record["status"] != "ok" for record in records))
