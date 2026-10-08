"""无界面批量导出；每张图独立处理，损坏图片不会中断整批。"""

import json
from dataclasses import asdict
from pathlib import Path

from common.image_io import collect_images, write_image
from common.segmentation.inference import PartSegmenter
from common.segmentation.models import SegmentationSettings
from common.segmentation.visualization import render_comparison, render_overlay


def export_batch(weights: Path, source: Path, output: Path, settings: SegmentationSettings,
                 recursive: bool = False) -> int:
    images = collect_images(source, recursive)
    source, output = source.resolve(), output.resolve()
    # 禁止污染数据集，也避免下次递归扫描把输出再次作为输入。
    source_root = source if source.is_dir() else source.parent
    if output.is_relative_to(source_root):
        raise ValueError("导出目录不能位于输入图片目录内，请使用独立的 outputs 目录")
    segmenter = PartSegmenter(weights)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    failed = 0
    for index, path in enumerate(images, start=1):
        relative = path.relative_to(source_root)
        record = {"image": str(path), "status": "error"}
        try:
            result = segmenter.predict(path, settings)
            # 保留输入扩展名及子目录，避免 a.jpg/a.png 或同名子目录碰撞。
            prefix = output / relative
            write_image(prefix.with_name(prefix.name + ".overlay.png"), render_overlay(result))
            write_image(prefix.with_name(prefix.name + ".comparison.jpg"), render_comparison(result))
            record.update(status="ok", elapsed_ms=result.elapsed_ms, count=len(result.segments),
                          image_size=[result.image.shape[1], result.image.shape[0]],
                          segments=[{"id": i, "class_id": s.class_id, "class_name": s.class_name,
                                     "confidence": s.confidence, "box_xyxy": s.box, "area_pixels": s.area}
                                    for i, s in enumerate(result.segments, start=1)])
            print(f"[{index}/{len(images)}] {relative}: {len(result.segments)} instances")
        except Exception as exc:
            failed += 1
            record["error"] = str(exc)
            print(f"[{index}/{len(images)}] {relative}: ERROR - {exc}")
        records.append(record)
    summary = {"weights": str(Path(weights).resolve()), "settings": asdict(settings), "images": records}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Output: {output} | success={len(images) - failed}, failed={failed}")
    return 1 if failed else 0

