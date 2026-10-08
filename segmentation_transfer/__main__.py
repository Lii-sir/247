"""python -m segmentation_transfer：GUI 标定/预览，或 CLI 单对图片映射。"""

import argparse
from dataclasses import asdict
from pathlib import Path

import cv2 as cv

from common.segmentation.models import SegmentationSettings
from common.matching import MatchSettings
from common.paths import PROJECT_ROOT

from .calibration_io import load_calibration
from .models import MappingSettings

def main(argv=None):
    parser = argparse.ArgumentParser(description="双模板对应点标定 → A1 分割 → B1 掩膜映射")
    parser.add_argument("--calibration", type=Path, help="GUI 保存的标定 JSON，路径相对 JSON 所在目录解析")
    parser.add_argument("--image-a", type=Path, help="图片 A1")
    parser.add_argument("--image-b", type=Path, help="图片 B1")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights/best.pt")
    parser.add_argument("--output", type=Path, help="新建结果目录；设置后无界面运行")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--point-threshold", type=float, default=3.0, help="人工点 RANSAC 阈值，单位为 B 模板像素")
    args = parser.parse_args(argv)
    if args.output is not None and not all((args.calibration, args.image_a, args.image_b)):
        parser.error("--output 需要 --calibration、--image-a 和 --image-b")
    try:
        segmentation = SegmentationSettings(confidence=args.conf, image_size=args.imgsz, device=args.device)
        mapping = MappingSettings(ransac_threshold=args.point_threshold)
        calibration = load_calibration(args.calibration) if args.calibration else None
        if args.output is None:
            from .app import run_app
            return run_app(args.weights, calibration, args.image_a, args.image_b, segmentation, mapping)
        from .pipeline import LazyYoloSegmenter, TransferPipeline
        from .export import export_result
        if args.output.exists():
            raise ValueError("输出目录已存在，请选择新目录")
        pipeline = TransferPipeline(calibration, mapping_settings=mapping)
        result = pipeline.run(args.image_a, args.image_b, LazyYoloSegmenter(args.weights), segmentation, print)
        metadata = {"weights": str(args.weights.resolve()), "segmentation": asdict(segmentation),
                    "mapping": asdict(mapping), "matching": asdict(MatchSettings())}
        output = export_result(args.output, result, metadata)
        print(f"已映射 {len(result.instances)} 个实例，结果：{output}")
        for warning in result.warnings:
            print(f"提示：{warning}")
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, cv.error) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
