"""Independent overflow CLI; never imports the continuity workflow."""

import argparse
from pathlib import Path

from common.cli import add_input_arguments, add_segmentation_arguments, segmentation_settings
from .geometry import OverflowSettings


def main(argv=None):
    parser = argparse.ArgumentParser(description="银浆溢出检测：分割 → 模板边界映射 → 框外判断")
    add_input_arguments(parser)
    add_segmentation_arguments(parser)
    parser.add_argument("--silver-class", default="silver")
    parser.add_argument("--calibration", type=Path, help="允许框的标定 JSON；批量检测必填")
    parser.add_argument("--tolerance-px", type=float, default=0, help="允许越界距离，目标原图 px")
    parser.add_argument("--min-area-px", type=int, default=1, help="最小溢出连通域面积")
    args = parser.parse_args(argv)
    if args.export is not None and args.calibration is None:
        parser.error("批量溢出检测需要 --calibration；界面可创建允许框")
    try:
        settings = OverflowSettings(silver_class=args.silver_class, tolerance_px=args.tolerance_px,
                                    min_area_px=args.min_area_px)
        inference = segmentation_settings(args)
        from .calibration import load_calibration
        calibration = load_calibration(args.calibration) if args.calibration else None
        if args.export is not None:
            from .export import export_batch
            return export_batch(args.weights, args.source, args.export, calibration, settings, inference,
                                args.recursive, args.calibration)
        from .app import run_app
        return run_app(args.weights, args.source, settings, inference, calibration, args.recursive)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
