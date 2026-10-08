"""Independent continuity CLI; never imports the overflow workflow."""

import argparse

from common.cli import add_input_arguments, add_segmentation_arguments, segmentation_settings
from .geometry import ContinuitySettings


def main(argv=None):
    parser = argparse.ArgumentParser(description="银浆断连检测：分割预览 → chip 矩形环带 → 扇区判断")
    add_input_arguments(parser)
    add_segmentation_arguments(parser)
    parser.add_argument("--silver-class", default="silver")
    parser.add_argument("--chip-class", default="chip")
    parser.add_argument("--occlusion-classes", default="thin,bond,wire", help="忽略的遮挡类别，逗号分隔")
    parser.add_argument("--outward-px", type=int, default=20, help="chip 外接矩形外扩距离，原图 px")
    parser.add_argument("--sectors", type=int, default=72, help="360° 扇区数量")
    parser.add_argument("--min-silver-px", type=int, default=3, help="每个有效扇区最少 silver 像素")
    parser.add_argument("--min-coverage", type=float, default=0.01, help="silver / 有效像素的最低比例")
    parser.add_argument("--min-valid-px", type=int, default=1, help="每扇区最少有效像素")
    parser.add_argument("--min-visible-ratio", type=float, default=0.1, help="遮挡时扇区最小可见比例")
    parser.add_argument("--occlusion-dilation-px", type=int, default=0, help="遮挡掩膜额外膨胀像素")
    args = parser.parse_args(argv)
    try:
        settings = ContinuitySettings(
            silver_class=args.silver_class, chip_class=args.chip_class,
            occlusion_classes=tuple(s.strip() for s in args.occlusion_classes.replace("，", ",").split(",") if s.strip()),
            outward_length_px=args.outward_px, sector_count=args.sectors,
            min_sector_silver_px=args.min_silver_px, min_sector_coverage=args.min_coverage,
            min_valid_sector_px=args.min_valid_px, min_visible_sector_ratio=args.min_visible_ratio,
            occlusion_dilation_px=args.occlusion_dilation_px,
        )
        inference = segmentation_settings(args)
        if args.export is not None:
            from .export import export_continuity_batch
            return export_continuity_batch(args.weights, args.source, args.export, settings, inference, args.recursive)
        from .app import run_continuity_app
        return run_continuity_app(args.weights, args.source, settings, inference, args.recursive)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
