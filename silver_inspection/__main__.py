"""python -m silver_inspection: chip-ring silver continuity inspection."""

import argparse
from pathlib import Path

from part_segmentation.models import SegmentationSettings
from .continuity import ContinuitySettings
from .geometry import OverflowSettings

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="分割预览、银浆断连检测、银浆溢出检测")
    parser.add_argument("--mode", choices=("continuity", "overflow", "segment"), default="continuity",
                        help="独立功能：continuity 断连，overflow 溢出，segment 只看分割")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/best.pt")
    parser.add_argument("--source", type=Path, default=ROOT / "datasets")
    parser.add_argument("--silver-class", default="silver")
    parser.add_argument("--chip-class", default="chip")
    parser.add_argument("--occlusion-classes", default="thin,bond,wire", help="被遮挡、从判定分母排除的类别，逗号分隔")
    parser.add_argument("--outward-px", type=int, default=20, help="chip 外扩形成检查环带的长度，单位 px")
    parser.add_argument("--sectors", type=int, default=72, help="360° 检查扇区数量")
    parser.add_argument("--min-silver-px", type=int, default=3, help="每个扇区判定有银浆所需的最少像素数")
    parser.add_argument("--min-coverage", type=float, default=0.01, help="扇区 silver / 有效环带像素的最低比例")
    parser.add_argument("--min-valid-px", type=int, default=1, help="扇区参与判定所需的最少有效像素数")
    parser.add_argument("--occlusion-dilation-px", type=int, default=0, help="thin/bond 遮挡掩膜额外膨胀像素")
    parser.add_argument("--calibration", type=Path, help="溢出检测的矩形框标定 JSON")
    parser.add_argument("--tolerance-px", type=float, default=0, help="溢出检测允许越界距离")
    parser.add_argument("--min-area-px", type=int, default=1, help="溢出检测最小连通域面积")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA 设备，默认 0；仅调试时使用 cpu")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--export", type=Path, metavar="DIRECTORY", help="无界面批量检测并导出结果")
    args = parser.parse_args(argv)
    try:
        if args.mode == "segment":
            from part_segmentation.__main__ import main as segmentation_main
            forwarded = ["--weights", str(args.weights), "--source", str(args.source),
                         "--conf", str(args.conf), "--iou", str(args.iou),
                         "--imgsz", str(args.imgsz), "--device", args.device]
            if args.recursive:
                forwarded.append("--recursive")
            if args.export:
                forwarded.extend(("--export", str(args.export)))
            return segmentation_main(forwarded)
        if args.mode == "overflow":
            if args.export and not args.calibration:
                parser.error("批量溢出检测需要 --calibration；界面可直接打开并创建四点框")
            from .calibration import load_calibration
            calibration = load_calibration(args.calibration) if args.calibration else None
            settings = OverflowSettings(args.silver_class, args.tolerance_px, args.min_area_px)
            segmentation = SegmentationSettings(args.conf, args.iou, args.imgsz, args.device)
            if args.export:
                from .export import export_batch
                return export_batch(args.weights, args.source, args.export, calibration, settings, segmentation,
                                    args.recursive, args.calibration)
            from .app import run_app
            return run_app(args.weights, args.source, settings, segmentation, calibration, args.recursive)
        occlusions = tuple(value.strip() for value in args.occlusion_classes.replace("，", ",").split(",") if value.strip())
        settings = ContinuitySettings(args.silver_class, args.chip_class, occlusions, args.outward_px,
                                      args.sectors, args.min_silver_px, args.min_coverage,
                                      args.min_valid_px, args.occlusion_dilation_px)
        segmentation = SegmentationSettings(args.conf, args.iou, args.imgsz, args.device)
        if args.export:
            from .continuity_export import export_continuity_batch
            return export_continuity_batch(args.weights, args.source, args.export, settings, segmentation, args.recursive)
        from .continuity_app import run_continuity_app
        return run_continuity_app(args.weights, args.source, settings, segmentation, args.recursive)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
