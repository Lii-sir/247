"""python -m part_segmentation：默认桌面预览，--export 切换为批量导出。"""

import argparse
from pathlib import Path

from .models import SegmentationSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="独立 YOLO 部件分割展示（不执行匹配找点）")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值，默认 0.7")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸（32 的倍数），默认 640")
    parser.add_argument("--device", default="cpu", help="cpu 或 GPU 编号，例如 0")
    parser.add_argument("--recursive", action="store_true", help="扫描子文件夹")
    parser.add_argument("--export", type=Path, metavar="DIRECTORY", help="批量导出，不打开界面")
    args = parser.parse_args(argv)
    try:
        settings = SegmentationSettings(args.conf, args.iou, args.imgsz, args.device)
        if args.export is not None:
            from .export import export_batch
            return export_batch(args.weights, args.source, args.export, settings, args.recursive)
        from .app import run_app
        return run_app(args.weights, args.source, settings, args.recursive)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
