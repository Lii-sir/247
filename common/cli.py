"""Shared input/inference arguments only; no feature routing or environment sync."""

from pathlib import Path

from .paths import DEFAULT_SOURCE, DEFAULT_WEIGHTS


def add_input_arguments(parser):
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--recursive", action="store_true", help="扫描子文件夹")
    parser.add_argument("--export", type=Path, metavar="DIRECTORY", help="批量导出，不打开界面")


def add_segmentation_arguments(parser):
    parser.add_argument("--conf", type=float, default=0.25, help="分割置信度阈值，默认 0.25")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值，默认 0.7")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸，32 的倍数，默认 640")
    parser.add_argument("--device", default="0", help="CUDA GPU 编号，默认 0；显式 cpu 用于调试")


def segmentation_settings(args):
    from .segmentation.models import SegmentationSettings
    return SegmentationSettings(confidence=args.conf, iou=args.iou, image_size=args.imgsz, device=args.device)

