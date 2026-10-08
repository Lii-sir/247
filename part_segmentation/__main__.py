"""python -m part_segmentation：默认桌面预览，--export 切换为批量导出。"""

import argparse
from common.cli import add_input_arguments, add_segmentation_arguments, segmentation_settings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="独立 YOLO 部件分割展示（不执行匹配找点）")
    add_input_arguments(parser)
    add_segmentation_arguments(parser)
    args = parser.parse_args(argv)
    try:
        settings = segmentation_settings(args)
        if args.export is not None:
            from .export import export_batch
            return export_batch(args.weights, args.source, args.export, settings, args.recursive)
        from .app import run_app
        return run_app(args.weights, args.source, settings, args.recursive)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
