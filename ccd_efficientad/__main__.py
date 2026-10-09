"""统一 CLI：python -m ccd_efficientad <命令>，或 python run.py <命令>。"""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "inspect": ("ccd_efficientad.cli", True, "检查数据与快照"),
    "train": ("ccd_efficientad.cli", True, "训练、校准并评估"),
    "evaluate": ("ccd_efficientad.cli", True, "读取权重评估"),
    "predict": ("ccd_efficientad.cli", True, "原单图预测入口"),
    "infer": ("ccd_efficientad.inference", False, "单图/文件夹推理"),
    "branch-maps": ("tools.diagnostics.export_branch_maps", False, "导出各分支异常图"),
    "generate-mask": ("tools.masks.generate_circle_mask", False, "从参考图生成 mask"),
    "detect-circle": ("tools.masks.detect_background_circle", False, "检测图片或文件夹中的圆"),
    "mask-gui": ("tools.masks.circle_mask_gui", False, "打开 mask 图形界面"),
    "download-visa": ("tools.data.download_visa_aux", False, "下载整理 VisA 辅助数据"),
    "make-dataset": ("tools.data.make_anomaly_dataset", False, "划分数据并合成异常图片"),
}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        print("用法：python run.py <命令> [参数]\n      python -m ccd_efficientad <命令> [参数]\n")
        for command, (_, _, description) in COMMANDS.items():
            print(f"  {command:16s} {description}")
        print("\n命令参数：python run.py <命令> --help")
        return 0
    command = arguments[0]
    if command not in COMMANDS:
        print(f"未知命令：{command}；使用 --help 查看命令列表。", file=sys.stderr)
        return 2
    module_name, keep_command, _ = COMMANDS[command]
    previous = sys.argv
    sys.argv = [previous[0], *(arguments if keep_command else arguments[1:])]
    try:
        module = importlib.import_module(module_name)
        result = module.main()
        return result if isinstance(result, int) else 0
    finally:
        sys.argv = previous


def entrypoint() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    entrypoint()
