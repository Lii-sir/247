"""推理交付包唯一入口：源码来自 src/，默认结果写入 results/。"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> Path:
    source = PACKAGE_DIR / "src"
    if not (source / "ccd_efficientad" / "inference.py").is_file():
        raise FileNotFoundError("交付包缺少源码；请在原项目运行 tools/build_inference_delivery.py。")
    sys.path.insert(0, str(source))
    from ccd_efficientad.inference import main as infer

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(argument == "--output-dir" or argument.startswith("--output-dir=")
               for argument in arguments):
        arguments.extend(["--output-dir", str(PACKAGE_DIR / "results")])
    return infer(arguments)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)

