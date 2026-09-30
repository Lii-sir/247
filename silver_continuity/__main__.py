"""python -m silver_continuity opens only the continuity workflow."""

import sys


def main(argv=None):
    from silver_inspection.__main__ import main as run
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--mode" or arg.startswith("--mode=") for arg in args):
        raise SystemExit("本入口仅用于断连检测；溢出请运行 python -m silver_overflow")
    return run(["--mode", "continuity", *args])


if __name__ == "__main__":
    raise SystemExit(main())
