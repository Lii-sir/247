"""Independent CUDA segmentation, silver continuity and overflow entry points."""

import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    modes = {"--silver": "continuity", "--overflow": "overflow", "--segment": "segment"}
    selected = [flag for flag in modes if flag in args]
    if len(selected) > 1:
        raise SystemExit("请分别启动 --segment、--silver 或 --overflow；它们是独立的窗口")
    if selected:
        flag = selected[0]
        args.remove(flag)
        if any(arg == "--mode" or arg.startswith("--mode=") for arg in args):
            raise SystemExit("快捷入口不能同时指定 --mode，请选择一种启动方式")
        if flag != "--silver":
            args = ["--mode", modes[flag], *args]
        from silver_inspection.__main__ import main as run
    else:
        from part_segmentation.__main__ import main as run
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
