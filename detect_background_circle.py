"""独立验证圆检测、偏心内圆选择和白色 mask 的脚本。

示例：
    uv run python detect_background_circle.py --input "图片目录" \
        --output-dir circle_test --circle-target inner \
        --roi "0.55,0.20,0.35,0.50"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from circle_mask import category_config, detect_circle, load_configs, make_mask, overlay_diagnostics, read_rgb


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def write_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(ext, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"无法编码图片：{path}")
    encoded.tofile(path)


def iter_images(path: Path, *, exclude: Path | None = None) -> list[Path]:
    """列出输入图片；输出目录位于输入树中时明确排除，避免重复处理。"""
    if path.is_file():
        return [path]
    if path.is_dir():
        excluded = exclude.resolve() if exclude is not None else None
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in IMAGE_SUFFIXES
            and not (excluded is not None and candidate.resolve().is_relative_to(excluded))
        )
    raise FileNotFoundError(f"输入路径不存在：{path}")


def output_stem(input_path: Path, source: Path, output_dir: Path) -> Path:
    """在输出目录中保留递归输入的相对层级，避免同名文件覆盖。"""
    relative_parent = source.relative_to(input_path).parent if input_path.is_dir() else Path()
    return output_dir / relative_parent / source.stem


def parse_roi(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI 必须是 x,y,w,h 四个数字")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="验证工件圆检测、偏心内圆选择和白色 mask")
    parser.add_argument("--input", type=Path, required=True, help="单张图片或图片目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--circle-config", type=Path, help="按工件类型配置的 JSON 文件")
    parser.add_argument("--category", default="default", help="使用 circle-config 中的类别；默认 default")
    parser.add_argument("--roi", type=parse_roi, help="覆盖配置中的 ROI：x,y,w,h")
    parser.add_argument(
        "--detection-method", choices=["outer_inner_ring", "hybrid", "dark_contour", "hough"],
        help="outer_inner_ring=先找外圆，再按黑环寻找允许偏心的内圆",
    )
    parser.add_argument("--circle-target", choices=["inner", "outer", "best_contrast"], help="覆盖配置中的目标圆选择规则")
    parser.add_argument("--group-target", choices=["best_score", "largest", "strongest"], help="覆盖配置中的目标圆组选择规则")
    parser.add_argument("--dark-threshold-offset", type=float, help="Otsu 暗区阈值偏移，正数保留更多暗区")
    parser.add_argument("--morph-kernel", type=int, help="暗区轮廓开闭运算窗口（正奇数）")
    parser.add_argument("--min-axis-ratio", type=float, help="拟合椭圆最小短轴/长轴比例")
    parser.add_argument("--outer-min-axis-ratio", type=float, help="外圆拟合椭圆最小短轴/长轴比例")
    parser.add_argument("--min-contour-score", type=float, help="暗区轮廓候选最低综合置信度")
    parser.add_argument("--ransac-iterations", type=int)
    parser.add_argument("--ransac-tolerance-ratio", type=float, help="RANSAC 内点距离容差/半径")
    parser.add_argument("--min-ransac-inlier-ratio", type=float)
    parser.add_argument("--min-angular-coverage", type=float)
    parser.add_argument("--inner-radius-min-ratio", type=float, help="内圆半径/外圆半径下限")
    parser.add_argument("--inner-radius-max-ratio", type=float, help="内圆半径/外圆半径上限")
    parser.add_argument("--black-ring-width-ratio", type=float, help="黑环采样宽度/外圆半径")
    parser.add_argument("--min-black-ring-coverage", type=float, help="黑环最低角度覆盖率")
    parser.add_argument("--min-inner-angular-coverage", type=float, help="内圆最低圆弧覆盖率")
    parser.add_argument("--dp", type=float, help="覆盖 Hough dp")
    parser.add_argument("--param1", type=float, help="覆盖 Canny 高阈值")
    parser.add_argument("--param2", type=float, help="覆盖 Hough 累加器阈值")
    parser.add_argument("--min-dist-ratio", type=float, help="覆盖候选圆心最小距离/ROI短边")
    parser.add_argument("--blur-kernel", type=int, help="覆盖中值滤波窗口")
    parser.add_argument("--min-radius-ratio", type=float)
    parser.add_argument("--max-radius-ratio", type=float)
    parser.add_argument("--mask-radius-scale", type=float)
    parser.add_argument("--mask-margin", type=int)
    args = parser.parse_args()

    params = category_config(load_configs(args.circle_config), args.category).copy()
    overrides = {
        "roi": args.roi,
        "detection_method": args.detection_method,
        "circle_target": args.circle_target,
        "group_target": args.group_target,
        "dark_threshold_offset": args.dark_threshold_offset,
        "morph_kernel": args.morph_kernel,
        "min_axis_ratio": args.min_axis_ratio,
        "outer_min_axis_ratio": args.outer_min_axis_ratio,
        "min_contour_score": args.min_contour_score,
        "ransac_iterations": args.ransac_iterations,
        "ransac_tolerance_ratio": args.ransac_tolerance_ratio,
        "min_ransac_inlier_ratio": args.min_ransac_inlier_ratio,
        "min_angular_coverage": args.min_angular_coverage,
        "inner_radius_min_ratio": args.inner_radius_min_ratio,
        "inner_radius_max_ratio": args.inner_radius_max_ratio,
        "black_ring_width_ratio": args.black_ring_width_ratio,
        "min_black_ring_coverage": args.min_black_ring_coverage,
        "min_inner_angular_coverage": args.min_inner_angular_coverage,
        "dp": args.dp,
        "param1": args.param1,
        "param2": args.param2,
        "min_dist_ratio": args.min_dist_ratio,
        "blur_kernel": args.blur_kernel,
        "min_radius_ratio": args.min_radius_ratio,
        "max_radius_ratio": args.max_radius_ratio,
        "mask_radius_scale": args.mask_radius_scale,
        "mask_margin": args.mask_margin,
    }
    params.update({key: value for key, value in overrides.items() if value is not None})
    params.setdefault("enabled", True)

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if input_path.is_dir() and output_dir == input_path:
        raise ValueError("输入目录和输出目录不能相同；请为检测结果指定单独的输出目录。")
    images = iter_images(input_path, exclude=output_dir)
    if not images:
        raise ValueError(f"输入路径中没有受支持的图片：{input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    timings = []
    for path in images:
        try:
            image = read_rgb(path)
            result = detect_circle(image, params)
            timings.append(float(result["detection_ms"]))
            selected = result["selected"]
            mask = make_mask(image.shape[:2], selected, params)
            masked = image.copy()
            masked[mask > 0] = 255
            overlay = overlay_diagnostics(image, result, params)
            destination = output_stem(input_path, path, output_dir)
            write_image(destination.with_name(f"{destination.name}_overlay.jpg"), overlay)
            write_image(destination.with_name(f"{destination.name}_masked.jpg"), masked)
            write_image(
                destination.with_name(f"{destination.name}_mask.png"),
                np.repeat(mask[:, :, None], 3, axis=2),
            )
            metadata = {"image": str(path), "params": params, **result}
            metadata_path = destination.with_name(f"{destination.name}_circle.json")
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if selected.get("enabled", True):
                print(
                    f"[OK] {path.name}: detector={result['detection'].get('detector_used')} "
                    f"target={selected['circle_target']} center=({selected['center_x']},{selected['center_y']}) "
                    f"r={selected['radius']} confidence={selected.get('candidate_score', 0.0):.3f} "
                    f"candidates={selected['candidate_count']} detection={result['detection_ms']:.2f} ms"
                )
            else:
                print(f"[OK] {path.name}: mask disabled detection={result['detection_ms']:.2f} ms")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {path}: {exc}")
    summary = ""
    if timings:
        summary = f"，检测平均 {sum(timings) / len(timings):.2f} ms/张，总计 {sum(timings):.2f} ms"
    print(f"完成：共 {len(images)} 张，成功 {len(images) - failed} 张，失败 {failed} 张{summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
