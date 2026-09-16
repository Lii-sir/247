"""从参考图中检测目标圆，并生成可供 EfficientAD 使用的默认圆形 mask。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from circle_mask import (
    category_config,
    detect_circle,
    effective_mask_radius,
    load_configs,
    make_mask,
    overlay_diagnostics,
    read_rgb,
)


def parse_roi(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI 必须是 x,y,w,h 四个数字") from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI 必须是 x,y,w,h 四个数字")
    return values


def write_color(path: Path, image_rgb: np.ndarray) -> None:
    """支持 Windows/中文路径的 RGB 图片写入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"无法编码图片：{path}")
    encoded.tofile(path)


def write_mask(path: Path, mask: np.ndarray) -> None:
    """以无损单通道 PNG 保存 0/255 二值 mask。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".png":
        raise ValueError("mask 输出必须使用 .png 后缀，避免有损压缩改变二值区域。")
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", binary)
    if not ok:
        raise ValueError(f"无法编码 mask：{path}")
    encoded.tofile(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检测参考图中的圆，并输出圆内白色、圆外黑色的原图尺寸二值 mask。"
    )
    parser.add_argument("--input", type=Path, required=True, help="用于定位圆的参考图片")
    parser.add_argument("--output-dir", type=Path, required=True, help="mask 和诊断结果输出目录")
    parser.add_argument("--circle-config", type=Path, help="包含 CCD1、CCD2 等配置的 JSON")
    parser.add_argument("--category", default="default", help="使用配置中的型号，例如 CCD1")
    parser.add_argument("--mask-name", default="default_mask.png", help="二值 mask 文件名，必须为 PNG")
    parser.add_argument("--roi", type=parse_roi, help="覆盖配置中的归一化 ROI：x,y,w,h")
    parser.add_argument(
        "--detection-method",
        choices=["outer_inner_ring", "hybrid", "dark_contour", "hough"],
    )
    parser.add_argument("--circle-target", choices=["inner", "outer", "best_contrast"])
    parser.add_argument("--group-target", choices=["best_score", "largest", "strongest"])
    parser.add_argument("--dark-threshold-offset", type=float)
    parser.add_argument("--morph-kernel", type=int)
    parser.add_argument("--min-axis-ratio", type=float)
    parser.add_argument("--min-contour-score", type=float)
    parser.add_argument("--ransac-iterations", type=int)
    parser.add_argument("--ransac-tolerance-ratio", type=float)
    parser.add_argument("--min-ransac-inlier-ratio", type=float)
    parser.add_argument("--min-angular-coverage", type=float)
    parser.add_argument("--inner-radius-min-ratio", type=float)
    parser.add_argument("--inner-radius-max-ratio", type=float)
    parser.add_argument("--black-ring-width-ratio", type=float)
    parser.add_argument("--min-black-ring-coverage", type=float)
    parser.add_argument("--min-inner-angular-coverage", type=float)
    parser.add_argument("--dp", type=float)
    parser.add_argument("--param1", type=float)
    parser.add_argument("--param2", type=float)
    parser.add_argument("--min-dist-ratio", type=float)
    parser.add_argument("--blur-kernel", type=int)
    parser.add_argument("--min-radius-ratio", type=float)
    parser.add_argument("--max-radius-ratio", type=float)
    parser.add_argument("--mask-radius-scale", type=float)
    parser.add_argument("--mask-margin", type=int)
    parser.add_argument(
        "--allow-circle-cross-roi",
        action="store_true",
        help="允许候选圆跨出 ROI；默认要求整个候选圆位于 ROI 内",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"参考图片不存在：{input_path}")

    configs = load_configs(args.circle_config.resolve()) if args.circle_config else {}
    params = category_config(configs, args.category).copy()
    overrides = {
        "roi": args.roi,
        "detection_method": args.detection_method,
        "circle_target": args.circle_target,
        "group_target": args.group_target,
        "dark_threshold_offset": args.dark_threshold_offset,
        "morph_kernel": args.morph_kernel,
        "min_axis_ratio": args.min_axis_ratio,
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
    params.setdefault("require_circle_inside_roi", True)
    if args.allow_circle_cross_roi:
        params["require_circle_inside_roi"] = False

    output_dir = args.output_dir.resolve()
    mask_path = output_dir / args.mask_name
    if mask_path.suffix.lower() != ".png" or Path(args.mask_name).name != args.mask_name:
        raise ValueError("--mask-name 必须是当前输出目录中的 .png 文件名，不能包含子目录。")

    image = read_rgb(input_path)
    result = detect_circle(image, params)
    selected = result["selected"]
    if not selected.get("enabled", True):
        raise ValueError("当前配置关闭了圆检测，无法生成圆形 mask。")
    mask = make_mask(image.shape[:2], selected, params)
    if not np.any(mask):
        raise ValueError("检测结果生成了空 mask。")

    masked_preview = image.copy()
    masked_preview[mask > 0] = 255
    overlay = overlay_diagnostics(image, result, params)
    overlay_path = output_dir / "circle_overlay.jpg"
    preview_path = output_dir / "circle_white_preview.jpg"
    metadata_path = output_dir / "circle_mask.json"
    write_mask(mask_path, mask)
    write_color(overlay_path, overlay)
    write_color(preview_path, masked_preview)

    metadata = {
        "input": str(input_path),
        "category": args.category,
        "params": params,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "mask_semantics": "255=ignored circle, 0=valid image region",
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "white_preview_path": str(preview_path),
        "mask_radius": effective_mask_radius(selected, params),
        "mask_pixel_count": int(np.count_nonzero(mask)),
        "mask_area_ratio": float(np.count_nonzero(mask) / mask.size),
        **result,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"圆检测完成：center=({selected['center_x']}, {selected['center_y']}), "
        f"detected_r={selected['radius']}, mask_r={metadata['mask_radius']}, "
        f"detection={result['detection_ms']:.2f} ms"
    )
    print(f"二值 mask：{mask_path}")
    print(f"检测叠加图：{overlay_path}")
    print(f"白色填充预览：{preview_path}")
    print(f"检测记录：{metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
