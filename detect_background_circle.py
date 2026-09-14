"""独立验证圆检测、同心圆选择和白色 mask 的脚本。

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


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"输入路径不存在：{path}")


def parse_roi(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI 必须是 x,y,w,h 四个数字")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="验证工件圆检测、同心圆选择和白色 mask")
    parser.add_argument("--input", type=Path, required=True, help="单张图片或图片目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--circle-config", type=Path, help="按工件类型配置的 JSON 文件")
    parser.add_argument("--category", default="default", help="使用 circle-config 中的类别；默认 default")
    parser.add_argument("--roi", type=parse_roi, help="覆盖配置中的 ROI：x,y,w,h")
    parser.add_argument("--circle-target", choices=["inner", "outer", "best_contrast"], help="覆盖配置中的目标圆选择规则")
    parser.add_argument("--group-target", choices=["largest", "strongest"], help="覆盖配置中的目标圆组选择规则")
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
        "circle_target": args.circle_target,
        "group_target": args.group_target,
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

    images = iter_images(args.input.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
            stem = path.stem
            write_image(args.output_dir / f"{stem}_overlay.jpg", overlay)
            write_image(args.output_dir / f"{stem}_masked.jpg", masked)
            write_image(args.output_dir / f"{stem}_mask.png", np.repeat(mask[:, :, None], 3, axis=2))
            metadata = {"image": str(path), "params": params, **result}
            (args.output_dir / f"{stem}_circle.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if selected.get("enabled", True):
                print(f"[OK] {path.name}: target={selected['circle_target']} center=({selected['center_x']},{selected['center_y']}) r={selected['radius']} candidates={selected['candidate_count']} detection={result['detection_ms']:.2f} ms")
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
