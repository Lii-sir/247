"""Render the exact configured silver-overflow boundary without running EfficientAD/3D.

This is a visualization client only. It does not change algorithm configuration,
model weights, thresholds, or the production inspection functions.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
import webbrowser
from datetime import datetime
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
YELLOW, CYAN, PURPLE, RED = (255, 215, 60), (73, 226, 226), (187, 120, 255), (255, 67, 88)


def expansion_region(chip_mask: np.ndarray, distance: int, min_area: int):
    """Reproduce production's clipped, axis-aligned, half-open rectangles exactly."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(chip_mask.astype(np.uint8), 8)
    height, width = chip_mask.shape
    allowed = np.zeros(chip_mask.shape, dtype=bool)
    rectangles = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area < min_area:
            continue
        box = [max(0, x - distance), max(0, y - distance),
               min(width, x + w + distance), min(height, y + h + distance)]
        allowed[box[1]:box[3], box[0]:box[2]] = True
        rectangles.append({"chip_box": [x, y, x + w, y + h], "allowed_box": box})
    return allowed, rectangles


def font(size):
    for path in (ROOT / "src/Bridge.Demo/simhei.ttf", Path("C:/Windows/Fonts/msyh.ttc")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def overlay(rgb, mask, color, opacity):
    result = rgb.copy()
    result[mask] = ((1 - opacity) * result[mask] + opacity * np.asarray(color)).astype(np.uint8)
    return result


def contours(rgb, mask, color, thickness=2):
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, found, -1, color, thickness, cv2.LINE_AA)


def draw_boundaries(rgb, rectangles, polygons):
    for item in rectangles:
        x1, y1, x2, y2 = item["chip_box"]
        cv2.rectangle(rgb, (x1, y1), (x2 - 1, y2 - 1), CYAN, 2, cv2.LINE_AA)
        x1, y1, x2, y2 = item["allowed_box"]
        cv2.rectangle(rgb, (x1, y1), (x2 - 1, y2 - 1), YELLOW, 3, cv2.LINE_AA)
    for item in polygons:
        cv2.polylines(rgb, [np.rint(item["points"]).astype(np.int32)],
                      item["mode"] == "polygon", YELLOW, 3, cv2.LINE_AA)


def compose_panels(panels, titles, subtitles, top, bottom):
    width, image_height, margin, gap = 520, 620, 24, 18
    canvas = Image.new("RGB", (margin * 2 + width * 3 + gap * 2, 855), (13, 20, 30))
    pen = ImageDraw.Draw(canvas)
    pen.text((margin, 20), top, fill=(238, 242, 248), font=font(27))
    for index, (pixels, title, subtitle) in enumerate(zip(panels, titles, subtitles)):
        x = margin + index * (width + gap)
        pen.text((x, 78), title, fill=(238, 242, 248), font=font(23))
        pen.text((x, 114), subtitle, fill=(170, 184, 201), font=font(16))
        image = Image.fromarray(pixels)
        image.thumbnail((width, image_height), Image.Resampling.LANCZOS)
        pen.rounded_rectangle((x, 145, x + width, 145 + image_height), 8, fill=(22, 32, 47))
        canvas.paste(image, (x + (width - image.width) // 2, 145 + (image_height - image.height) // 2))
    pen.text((margin, 785), "黄色：允许边界    青色：芯片外接框    紫色：银浆分割    红色：判定为溢出", fill=(238, 242, 248), font=font(21))
    pen.text((margin, 821), bottom, fill=(170, 184, 201), font=font(16))
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize the actual silver overflow boundary on a sample image.")
    parser.add_argument("--image", type=Path, default=ROOT / "demodatasets/2D/001.png")
    parser.add_argument("--config", type=Path, default=ROOT / "config/2D/config.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    overflow_config = config["silver_overflow_2d"]["parameters"]
    boundary_config = overflow_config["boundary"]
    mode = str(boundary_config.get("mode", "polygon")).casefold()

    sys.path.insert(0, str(ROOT / "python"))
    from algorithm.detect_board import DetectBoard, _mask_geometry
    from ultralytics import YOLO
    silver = import_module("algorithm.2D.silver_inspection")

    image = cv2.imdecode(np.fromfile(str(args.image.resolve()), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {args.image}")
    # Use the board detector's actual crop and instance extraction, but skip unrelated AD scoring.
    detector = DetectBoard(lambda message: print(message, file=sys.stderr))
    detector.min_component_area = int(config["common_2d"]["parameters"].get("mask", {}).get("min_component_area", 100))
    crop_config = dict(config["common_2d"]["parameters"]["crop"])
    crop_config["rectangle_roi_offsets"] = config["wire_bending_2d"]["parameters"].get("roi_offsets", [])
    detector.circle_crop_config = detector._parse_circle_crop_config(crop_config)
    detector.yolo_parameters = {k: v for k, v in config["common_2d"]["parameters"]["yolo"].items()
                                if k in {"imgsz", "conf", "iou", "max_det", "device", "half", "end2end"}}
    detector._validate_gpu_requirement()
    detector.model = YOLO(str((config_path.parent / config["model_path"]["YOLO"]).resolve()))
    crop, crop_box, circle, _ = detector._stream_crop(image)
    prediction = detector._predict_batch([crop])[0]
    instances = detector._extract_instances(prediction, crop.shape[:2])
    chip_mask, _ = silver._union_masks(instances, (str(boundary_config.get("chip_class", "chip")),), crop.shape[:2])
    silver_mask, _ = silver._union_masks(instances, (str(overflow_config.get("silver_class", "silver")),), crop.shape[:2])
    settings = silver.overflow_settings(overflow_config)
    rectangles, polygons = [], []
    if mode in {"chip_expansion", "chip_expand", "chip"}:
        distance = int(boundary_config.get("outward_length_px", 20))
        min_area = int(boundary_config.get("min_chip_area_px", 100))
        allowed, rectangles = expansion_region(chip_mask, distance, min_area)
        measurement = silver.measure_chip_expansion_overflow(
            instances, crop.shape[:2], settings,
            chip_class=str(boundary_config.get("chip_class", "chip")),
            outward_length_px=distance, min_chip_area_px=min_area)
        if not np.array_equal(silver_mask & ~allowed, measurement.outside_mask):
            raise AssertionError("Visualization geometry differs from production overflow mask")
        subtitle = f"当前模式 chip_expansion；芯片水平外接框四边各外扩 {distance} px；本次未使用 SIFT"
        coordinates = [{"chip_box": item["chip_box"], "allowed_box": item["allowed_box"],
                        "allowed_box_image": [item["allowed_box"][0] + crop_box[0], item["allowed_box"][1] + crop_box[1],
                                              item["allowed_box"][2] + crop_box[0], item["allowed_box"][3] + crop_box[1]]}
                       for item in rectangles]
    else:
        boundary = silver.build_boundary(overflow_config, crop_box=crop_box, circle=circle,
                                         image_shape=crop.shape[:2], image=image, base_dir=config_path.parent)
        if boundary is None:
            raise ValueError("No configured boundary to visualize")
        measurement = silver.measure_overflow(silver_mask, boundary, settings)
        # For line mode the third point defines the inside; it is not another line segment.
        polygons = [{"points": list(boundary.points if boundary.mode == "polygon" else boundary.points[:2]), "mode": boundary.mode}]
        coordinates = [{"points_crop": list(boundary.points),
                        "points_image": [[x + crop_box[0], y + crop_box[1]] for x, y in boundary.points]}]
        subtitle = "黄色为实际标定边界；SIFT 配准仅在配置模板图时执行"

    folder = args.output_dir or ROOT / "visualization/output" / ("silver-boundary-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    segmented = overlay(rgb, silver_mask, PURPLE, 0.65)
    contours(segmented, chip_mask, CYAN)
    checked = segmented.copy()
    checked[measurement.defect_mask] = RED
    draw_boundaries(checked, rectangles, polygons)
    # Keep an unscaled image for inspecting the actual twenty-pixel margin.
    Image.fromarray(checked).save(folder / "boundary_overlay_fullres.png")
    ys, xs = np.nonzero(chip_mask | silver_mask)
    margin = 80
    x1, x2 = max(0, int(xs.min()) - margin), min(rgb.shape[1], int(xs.max()) + margin + 1)
    y1, y2 = max(0, int(ys.min()) - margin), min(rgb.shape[0], int(ys.max()) + margin + 1)
    selected = [pixels[y1:y2, x1:x2] for pixels in (rgb, segmented, checked)]
    compose_panels(selected, ["1 / 原图", "2 / YOLO 分割", "3 / 允许边界与越界"],
                   ["芯片和周围银浆的真实图像", "紫色为 silver；青色描出 chip", "黄色以内允许；红色为越界银浆"],
                   f"银浆溢出边界在哪里？ · {args.image.name}", subtitle).save(folder / "silver_boundary.png")
    for index, item in enumerate(rectangles, 1):
        left, top, right, bottom = item["allowed_box"]
        left, top = max(0, left - 65), max(0, top - 65)
        right, bottom = min(rgb.shape[1], right + 65), min(rgb.shape[0], bottom + 65)
        zoom = [pixels[top:bottom, left:right] for pixels in (rgb, segmented, checked)]
        compose_panels(zoom, ["1 / 芯片局部原图", "2 / 银浆分割", "3 / 判定边界"],
                       ["未叠加标记", "紫色：silver；青色：chip 掩码轮廓", "青框：芯片框；黄框：允许范围"],
                       f"芯片区域 {index} · 边界细节（等比例放大）", subtitle).save(folder / f"chip_{index:02d}.png")
    full = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for item in coordinates:
        if "allowed_box_image" in item:
            a, b, c, d = item["allowed_box_image"]
            cv2.rectangle(full, (a, b), (c - 1, d - 1), YELLOW, 6)
    full_preview = Image.fromarray(full)
    full_preview.thumbnail((1800, 1200), Image.Resampling.LANCZOS)
    full_preview.save(folder / "image_location.png")

    debug = dict(image=str(args.image.resolve()), config=str(config_path), boundary_mode=mode,
                 crop_box=list(crop_box), circle=list(circle), boundaries=coordinates,
                 measurement=measurement.summary(),
                 note="box x2/y2 are exclusive; overlay pixel borders use x2-1/y2-1")
    (folder / "boundary.json").write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Boundary mode: {mode}; regions: {len(rectangles) or len(polygons)}; overflow pixels: {measurement.defect_area_px}")
    print(f"Comparison: {folder / 'silver_boundary.png'}\nFull-resolution overlay: {folder / 'boundary_overlay_fullres.png'}\nCoordinates: {folder / 'boundary.json'}")
    if args.open:
        webbrowser.open((folder / "silver_boundary.png").as_uri())
    detector.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
