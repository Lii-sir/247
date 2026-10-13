"""Build a click-to-calibrate page using the existing standalone report viewer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

from generate_report import render_report

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/2D/config.json"
CALIBRATION = ROOT / "config/2D/calibration/silver_boundary.json"


def validate_calibration(document: dict, image: Path) -> dict:
    """Validate the downloaded points against the original image, not its preview."""
    from importlib import import_module
    from PIL import Image
    sys.path.insert(0, str(ROOT / "python"))
    silver = import_module("algorithm.2D.silver_inspection")
    if not isinstance(document, dict) or document.get("mode") != "polygon":
        raise ValueError("标定 JSON 必须为 mode=polygon 的对象")
    points = document.get("template_points")
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("至少需要三个真实标定点；不能安装空标定")
    with Image.open(image) as source:
        width, height = source.size
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    if document.get("template_size") != [width, height]:
        raise ValueError("模板尺寸与标定记录不一致，请使用标定时的原图")
    if document.get("template_sha256") != digest:
        raise ValueError("模板内容与标定记录不一致，请使用标定时的原图")
    boundary = silver.Boundary(tuple(tuple(point) for point in points), "polygon")
    boundary.validate_image((height, width))
    saved = {"mode": "polygon", "template_points": [list(point) for point in boundary.points],
             "template_size": [width, height], "template_sha256": digest}
    defaults = {"ratio_threshold": 0.7, "min_matches": 11, "ransac_threshold": 5.0,
                "min_inliers": 8, "min_inlier_ratio": 0.25,
                "max_image_side": 2400, "max_features": 10000}
    for key, default in defaults.items():
        value = document.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} 必须为数值")
        import math
        if not math.isfinite(value):
            raise ValueError(f"{key} 必须为有限数值")
        if key in {"ratio_threshold", "min_inlier_ratio"}:
            if not 0 < value <= 1:
                raise ValueError(f"{key} 必须在 (0,1] 内")
        elif key == "ransac_threshold":
            if value <= 0:
                raise ValueError("ransac_threshold 必须大于零")
        elif int(value) != value or value < (4 if key in {"min_matches", "min_inliers"} else 1):
            raise ValueError(f"{key} 必须为有效正整数")
        saved[key] = value
    return saved


def apply_calibration(document: dict, image: Path, config_path: Path) -> Path:
    """Explicitly install validated calibration, preserving backups of existing files."""
    saved = validate_calibration(document, image)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    parameters = config["silver_overflow_2d"]["parameters"]
    folder = config_path.parent / "calibration"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "silver_boundary.json"
    template = folder / ("silver_boundary_template_" + saved["template_sha256"][:12] + image.suffix.lower())
    # The relative template path is resolved against calibration JSON's directory.
    saved["template_path"] = template.name
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    for path in (config_path, target, template, folder):
        if path.is_symlink():
            raise ValueError(f"拒绝修改链接路径：{path}")
    if template.exists():
        if hashlib.sha256(template.read_bytes()).hexdigest() != saved["template_sha256"]:
            raise ValueError("模板文件名称冲突，未覆盖原文件")
    else:
        shutil.copyfile(image, template)
    if target.exists():
        shutil.copyfile(target, target.with_name(target.name + "." + stamp + ".bak"))
    shutil.copyfile(config_path, config_path.with_name(config_path.name + "." + stamp + ".bak"))
    parameters["boundary"] = {"mode": "polygon", "calibration_path": "calibration/silver_boundary.json"}
    for path, value in ((target, saved), (config_path, config)):
        temporary = path.with_name(path.name + "." + stamp + ".tmp")
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the existing visualization viewer to mark silver boundary vertices.")
    parser.add_argument("--image", type=Path, default=ROOT / "demodatasets/2D/001.png",
                        help="Exact full-resolution template image; choose a representative production frame when possible")
    parser.add_argument("--output-dir", type=Path,
                        help="Optional new directory for the standalone preview report")
    parser.add_argument("--apply", type=Path,
                        help="Install the downloaded silver_boundary.json using the matching original --image")
    parser.add_argument("--config", type=Path, default=CONFIG,
                        help="Stage configuration to update when --apply is used")
    parser.add_argument("--open", action="store_true", help="Open the report after generating it")
    args = parser.parse_args()
    image = args.image.resolve()
    if not image.is_file():
        print(f"Template image does not exist: {image}", file=sys.stderr)
        return 1
    try:
        if args.apply is not None:
            document = json.loads(args.apply.resolve().read_text(encoding="utf-8-sig"))
            target = apply_calibration(document, image, args.config.resolve())
            print(f"Calibration installed: {target}\n模板已复制；旧标定与配置已备份。请重启检测服务或重新初始化算法。")
            return 0
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        from PIL import Image
        with Image.open(image) as source:
            width, height = source.size
        folder = args.output_dir or ROOT / "visualization/output" / (
            "silver-calibration-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
        folder = folder.resolve()
        folder.mkdir(parents=True, exist_ok=False)
        document = {
            "viewer_version": 1,
            "title": "银浆允许边界标定 · 预标定模板多边形",
            "board_id": "silver-boundary-calibration",
            "mode": "calibration",
            "images_2d": [{"piece_id": 1, "path": str(image)}],
            "image_3d": None,
            "responses_2d": [],
            "response": {
                "ok": True, "cmd": "silver_boundary_calibration", "stage": "2d",
                "board_id": "silver-boundary-calibration", "piece_id": 1,
                "results_2d": {"is_ok": None, "instances": [], "silver_overflow_2d": {
                    "status": "calibration_preview", "is_ok": None}},
            },
            "calibration": {
                "enabled": True,
                "template_path": image.name,
                "template_source": str(image),
                "template_size": [width, height],
                "template_sha256": digest,
                "target_config": str(CONFIG),
                "target_calibration": str(CALIBRATION),
                "template_points": [],
            },
            "warnings": [
                "仅生成标定预览，不运行模型、不修改配置。请沿真实银浆允许区域边缘依次点击至少三个点；不要点击芯片掩码外扩框。",
                "下载标定 JSON 后必须用 --apply 安装。安装流程校验模板尺寸与 SHA-256，并自动备份现有 calibration JSON 和 2D 配置。",
            ],
        }
        report = render_report(document, folder / "report.html", base_dir=folder)
        print(f"Viewer: {report}\nTemplate: {image}\n下一步：点击边界顶点并下载 silver_boundary.json，再按 visualization/README.md 指令安装。")
        if args.open:
            import webbrowser
            webbrowser.open(report.as_uri())
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Cannot build calibration viewer: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
