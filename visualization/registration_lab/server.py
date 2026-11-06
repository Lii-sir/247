"""Local CPU-only workbench using the production registration functions."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import uuid
import webbrowser

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))
from algorithm.registration import (RegistrationConfig, platform_mask_2d,
    platform_mask_3d, register_platform_masks, refine_affine_registration, _affine_regions)
from algorithm.tilt_correction import correct_smartray_pitch
segment = import_module("algorithm.3D.segment")
SECTIONS = ("registration_3d", "dynamic_affine_3d", "segmentation_3d")
PREVIEW_VERSION = 5


def defaults():
    config = json.loads((ROOT / "config/3D/config.json").read_text(encoding="utf-8-sig"))
    return dict(preview_version=PREVIEW_VERSION, parameters={key: config[key]["parameters"] for key in SECTIONS},
                images_2d=[str(p) for p in sorted((ROOT / "demodatasets/2D").glob("*.png"))],
                images_3d=[str(p) for p in sorted((ROOT / "demodatasets/3D").glob("*.png"))])


def read_image(path, depth=False):
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"找不到图像：{path}")
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8),
                         cv2.IMREAD_UNCHANGED if depth else cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"无法解码图像：{path.name}")
    if depth and (image.ndim != 2 or image.dtype.kind not in "ui"):
        raise ValueError("深度图必须是单通道整数图像，建议使用原始16位PNG。")
    return image


def png(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("预览编码失败")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def normalize_depth(depth):
    valid = (depth > 0) & (depth < 65535)
    output = np.zeros(depth.shape, np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(depth[valid], [1, 99])
        output[valid] = np.clip((depth[valid].astype(float) - lo) / max(hi-lo, 1) * 220 + 25, 0, 255)
    return output


def color_depth(depth):
    """Relative depth colors for display only; invalid pixels stay transparent."""
    valid = (depth > 0) & (depth < 65535)
    normalized = normalize_depth(depth)
    color = cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2BGRA)
    color[:, :, 3] = valid.astype(np.uint8) * 255
    limits = np.percentile(depth[valid], [1, 99]).tolist() if np.any(valid) else None
    return color, limits


def matrix_context(image, depth, correction, settings):
    """Only geometry-affecting preprocessing is relevant to coefficient reuse."""
    role = (settings or {}).get("image_role", "independent_camera")
    prep = dict(model=correction["model"], image_role=role,
                state="applied" if correction["applied"] else correction["reason"])
    if correction["applied"]:
        for key in ("angle_deg", "scan_axis", "scan_pitch_mm", "transform_direction",
                    "z_step_mm", "gray_offset", "z_origin_mm"):
            prep[key] = correction[key]
    return dict(source_space="independent_2d_image" if role == "independent_camera" else "smartray_intensity_depth_piece",
                target_space="depth_piece_local", image_size=[image.shape[1], image.shape[0]],
                depth_size=[depth.shape[1], depth.shape[0]], preprocessing=prep)


def validate_saved_matrix(profile, context):
    if not isinstance(profile, dict) or profile.get("format") != "registration_lab_matrix" or profile.get("version") != 1:
        raise ValueError("请载入由“保存当前阶段矩阵”导出的矩阵文件")
    raw = profile.get("matrix")
    if (not isinstance(raw, list) or len(raw) != 3 or
            any(not isinstance(row, list) or len(row) != 3 for row in raw) or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) for row in raw for v in row)):
        raise ValueError("变换矩阵必须是3×3数值数组")
    matrix = np.asarray(raw, dtype=float)
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("变换矩阵含非有限值或不可逆，请重新保存有效矩阵")
    if not np.allclose(matrix[2, :2], 0, rtol=0, atol=1e-12):
        raise ValueError("当前流程已移除透视变换，请重新保存仿射矩阵")
    saved = profile.get("context")
    if not isinstance(saved, dict):
        raise ValueError("矩阵文件缺少坐标信息")
    for key in ("image_size", "depth_size"):
        size = saved.get(key)
        if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError("矩阵文件的图像尺寸无效")
    if any(saved.get(k) != context[k] for k in ("source_space", "target_space", "preprocessing")):
        raise ValueError("矩阵的坐标或Pitch设置与当前输入不一致，请恢复保存时的图像来源和补偿设置")
    if "segmentation" in saved and saved["segmentation"] != context.get("segmentation"):
        raise ValueError("矩阵的分片设置与当前不同，局部原点可能改变，请重新配准")
    same_source = saved["image_size"] == context["image_size"]
    if context["source_space"] == "smartray_intensity_depth_piece":
        same_source = saved["image_size"][0] == context["image_size"][0]
    if not same_source or saved["depth_size"][0] != context["depth_size"][0]:
        raise ValueError("图像分辨率或深度图宽度与保存时不同；不自动缩放矩阵，请重新配准")
    w, h = context["image_size"]
    denominator = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]]) @ matrix[2]
    if np.any(np.abs(denominator) < 1e-12) or denominator.min() < 0 < denominator.max():
        raise ValueError("矩阵投影在当前图像内经过无穷远线，无法复用")
    warnings = []
    if "segmentation" not in saved and "segmentation" in context:
        warnings.append("旧矩阵未记录分片模式，请核对深度片局部原点是否一致")
    if saved["depth_size"][1] != context["depth_size"][1]:
        warnings.append("深度片高度与来源不同；按当前画幅显示，矩阵未缩放、未叠加整板行偏移")
    return matrix, warnings


def calibrate_segmentation(request, progress=lambda value: None):
    """Scan one depth board once, without 2D input or registration."""
    if request.get("depth_mode", "board") != "board":
        raise ValueError("固定分片标定需要整板深度图，不能使用已分好的单片")
    parameters = request.get("parameters")
    values = parameters.get("segmentation_3d") if isinstance(parameters, dict) else None
    if not isinstance(values, dict):
        raise ValueError("整板分片参数必须是对象")
    # Recalibration always scans pixels, even if the current settings are fixed.
    automatic = {**values, "mode": "auto", "fixed_rois": []}
    config = segment.SegmentationConfig.from_dict(automatic)
    timings = {}

    def timed(name, function, *args):
        progress(name)
        started = time.perf_counter()
        result = function(*args)
        timings[name] = round((time.perf_counter() - started) * 1000, 3)
        return result

    depth = timed("读取深度", read_image, request["image_3d"], True)
    settings = {**request.get("tilt", {}), "enabled_2d": False,
                "image_role": "independent_camera"}
    _, depth, correction = timed("SmartRay Pitch补偿", correct_smartray_pitch,
                                 None, depth, settings)
    pieces = timed("自动扫描整板", segment.segment_depth_image, depth, config)
    fixed = {**values, "mode": "fixed", "expected_piece_count": config.expected_piece_count,
             "fixed_rois": [[p[k] for k in ("x1", "y1", "x2", "y2")] for p in pieces]}
    # Reject padded overlapping bands rather than exporting an unusable config.
    segment.SegmentationConfig.from_dict(fixed)
    return dict(segmentation_3d=fixed, pieces=pieces,
                board_size=[depth.shape[1], depth.shape[0]], correction=correction,
                timings=timings)


def run_registration(request, progress=lambda value: None):
    """Return stage snapshots; board slices preserve full X exactly as DetectBoard."""
    started = time.perf_counter()
    parameters = request["parameters"]
    if not isinstance(parameters, dict) or any(not isinstance(parameters.get(k), dict) for k in SECTIONS):
        raise ValueError("参数必须包含四个配准/分片参数对象。")
    mode = request.get("depth_mode", "board")
    if mode not in {"board", "piece"}:
        raise ValueError("深度输入模式必须是board或piece")
    registration_mode = request.get("registration_mode", "estimate")
    if registration_mode not in {"estimate", "reuse"}:
        raise ValueError("对齐方式必须是estimate或reuse")
    timings = {}

    def timed(name, function, *args):
        progress(name)
        t = time.perf_counter()
        value = function(*args)
        timings[name] = round((time.perf_counter()-t)*1000, 3)
        return value

    image = timed("读取2D", read_image, request["image_2d"])
    depth = timed("读取深度", read_image, request["image_3d"], True)
    intensity_input = request.get("tilt", {}).get("image_role") == "smartray_intensity"
    if intensity_input and image.shape[:2] != depth.shape:
        raise ValueError("强度图必须与输入深度整图同尺寸、逐像素对应")
    image, depth, correction = timed("SmartRay Pitch补偿", correct_smartray_pitch, image, depth, request.get("tilt", {}))
    y1, y2 = 0, depth.shape[0]
    pieces = []
    if mode == "board":
        pieces = timed("整板分片", segment.segment_depth_image, depth,
                       segment.SegmentationConfig.from_dict(parameters["segmentation_3d"]))
        pid = int(request.get("piece_id", 1))
        selected = next((p for p in pieces if p["piece_id"] == pid), None)
        if selected is None:
            raise ValueError(f"片号超出范围：1～{len(pieces)}")
        y1, y2 = selected["y1"], selected["y2"]
        if intensity_input:
            image = image[y1:y2]
            correction["intensity_crop_rows"] = [y1, y2]
            transform = correction["image_transform_original_to_corrected"]
            if transform is not None:
                correction["image_transform_original_to_corrected"] = (
                    np.asarray([[1, 0, 0], [0, 1, -y1], [0, 0, 1]]) @ np.asarray(transform)).tolist()
    local = depth[y1:y2]  # Production keeps the full width, not the segmentation X bbox.
    context = matrix_context(image, local, correction, request.get("tilt", {}))
    context["segmentation"] = {"mode": "piece"}
    if mode == "board":
        segmentation = segment.SegmentationConfig.from_dict(parameters["segmentation_3d"])
        context["segmentation"] = dict(mode=segmentation.mode,
            expected_piece_count=segmentation.expected_piece_count)
        if segmentation.mode == "fixed":
            context["segmentation"]["fixed_rois"] = [list(box) for box in segmentation.fixed_rois]
    affine_parameters = parameters["dynamic_affine_3d"]
    all_regions = _affine_regions(affine_parameters.get("rois"), local.shape[1], local.shape[0])
    used_regions = ([] if registration_mode == "reuse" else
                    _affine_regions(affine_parameters.get("rois"), local.shape[1], local.shape[0],
                                    affine_parameters.get("selected_regions")))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stages = []
    reuse = None
    if registration_mode == "reuse":
        matrix, warnings = timed("载入矩阵", validate_saved_matrix, request.get("matrix_profile"), context)
        profile = request["matrix_profile"]
        reuse = dict(source=profile.get("source", {}), warnings=warnings, quality_evaluated=False)
        stages.append(("reused", "复用矩阵", dict(homography_2d_to_3d=matrix.tolist(),
                       is_valid=None, matrix_valid=True, quality_evaluated=False)))
    else:
        config = RegistrationConfig.from_dict(parameters["registration_3d"])
        source = timed("2D平台掩码", platform_mask_2d, image, config)
        target = timed("3D平台掩码", platform_mask_3d, local, config)
        registration = timed("粗配准", register_platform_masks, source, target, config)
        stages.append(("coarse", "粗配准", registration))
        if request.get("affine", True):
            registration = timed("仿射配准", refine_affine_registration, gray, local, registration,
                                 parameters["dynamic_affine_3d"])
            stages.append(("affine", "仿射配准", registration))
    progress("生成预览")
    preview_started = time.perf_counter()
    previews = []
    target_edge = cv2.morphologyEx((local > 0).astype(np.uint8)*255, cv2.MORPH_GRADIENT,
                                   np.ones((3, 3), np.uint8))
    for key, name, result in stages:
        matrix = np.asarray(result["homography_2d_to_3d"], dtype=float)
        warped = cv2.warpPerspective(gray, matrix, (local.shape[1], local.shape[0]), flags=cv2.INTER_LINEAR)
        warped_color = cv2.warpPerspective(image, matrix, (local.shape[1], local.shape[0]), flags=cv2.INTER_LINEAR)
        edge = cv2.morphologyEx((warped > parameters["dynamic_affine_3d"].get("threshold_2d", 50)).astype(np.uint8)*255,
                                cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        # BGR: cyan 2D edges, orange depth edges; overlapping edges appear pale.
        color = np.zeros((*local.shape, 3), np.uint8)
        color[edge > 0] = (220, 210, 35)
        color[target_edge > 0] = (30, 155, 255)
        color[(edge > 0) & (target_edge > 0)] = (235, 245, 250)
        previews.append(dict(key=key, name=name, registration=result, warped=png(warped), warped_color=png(warped_color), edges=png(color)))
        image_transform = correction["image_transform_original_to_corrected"]
        previews[-1]["homography_original_2d_to_corrected_depth"] = (
            (matrix @ np.asarray(image_transform)).tolist() if image_transform is not None else None)
    timings["生成预览"] = round((time.perf_counter()-preview_started)*1000, 3)
    depth_color, limits = color_depth(local)
    return dict(stages=previews, depth_preview=png(normalize_depth(local)),
                depth_color_preview=png(depth_color), depth_color_limits_raw=limits,
                source_preview=png(cv2.resize(image, (round(image.shape[1]*min(1, 1200/image.shape[1])),
                                                     round(image.shape[0]*min(1, 1200/image.shape[1]))))),
                image_size=[image.shape[1], image.shape[0]], depth_size=[local.shape[1], local.shape[0]],
                depth_dtype=str(depth.dtype), depth_rows=[y1, y2], pieces=pieces,
                affine_scoring=dict(coordinate_system="depth_piece_local",
                                    enabled=registration_mode == "estimate" and bool(request.get("affine", True)),
                                    rois=[box for _, box in all_regions],
                                    region_ids=[i for i, _ in all_regions],
                                    selected_region_ids=[i for i, _ in used_regions]),
                parameters=parameters, tilt=request.get("tilt", {}), correction=correction,
                registration_mode=registration_mode, matrix_context=context, matrix_reuse=reuse,
                inputs={k: request.get(k) for k in ("image_2d", "image_3d", "depth_mode", "piece_id", "affine", "perspective")},
                timings=timings, algorithm_ms=round(sum(v for k,v in timings.items() if k in
                    {"2D平台掩码", "3D平台掩码", "粗配准", "仿射配准", "透视配准"}), 3),
                total_ms=round((time.perf_counter()-started)*1000, 3))


class LabServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address):
        super().__init__(address, Handler)
        # Keep page assets and the running backend from the same launch.
        self.assets = {name: (ASSETS / name).read_bytes() for name in
                       ("index.html", "app.js", "style.css", "preview.js")}
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobs = {}
        self.lock = threading.Lock()
        self.upload_dir = tempfile.TemporaryDirectory(prefix="registration-lab-")

    def server_close(self):
        super().server_close()
        self.pool.shutdown(wait=True)
        self.upload_dir.cleanup()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, value, mime="application/json; charset=utf-8"):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if mime.startswith("application/json") else value
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def allowed(self):
        expected = f"127.0.0.1:{self.server.server_port}"
        return self.headers.get("Host") == expected and self.headers.get("Origin", f"http://{expected}") == f"http://{expected}"

    def do_GET(self):
        if not self.allowed():
            return self.send(403, {"error": "仅允许本机页面访问"})
        path = urlsplit(self.path).path
        if path == "/api/defaults":
            return self.send(200, defaults())
        if path.startswith("/api/jobs/"):
            with self.server.lock:
                job = dict(self.server.jobs.get(path.rsplit("/", 1)[-1], {}))
            return self.send(200 if job else 404, job or {"error": "测试记录不存在"})
        files = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/preview.js": ("preview.js", "text/javascript"), "/style.css": ("style.css", "text/css")}
        if path not in files:
            return self.send(404, {"error": "页面不存在"})
        name, mime = files[path]
        self.send(200, self.server.assets[name], mime+"; charset=utf-8")

    def do_POST(self):
        if not self.allowed():
            return self.send(403, {"error": "仅允许本机页面访问"})
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 160*1024*1024:
                raise ValueError("请求为空或超过160MB")
            path = urlsplit(self.path).path
            if path == "/api/upload":
                upload = Path(self.server.upload_dir.name) / (uuid.uuid4().hex + ".image")
                upload.write_bytes(self.rfile.read(size))
                return self.send(200, {"path": str(upload)})
            if path not in {"/api/run", "/api/calibrate-segmentation"}:
                return self.send(404, {"error": "接口不存在"})
            request = json.loads(self.rfile.read(size))
            with self.server.lock:
                if any(j["status"] == "running" for j in self.server.jobs.values()):
                    return self.send(409, {"error": "已有测试运行中，请等待完成"})
                self.server.jobs.clear()  # Browser retains A/B snapshots; server retains only the latest.
                key = uuid.uuid4().hex
                self.server.jobs[key] = {"status": "running", "phase": "准备输入"}

            def update(**values):
                with self.server.lock:
                    self.server.jobs[key].update(values)

            def execute():
                try:
                    function = calibrate_segmentation if path == "/api/calibrate-segmentation" else run_registration
                    result = function(request, lambda phase: update(phase=phase))
                    update(status="done", result=result)
                except Exception as exc:
                    update(status="failed", error=str(exc))
            self.server.pool.submit(execute)
            self.send(202, {"id": key})
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.send(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    with LabServer(("127.0.0.1", args.port)) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Registration Lab: {url}", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
