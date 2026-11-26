"""2D single-piece flow: circle crop, YOLO, bending and silver checks."""
from __future__ import annotations

import gc
import io
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch

from algorithm.runtime.config import (
    DEFAULT_ALGORITHM_PARAMETERS,
    feature_parameters,
    inspection_parameters,
    load_runtime_config,
    parse_config_text,
)
from algorithm.application.defect_checks import silver_check, wire_bending_result
from algorithm.runtime.defect_plan import (
    CHECKS_2D, defect_orders, evaluate_in_order, require_configured_checks, unrun_checks,
)
from algorithm.vision2d import chip_alignment, chip_anomaly, circle_roi, silver_inspection, wire_roi_presence
from algorithm.vision2d import defect_geometry as geometry
from algorithm.vision2d.instance_geometry import component_geometry, mask_geometry
from algorithm.vision2d.segmentation_instances import extract_instances
from transport.shared_memory import read_shared_memory

class Detect2D:
    def __init__(self, log):
        self.log = log
        self.model = None
        self.model_path = None
        self.model_paths: Dict[str, str] = {}
        self.config_path = None
        self.config: Dict[str, Any] = {}
        self.algorithm_parameters = dict(DEFAULT_ALGORITHM_PARAMETERS)
        self.crop_config: Dict[str, Any] = {}
        self.mask_config: Dict[str, Any] = {}
        self.silver_config: Dict[str, Any] = {}
        self.chip_alignment_config: Dict[str, Any] = {}
        self.chip_anomaly_config: Dict[str, Any] = {"configured": False}
        self.chip_anomaly_inspector = None
        self.defect_order = self.height_order = None
        # Per-request diagnostic artifacts are kept outside the JSON protocol.
        # The visualization tool can consume them without forcing numpy/torch
        # objects through the bridge response.
        self._last_visualization = None

    def handle_init(self, req: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        raw_model_paths = req.get("model_path")
        if isinstance(raw_model_paths, dict):
            self.model_paths = {
                str(name): str(path)
                for name, path in raw_model_paths.items()
                if str(name).strip() and str(path).strip()
            }
        elif raw_model_paths:
            # 兼容旧版 Bridge 发送的单模型路径。
            self.model_paths = {"default": str(raw_model_paths)}
        else:
            self.model_paths = {}

        # 对外只暴露 2D 初始化；具体使用哪个模型由 2D 算法内部固定决定。
        self.model_path = self.model_paths.get("YOLO") or self.model_paths.get("default")
        self.config_path = str(req.get("config_path") or "") or None
        try:
            self._load_config(str(req.get("config_text") or ""))
            if not self.model_path:
                raise ValueError("2D model_path is empty")
            self._validate_gpu_requirement()
            load_start = time.perf_counter()
            with redirect_stdout(io.StringIO()):
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
            model_load_ms = self.elapsed_ms(load_start)
            self.chip_anomaly_inspector = chip_anomaly.ChipAnomalyInspector(
                self.chip_anomaly_config, self.model_paths
            )
            anomaly_load_timings = self.chip_anomaly_inspector.load()
        except Exception as exc:
            self.model = None
            self.chip_anomaly_inspector = None
            return {"ok": False, "cmd": "init_algorithm", "stage": "2d", "error": str(exc),
                    "timings": {"total_init_ms": self.elapsed_ms(start)}, "model_load_timings": {}}
        return {"ok": True, "cmd": "init_algorithm", "stage": "2d",
                "message": "2D circle crop and segmentation model ready",
                "timings": {"total_init_ms": self.elapsed_ms(start)},
                "model_load_timings": {"YOLO": model_load_ms, **anomaly_load_timings}}

    def _load_config(self, text: str) -> None:
        config = load_runtime_config(text, self.config_path)
        if not isinstance(config, dict):
            raise ValueError("2D config must be an object")
        self.config = config
        common = feature_parameters(config, "common_2d")
        self.defect_order, self.height_order = defect_orders(inspection_parameters(config, common))
        crop = common.get("crop") or {}
        if not isinstance(crop, dict):
            raise ValueError("crop must be an object")
        offsets = crop.get("detection_roi_offsets", crop.get("roi_offsets"))
        if not isinstance(offsets, (list, tuple)) or len(offsets) != 4:
            raise ValueError("crop.detection_roi_offsets must contain four numbers")
        offsets = tuple(float(v) for v in offsets)
        radius = int(crop.get("circle_radius", 560))
        tolerance = float(crop.get("radius_tolerance_pixels", 10))
        if radius <= 0 or tolerance < 0 or tolerance >= radius:
            raise ValueError("invalid circle radius/tolerance")
        search_roi = crop.get("circle_search_roi")
        if not isinstance(search_roi, (list, tuple)) or len(search_roi) != 4:
            raise ValueError("crop.circle_search_roi must contain four integers")
        search_roi = tuple(int(value) for value in search_roi)
        if search_roi[2] <= search_roi[0] or search_roi[3] <= search_roi[1]:
            raise ValueError("crop.circle_search_roi coordinates are invalid")
        wire_block = feature_parameters(config, "wire_bending_2d", required=False)
        wire_block = wire_block or {}
        rectangle_offsets = wire_block.get("roi_offsets", [])
        if not rectangle_offsets:
            rectangle_offsets = [offsets]
        if not isinstance(rectangle_offsets, list):
            raise ValueError("crop.rectangle_roi_offsets must be a list")
        rois = []
        for item in rectangle_offsets:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                raise ValueError("each rectangle ROI must contain four numbers")
            rois.append(tuple(float(v) for v in item))
        self.crop_config = {"detection_roi_offsets": offsets, "rectangle_roi_offsets": rois,
                            "circle_radius": radius, "radius_tolerance_pixels": tolerance,
                            "circle_search_roi": search_roi}
        mask = common.get("mask") or {}
        if not isinstance(mask, dict):
            raise ValueError("mask must be an object")
        self.mask_config = {"min_component_area": int(mask.get("min_component_area", 500)),
                            "rectangle_ratio_threshold": float(
                                wire_block.get("rectangle_ratio_threshold",
                                                 mask.get("rectangle_ratio_threshold", 0.75)))}
        if self.mask_config["min_component_area"] < 1:
            raise ValueError("invalid mask configuration")
        if not 0 <= self.mask_config["rectangle_ratio_threshold"] <= 1:
            raise ValueError("rectangle_ratio_threshold must be in [0,1]")
        raw_params = common.get("yolo", {})
        if not isinstance(raw_params, dict):
            raise ValueError("common_2d.parameters.yolo must be an object")
        params = dict(raw_params)
        allowed = {"imgsz", "conf", "iou", "max_det", "device", "half", "end2end"}
        self.algorithm_parameters = {k: v for k, v in {**DEFAULT_ALGORITHM_PARAMETERS, **params}.items() if k in allowed}
        raw_overflow = feature_parameters(
            config, "silver_overflow_2d", required=False
        )
        raw_continuity = feature_parameters(
            config, "silver_continuity_2d", required=False
        )
        self.silver_config = {
            "overflow": raw_overflow,
            "continuity": raw_continuity,
        }
        if raw_overflow is not None:
            silver_inspection.overflow_settings(raw_overflow)
        if raw_continuity is not None:
            silver_inspection.continuity_settings(raw_continuity)
        raw_chip_alignment = feature_parameters(
            config, "chip_rotation_2d", required=False
        )
        self.chip_alignment_config = raw_chip_alignment
        if raw_chip_alignment is not None:
            threshold = float(raw_chip_alignment.get("max_angle_deg", 3.0))
            if not 0.0 <= threshold <= 45.0:
                raise ValueError("chip_alignment_2d.max_angle_deg must be in [0, 45]")
        self.wire_enabled = feature_parameters(
            config, "wire_bending_2d", required=False
        ) is not None
        self.silver_overflow_enabled = raw_overflow is not None
        self.silver_continuity_enabled = raw_continuity is not None
        raw_chip_anomaly = feature_parameters(
            config, "chip_anomaly_2d", required=False
        )
        self.chip_anomaly_config = chip_anomaly.validate_config(raw_chip_anomaly)
        if self.height_order is not None:
            require_configured_checks(self.defect_order, {
                "wire_bending_2d": self.wire_enabled,
                "chip_rotation_2d": raw_chip_alignment is not None,
                "chip_anomaly_2d": self.chip_anomaly_config.get("configured"),
                "silver_overflow_2d": self.silver_overflow_enabled,
                "silver_continuity_2d": self.silver_continuity_enabled,
            })

    def _validate_gpu_requirement(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("2D detection requires an available CUDA GPU")
        if str(self.algorithm_parameters.get("device", "")).lower() == "cpu":
            raise ValueError("2D algorithm.parameters.device must select a CUDA device")

    def _predict(self, roi: np.ndarray):
        params = {**self.algorithm_parameters, "verbose": False, "retina_masks": True}
        return self.model.predict(source=roi, **params)

    def take_last_visualization(self):
        """Return and clear the most recent in-process visualization payload."""
        payload = self._last_visualization
        self._last_visualization = None
        return payload

    def _synchronize_model(self) -> None:
        device = getattr(self.model, "device", None)
        if isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _decode_image(data: bytes, fmt: str, req: Dict[str, Any]) -> np.ndarray:
        fmt = fmt.lower()
        if fmt in {"rgb8", "bgr8", "gray8", "raw"}:
            w, h, channels = int(req.get("image_width", 0)), int(req.get("image_height", 0)), int(req.get("image_channels", 3) or 3)
            if w <= 0 or h <= 0 or len(data) < w * h * channels:
                raise ValueError("raw 2D image requires valid dimensions")
            arr = np.frombuffer(data[:w * h * channels], dtype=np.uint8).reshape(h, w, channels)
            if channels == 1:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if fmt == "rgb8" else arr.copy()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("unable to decode 2D image")
        return image

    def _crop(self, image: np.ndarray):
        circle = circle_roi.find_circle(image, fixed_radius=self.crop_config["circle_radius"],
                                        radius_tolerance=self.crop_config["radius_tolerance_pixels"] / self.crop_config["circle_radius"],
                                        search_roi=self.crop_config["circle_search_roi"])
        roi, box = circle_roi.crop_from_config(image, circle, {"roi_offsets": self.crop_config["detection_roi_offsets"]})
        roi_boxes = []
        for offsets in self.crop_config["rectangle_roi_offsets"]:
            _, full = circle_roi.crop_from_config(image, circle, {"roi_offsets": offsets})
            roi_boxes.append((full[0] - box[0], full[1] - box[1], full[2] - box[0], full[3] - box[1]))
        return roi, box, circle, roi_boxes

    def handle_detect(self, req: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            size = int(req.get("image_size", 0)); shm = str(req.get("shm_name", ""))
            if not shm or size <= 0:
                raise ValueError("invalid 2D shared memory image")
            stage_start = time.perf_counter()
            image = self._decode_image(read_shared_memory(shm, size), str(req.get("image_format", "encoded")), req)
            decode_ms = self.elapsed_ms(stage_start)
            stage_start = time.perf_counter()
            roi, roi_box, circle, roi_boxes = self._crop(image)
            circle_crop_ms = self.elapsed_ms(stage_start)
            if self.model is None:
                raise RuntimeError("2D model is not initialized")
            self._synchronize_model()
            stage_start = time.perf_counter()
            predictions = self._predict(roi)
            self._synchronize_model()
            inference_ms = self.elapsed_ms(stage_start)
            stage_start = time.perf_counter()
            results = self._convert_predictions(predictions, roi, roi_box, roi_boxes)
            diagnostics, first_ng = self._diagnose(
                results, roi.shape[:2], roi_box, circle, roi_boxes, image, req.get("threshold"))
            postprocess_ms = self.elapsed_ms(stage_start)
            rectangles = [{"points": item["rectangle"],
                           "mask_ratio": item["rectangle_ratio"],
                           "is_ok": item["is_ok"]}
                          for item in results if item["rectangle"] is not None]
            public_instances = [self._public_instance(item, roi_box[:2]) for item in results]
            total_ms = self.elapsed_ms(start)
            self._last_visualization = {
                "image": image,
                "roi": roi.copy(),
                "roi_box": tuple(int(v) for v in roi_box),
                "predictions": predictions,
                "chip_anomaly": diagnostics["chip_anomaly_2d"].pop("_visualizations", []),
            }
            if self.defect_order is not None and first_ng is not None:
                diagnostics["first_ng_check"] = first_ng
            return {"ok": True, "cmd": "detect", "stage": "2d", "elapsed_ms": self.elapsed_ms(start),
                    "results_2d": {"is_ok": bool(results) and first_ng is None
                                            and all(item["is_ok"] for item in results),
                                   "circle": {"x": circle[0], "y": circle[1], "radius": circle[2]},
                                   "crop_box": list(roi_box),
                                   "wire_rois_2d": [list(item) for item in roi_boxes],
                                   "rectangles": rectangles,
                                   "instances": public_instances,
                                   **diagnostics},
                    "timings": {"decode_ms": decode_ms, "circle_crop_ms": circle_crop_ms,
                                "inference_ms": inference_ms, "mask_postprocess_ms": postprocess_ms,
                                "total_detect_ms": total_ms},
                    "message": "2D segmentation complete"}
        except Exception as exc:
            return {"ok": False, "cmd": "detect", "stage": "2d", "error": f"2D inference failed: {exc}"}

    def _diagnose(self, instances, shape, crop_box, circle, rois, image, threshold):
        base_dir = Path(self.config_path).parent if self.config_path else None

        def wires():
            measurements = self._measure_2d_wire_regions(instances, rois, crop_box)
            roi_checks = wire_roi_presence.check_wire_roi_presence(instances, rois, crop_box[:2])
            return measurements, roi_checks

        def rotation():
            return chip_alignment.measure_chip_alignment(instances, self.chip_alignment_config, crop_box[:2])

        def anomaly():
            if self.chip_anomaly_inspector is None:
                raise RuntimeError("chip anomaly inspector is not initialized")
            return self.chip_anomaly_inspector.inspect(image, instances, crop_box,
                include_visualization=True, threshold=threshold)

        def silver(name, config):
            return silver_check(name, config, instances, shape, crop_box=crop_box,
                                circle=circle, image=image, base_dir=base_dir)

        if self.height_order is not None and not instances:
            return unrun_checks(self.defect_order, CHECKS_2D), None
        if self.defect_order is not None:
            checks = {
                "wire_bending_2d": (lambda: wire_bending_result(*wires())) if self.wire_enabled else None,
                "chip_rotation_2d": rotation if self.chip_alignment_config is not None else None,
                "chip_anomaly_2d": anomaly if self.chip_anomaly_config.get("configured") else None,
                "silver_overflow_2d": (lambda: silver("silver_overflow_2d", self.silver_config["overflow"]))
                                      if self.silver_overflow_enabled else None,
                "silver_continuity_2d": (lambda: silver("silver_continuity_2d", self.silver_config["continuity"]))
                                        if self.silver_continuity_enabled else None,
            }
            return evaluate_in_order(self.defect_order, checks, short_circuit=self.height_order is None)

        if self.wire_enabled:
            measurements, roi_checks = wires()
            wire = {"enabled": True, "is_ok": bool(roi_checks) and all(item["is_ok"] for item in roi_checks),
                    "measurements": measurements, "roi_checks": roi_checks}
        else:
            wire = {"enabled": False, "is_ok": None, "measurements": [], "roi_checks": []}
        chip, appearance = rotation(), anomaly()
        silver_results = silver_inspection.inspect_masks(instances, shape, self.silver_config,
            crop_box=crop_box, circle=circle, image=image, base_dir=base_dir)
        silver_results = silver_inspection.offset_geometry(silver_results, crop_box[:2])
        results = {
            "wire_bending_2d": wire,
            "chip_rotation_2d": chip,
            "chip_anomaly_2d": appearance,
            "silver_overflow_2d": {"enabled": self.silver_overflow_enabled, **silver_results["silver_overflow"]},
            "silver_continuity_2d": {"enabled": self.silver_continuity_enabled, **silver_results["disconnection"]},
        }
        first_ng = next(
            (name for name, result in results.items() if result.get("enabled")
             and result.get("status") != "disabled" and result.get("is_ok") is not True), None)
        return results, first_ng

    def _convert_predictions(self, predictions: Any, roi: np.ndarray, box, roi_boxes):
        output = []
        ox, oy = box[:2]
        for result in predictions or []:
            if getattr(result, "masks", None) is not None:
                for instance in extract_instances(result, roi.shape[:2], self.mask_config["min_component_area"]):
                    x, y, width, height = instance["_mask_bbox"]
                    instance.update(instance_id=len(output) + 1,
                        x1=float(x + ox), y1=float(y + oy),
                        x2=float(x + width + ox), y2=float(y + height + oy),
                        mask_area_px=instance["_mask_area_px"])
                    output.append(instance)
                continue

            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            # Retain the box-only contract for checkpoints without segmentation.
            names = getattr(result, "names", {}) or {}
            coords = boxes.xyxy.cpu().numpy()
            confidences, classes = boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
            for index, xy in enumerate(coords):
                class_id = int(classes[index])
                name = names.get(class_id, names.get(str(class_id), class_id)) if isinstance(names, dict) else class_id
                output.append(dict(instance_id=len(output) + 1, class_id=class_id, class_name=str(name),
                    confidence=float(confidences[index]), x1=float(xy[0] + ox), y1=float(xy[1] + oy),
                    x2=float(xy[2] + ox), y2=float(xy[3] + oy), mask_area_px=0,
                    rectangle_ratio=0., rectangle=None, is_ok=False))
        return output

    def _measure_2d_wire_regions(self, instances, roi_boxes, crop_box):
        """Return per-instance thin/wire bending diagnostics in image pixels."""
        output = []
        offset = np.asarray(crop_box[:2], dtype=np.int32)
        for instance in instances:
            if str(instance.get("class_name", "")).casefold() not in {"wire", "thin"}:
                continue
            if not isinstance(instance.get("mask"), np.ndarray):
                continue
            mask = np.asarray(instance["mask"], dtype=np.uint8)
            cleaned, mask_offset = geometry.clean_mask_crop(
                mask, min_area=self.mask_config["min_component_area"], kernel_size=5,
                bounds=instance.get("_mask_bbox"))
            bending = geometry.measure_wire_bending(
                cleaned, roi_boxes, self.mask_config["rectangle_ratio_threshold"],
                mask_offset=mask_offset, output_offset=offset,
            )
            instance["bending_2d"] = bending
            instance["is_ok"] = bool(bending["is_ok"])
            instance["rectangle_ratio"] = float(bending["mask_ratio"])
            instance["rectangle"] = bending["rectangle"] or None
            output.append({
                "region_id": len(output) + 1,
                "class_name": str(instance["class_name"]),
                "instance_id": int(instance["instance_id"]),
                "mask_2d": mask_geometry(cleaned, offset + mask_offset, epsilon=1.0),
                "bending_2d": bending,
            })
        return output

    @staticmethod
    def _public_instance(instance: Dict[str, Any], offset=(0, 0)) -> Dict[str, Any]:
        public = {}
        for key, value in instance.items():
            if key == "mask" or key.startswith("_"):
                continue
            if key == "chip_alignment":
                key = "chip_rotation_2d"
            public[key] = value
        if "mask_2d" not in public and isinstance(instance.get("mask"), np.ndarray):
            public["mask_2d"] = component_geometry(instance, offset, epsilon=1.0)
        return public

    @staticmethod
    def _box_iou(first: Dict[str, Any], second: Dict[str, Any]) -> float:
        x1, y1 = max(first["x1"], second["x1"]), max(first["y1"], second["y1"])
        x2, y2 = min(first["x2"], second["x2"]), min(first["y2"], second["y2"])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a = max(0.0, first["x2"] - first["x1"]) * max(0.0, first["y2"] - first["y1"])
        b = max(0.0, second["x2"] - second["x1"]) * max(0.0, second["y2"] - second["y1"])
        return inter / (a + b - inter) if a + b - inter else 0.0

    @staticmethod
    def _apply_nms(boxes: list[Dict[str, Any]], iou_threshold: float, agnostic: bool = False):
        kept = []
        for item in sorted(boxes, key=lambda value: value.get("confidence", 0), reverse=True):
            if all(agnostic or item.get("class_id") != old.get("class_id") or
                   Detect2D._box_iou(item, old) <= iou_threshold for old in kept):
                kept.append(item)
        return kept

    def handle_reload_defect_config(self, req):
        try:
            self._load_config(str(req.get("config_text") or ""))
            self._validate_gpu_requirement()
            return {"ok": True, "cmd": "reload_defect_config", "message": "2D config reloaded"}
        except Exception as exc:
            return {"ok": False, "cmd": "reload_defect_config", "error": str(exc)}

    def release(self):
        self.model = None; self.model_path = None; self.model_paths = {}; self.config_path = None; self.config = {}; self.crop_config = {}; self.mask_config = {}; self.silver_config = {}; self.chip_alignment_config = {}; self.chip_anomaly_config = {"configured": False}; self._last_visualization = None
        if self.chip_anomaly_inspector is not None:
            self.chip_anomaly_inspector.release()
        self.chip_anomaly_inspector = None
        gc.collect()

    @staticmethod
    def elapsed_ms(start):
        return (time.perf_counter() - start) * 1000.0
