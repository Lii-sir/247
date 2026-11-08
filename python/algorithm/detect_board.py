"""Streaming board flow: circle-relative 2D pieces, 3D split and alignment."""
from __future__ import annotations

import gc
import io
import queue
import sys
import threading
import time
from contextlib import nullcontext, redirect_stdout
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional

from algorithm.config import (
    DEFAULT_ALGORITHM_PARAMETERS,
    feature_parameters,
    load_runtime_config,
    parse_config_text,
)
from algorithm.height_measurement import (
    HeightConfig,
    dilate_mask,
    erode_wire_mask_in_place,
    erode_measurement_mask_crop,
    fit_chip_plane,
    fit_local_platform_plane,
    fit_platform_plane,
    height_points_to_board,
    map_platform_rois,
    measure_mask_height,
    split_thin_mask,
)
from algorithm.registration import (
    RegistrationConfig,
    affine_coordinate_context,
    align_crop_mask_to_depth,
    platform_mask_2d,
    platform_mask_3d,
    refine_affine_registration,
    register_platform_masks,
)
from algorithm.tilt_correction import correct_smartray_depth, production_pitch_parameters, depth_coordinate_space
from transport.shared_memory import read_shared_memory, SharedImage
from algorithm.timing import timed_call, measure, note, record_duration, current_timings
from algorithm.instance_tasks import InstanceTasks
from algorithm.joint_geometry import JointGeometry


_segment = import_module("algorithm.3D.segment")
_geometry = import_module("algorithm.2D.defect_geometry")
_instance_geometry = import_module("algorithm.2D.instance_geometry")
_circle_roi = import_module("algorithm.2D.circle_roi")
_silver_inspection = import_module("algorithm.2D.silver_inspection")
_chip_alignment = import_module("algorithm.2D.chip_alignment")
_chip_anomaly = import_module("algorithm.2D.chip_anomaly")
_wire_roi_presence = import_module("algorithm.2D.wire_roi_presence")
DepthImageDecodeError = _segment.DepthImageDecodeError
PieceCountMismatchError = _segment.PieceCountMismatchError
SegmentationConfig = _segment.SegmentationConfig
decode_depth_png = _segment.decode_depth_png
segment_depth_image = _segment.segment_depth_image


class DetectBoard(JointGeometry):
    def __init__(self, log):
        self.log = log
        self._joint_ops = sys.modules[__name__]
        self.model = None
        self.model_path = None
        self.model_paths: Dict[str, str] = {}
        self.gpu_worker_count = 1
        self._gpu_worker_resources = []
        self._gpu_worker_slots = queue.Queue()
        self.config_path: str | None = None
        self.config: Dict[str, Any] = {}
        self.expected_piece_count = 10
        self.crop_roi = (1700, 0, 3900, 2700)
        self.min_component_area = 100
        self.wire_rois = [(280, 350, 1120, 520), (380, 1400, 1230, 2460)]
        self.rectangle_ratio_threshold = 0.75
        self.yolo_parameters = dict(DEFAULT_ALGORITHM_PARAMETERS)
        self.segmentation_config = SegmentationConfig(expected_piece_count=10)
        self.registration_config = RegistrationConfig()
        self.height_config = HeightConfig()
        self.dynamic_affine_config: Dict[str, Any] = {}
        self.pitch_config: Dict[str, Any] = {}
        # The streaming 2D path uses the same circle-relative crop as the
        # standalone 2D detector.  ``crop_roi`` remains the explicit fallback
        # for callers that do not provide the circle configuration.
        self.circle_crop_config: Dict[str, Any] | None = None
        self.silver_config: Dict[str, Any] = {}
        self.chip_alignment_config: Dict[str, Any] = {}
        self.chip_anomaly_config: Dict[str, Any] = {"configured": False}
        self.chip_anomaly_inspector = None
        # Streaming mode stores only decoded/preprocessed 2D data and YOLO
        # masks.  The source shared-memory block is read synchronously and is
        # never retained by Python.
        self._piece_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._piece_cache_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._instance_tasks = InstanceTasks()
        self._pitch_tasks = InstanceTasks(thread_name_prefix="pitch")

    def handle_init(self, req: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        if (self.model is not None or self.chip_anomaly_inspector is not None
                or self._instance_tasks.workers > 1 or self._pitch_tasks.workers > 1):
            self.release()
        self.config_path = str(req.get("config_path") or "") or None
        raw_model_paths = req.get("model_path")
        model_paths = raw_model_paths if isinstance(raw_model_paths, dict) else {}
        self.model_paths = {str(name): str(path) for name, path in model_paths.items()}
        self.model_path = str(model_paths.get("YOLO") or "") or None
        try:
            timed_call("init_config_ms", self._load_config, str(req.get("config_text") or ""))
            if not self.model_path:
                raise ValueError("model_path.YOLO is empty")
            timed_call("init_cuda_check_ms", self._validate_gpu_requirement)
            load_start = time.perf_counter()
            with measure("init_yolo_load_ms"):
                with redirect_stdout(io.StringIO()):
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
            model_load_ms = self.elapsed_ms(load_start)
            self.chip_anomaly_inspector = _chip_anomaly.ChipAnomalyInspector(
                self.chip_anomaly_config,
                self.model_paths,
            )
            anomaly_load_timings = timed_call("init_efficientad_load_ms", self.chip_anomaly_inspector.load)
            gpu_worker_load_timings = timed_call(
                "init_gpu_worker_replicas_ms", self._load_gpu_worker_replicas
            )
        except Exception as exc:
            self.release()
            return {
                "ok": False,
                "cmd": "init_algorithm",
                "stage": "board",
                "error": str(exc),
                "timings": {"total_init_ms": self.elapsed_ms(start)},
                "model_load_timings": {},
            }
        return {
            "ok": True,
            "cmd": "init_algorithm",
            "stage": "board",
            "message": "unified board detector ready",
            "timings": {"total_init_ms": self.elapsed_ms(start)},
            "model_load_timings": {
                "YOLO": model_load_ms,
                **anomaly_load_timings,
                **gpu_worker_load_timings,
            },
        }

    def configure_gpu_worker_count(self, count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("gpu_workers must be a positive integer")
        self.gpu_worker_count = count

    def _gpu_worker_resource(self, model, inspector, lock) -> Dict[str, Any]:
        stream = None
        # YOLO chooses its prediction device lazily, so its model can still be
        # on CPU here. Prefer the configured inference device, then either
        # loaded model's device (EfficientAD may run on GPU independently).
        device = None
        candidates = (
            self.yolo_parameters.get("device"),
            getattr(model, "device", None),
            getattr(inspector, "device", None),
        )
        for candidate in candidates:
            if isinstance(candidate, torch.device):
                candidate_device = candidate
            elif candidate is not None and str(candidate).strip().lower() not in {"", "auto"}:
                device_text = str(candidate).strip().lower().split(",", 1)[0].strip()
                if device_text.isdecimal():
                    device_text = f"cuda:{device_text}"
                try:
                    candidate_device = torch.device(device_text)
                except (RuntimeError, ValueError):
                    continue
            else:
                continue
            if candidate_device.type == "cuda":
                device = candidate_device
                break
        if (
            device is None
            and self.gpu_worker_count > 1
            and torch.cuda.is_available()
        ):
            device = torch.device("cuda", torch.cuda.current_device())
        if self.gpu_worker_count > 1 and isinstance(device, torch.device) and device.type == "cuda":
            stream = torch.cuda.Stream(device=device)
            # Inspector weights were loaded on the initializing thread's stream.
            stream.wait_stream(torch.cuda.current_stream(device))
        return {"model": model, "inspector": inspector, "lock": lock, "stream": stream}

    def _load_gpu_worker_replicas(self) -> Dict[str, float]:
        if self.model is None or self.chip_anomaly_inspector is None:
            raise RuntimeError("load the primary GPU models before creating worker replicas")
        from ultralytics import YOLO

        resources = [self._gpu_worker_resource(
            self.model, self.chip_anomaly_inspector, self._inference_lock
        )]
        timings: Dict[str, float] = {}
        try:
            for worker_index in range(2, self.gpu_worker_count + 1):
                started = time.perf_counter()
                with redirect_stdout(io.StringIO()):
                    model = YOLO(self.model_path)
                timings[f"YOLO_worker_{worker_index}"] = self.elapsed_ms(started)

                inspector = _chip_anomaly.ChipAnomalyInspector(
                    self.chip_anomaly_config, self.model_paths
                )
                try:
                    loaded = timed_call(
                        f"init_efficientad_worker_{worker_index}_ms", inspector.load
                    )
                except Exception:
                    inspector.release()
                    raise
                timings.update({f"{name}_worker_{worker_index}": value
                                for name, value in loaded.items()})
                try:
                    resource = self._gpu_worker_resource(
                        model, inspector, threading.RLock()
                    )
                except Exception:
                    inspector.release()
                    raise
                resources.append(resource)
        except Exception:
            for resource in resources[1:]:
                resource["inspector"].release()
            raise

        slots = queue.Queue()
        for resource in resources:
            slots.put(resource)
        self._gpu_worker_resources = resources
        self._gpu_worker_slots = slots
        return timings

    def _load_config(self, text: str) -> None:
        config = load_runtime_config(text, self.config_path)
        if not isinstance(config, dict):
            raise ValueError("board config must be an object")
        self.config = config

        common_2d = feature_parameters(config, "common_2d")
        common_3d = feature_parameters(config, "common_3d")
        input_config = _dict(common_3d.get("input"), "common_3d.parameters.input")
        self.expected_piece_count = int(input_config.get("piece_count", 10))
        if self.expected_piece_count <= 0:
            raise ValueError("input.piece_count must be positive")

        crop_config = _dict(common_2d.get("crop_2d"), "common_2d.parameters.crop_2d")
        self.crop_roi = _box(crop_config.get("roi"), "crop_2d.roi")

        mask_config = _dict(common_2d.get("mask", {}), "common_2d.parameters.mask")
        self.min_component_area = int(mask_config.get("min_component_area", 100))
        if self.min_component_area <= 0:
            raise ValueError("mask.min_component_area must be positive")

        wire_config = feature_parameters(config, "wire_bending_2d")
        raw_rois = crop_config.get("wire_rois")
        if not isinstance(raw_rois, list) or not raw_rois:
            raise ValueError("common_2d.parameters.crop_2d.wire_rois must be a non-empty list")
        self.wire_rois = [
            _box(value, "common_2d.parameters.crop_2d.wire_rois item") for value in raw_rois
        ]
        self.rectangle_ratio_threshold = float(
            wire_config.get("rectangle_ratio_threshold", 0.75)
        )
        if not 0 <= self.rectangle_ratio_threshold <= 1:
            raise ValueError("wire rectangle ratio threshold must be in [0, 1]")

        raw_parameters = _dict(common_2d.get("yolo", {}), "common_2d.parameters.yolo")
        allowed = {"imgsz", "conf", "iou", "max_det", "device", "half", "end2end"}
        self.yolo_parameters = {
            key: value
            for key, value in {**DEFAULT_ALGORITHM_PARAMETERS, **raw_parameters}.items()
            if key in allowed
        }

        segmentation_values = _dict(
            feature_parameters(config, "segmentation_3d"),
            "segmentation_3d.parameters",
        )
        self.segmentation_config = SegmentationConfig.from_dict(segmentation_values)
        if self.segmentation_config.expected_piece_count != self.expected_piece_count:
            raise ValueError(
                "input.piece_count and segmentation_3d.expected_piece_count must match"
            )
        self.registration_config = RegistrationConfig.from_dict(
            _dict(
                feature_parameters(config, "registration_3d"),
                "registration_3d.parameters",
            )
        )
        self.height_config = HeightConfig.from_dict(
            _dict(
                feature_parameters(config, "height_measurement_3d"),
                "height_measurement_3d.parameters",
            )
        )
        circle_crop = dict(
            _dict(common_2d.get("crop"), "common_2d.parameters.crop")
        )
        circle_crop["rectangle_roi_offsets"] = wire_config.get("roi_offsets", [])
        self.circle_crop_config = self._parse_circle_crop_config(circle_crop)
        raw_overflow = feature_parameters(config, "silver_overflow_2d", required=False)
        raw_continuity = feature_parameters(config, "silver_continuity_2d", required=False)
        self.silver_config = {
            "overflow": raw_overflow,
            "continuity": raw_continuity,
        }
        if raw_overflow is not None:
            _silver_inspection.overflow_settings(raw_overflow)
        if raw_continuity is not None:
            _silver_inspection.continuity_settings(raw_continuity)
        raw_chip_alignment = feature_parameters(
            config, "chip_rotation_2d", required=False
        )
        self.chip_alignment_config = raw_chip_alignment
        if raw_chip_alignment is not None:
            threshold = float(raw_chip_alignment.get("max_angle_deg", 3.0))
            if not 0.0 <= threshold <= 45.0:
                raise ValueError("chip_alignment_2d.max_angle_deg must be in [0, 45]")
        raw_chip_anomaly = feature_parameters(
            config, "chip_anomaly_2d", required=False
        )
        self.chip_anomaly_config = _chip_anomaly.validate_config(raw_chip_anomaly)
        self.dynamic_affine_config = dict(
            feature_parameters(config, "dynamic_affine_3d") or {}
        )
        self.pitch_config = production_pitch_parameters(config, self.height_config.gray_offset,
                                                        self.height_config.vertical_resolution_mm)
        threading_values = _dict(common_3d.get("threading", {}), "common_3d.parameters.threading")
        self._instance_tasks.configure(threading_values.get("instance_workers", 16))
        self._pitch_tasks.configure(self.pitch_config.get("projection_workers", 1))

    def _validate_gpu_requirement(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("board detection requires an available CUDA GPU")
        if str(self.yolo_parameters.get("device", "")).lower() == "cpu":
            raise ValueError("yolo.parameters.device must select a CUDA device")

    # ------------------------------------------------------------------
    # Streaming board protocol
    # ------------------------------------------------------------------
    def handle_detect_2d_piece(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous compatibility entry; the server schedules the phases separately."""
        prepared = None
        try:
            prepared = self.prepare_2d_piece(req)
            self.infer_2d_piece(prepared)
            return self.finish_2d_piece(prepared)
        except (ValueError, OSError, RuntimeError) as exc:
            return self._stream_failure(
                "2d_piece_failed", str(exc),
                board_id=str(req.get("board_id") or ""),
                piece_id=self._optional_piece_id(req),
            )
        finally:
            if prepared is not None:
                self.release_prepared_2d(prepared)

    def prepare_2d_piece(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """CPU-only read/decode, circle ROI and registration-mask preparation."""
        start = time.perf_counter()
        board_id = self._stream_board_id(req)
        piece_id = self._stream_piece_id(req)
        read_start = time.perf_counter()
        owner = None
        image_format = str(req.get("image_format", "encoded")).lower()
        if image_format == "bgr8" or (image_format == "raw" and int(req.get("image_channels", 3)) == 3):
            owner = timed_call("input_shared_view_ms", SharedImage, req)
            image = owner.image
        else:
            image = timed_call("input_read_decode_ms", self._read_2d, req)
        read_decode_ms = self.elapsed_ms(read_start)
        circle_start = time.perf_counter()
        try:
            crop, crop_box, circle, roi_boxes = timed_call("roi_prepare_ms", self._stream_crop, image)
            crop = timed_call("roi_copy_ms", crop.copy)
            circle_crop_ms = self.elapsed_ms(circle_start)
            registration_mask = timed_call("platform_mask_2d_ms", platform_mask_2d, image, self.registration_config)
        except Exception:
            image = None
            if owner is not None:
                owner.close()
            raise
        return dict(start=start, board_id=board_id, piece_id=piece_id,
                    image=image, crop=crop, crop_box=crop_box, circle=circle,
                    roi_boxes=roi_boxes, registration_mask=registration_mask,
                    read_decode_ms=read_decode_ms, circle_crop_ms=circle_crop_ms, image_owner=owner,
                    threshold=req.get("threshold"), gray_lock=threading.Lock())

    @staticmethod
    def release_prepared_2d(prepared):
        owner = prepared.pop("image_owner", None)
        prepared.pop("image", None)
        if owner is not None:
            owner.close()

    def infer_2d_piece(self, prepared: Dict[str, Any]) -> None:
        """Run one piece on an independently owned GPU model replica."""
        crop = prepared["crop"]
        image, crop_box = prepared["image"], prepared["crop_box"]
        note("input_shape", image_shape=list(image.shape), roi_shape=list(crop.shape),
             imgsz=self.yolo_parameters.get("imgsz"))
        return_resource = bool(self._gpu_worker_resources)
        slot_started = time.perf_counter()
        if return_resource:
            resource = self._gpu_worker_slots.get()
        else:
            # Keep the direct/synchronous compatibility path usable in CPU-only
            # tests and legacy callers that set the primary model directly.
            if (self.model is None or self.chip_anomaly_inspector is None
                    or self.gpu_worker_count != 1):
                raise RuntimeError("GPU model resources are not initialized")
            resource = {
                "model": self.model,
                "inspector": self.chip_anomaly_inspector,
                "lock": self._inference_lock,
            }
        record_duration("gpu_model_slot_wait_ms", self.elapsed_ms(slot_started))
        model, inspector, model_lock = (
            resource["model"], resource["inspector"], resource["lock"]
        )
        stream = resource.get("stream")
        try:
            lock_start = time.perf_counter()
            with model_lock, (torch.cuda.stream(stream) if stream is not None else nullcontext()):
                record_duration("model_lock_wait_ms", self.elapsed_ms(lock_start))
                if model is None or inspector is None:
                    raise RuntimeError("GPU model resources are not initialized")
                inference_start = time.perf_counter()
                timed_call("cuda_before_yolo_ms", self._synchronize_model, model, stream)
                predictions = timed_call(
                    "yolo_predict_wall_ms", self._predict_batch,
                    [crop], model, model_lock,
                )
                timed_call("cuda_after_yolo_ms", self._synchronize_model, model, stream)
                inference_ms = self.elapsed_ms(inference_start)
                if len(predictions) != 1:
                    raise RuntimeError(
                        f"YOLO returned {len(predictions)} results for one 2D image"
                    )
                speed = getattr(predictions[0], "speed", None)
                if isinstance(speed, dict):
                    for stage in ("preprocess", "inference", "postprocess"):
                        value = speed.get(stage)
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            record_duration(f"yolo_reported_{stage}_ms", float(value))

                instances = timed_call(
                    "mask_extract_to_cpu_ms", self._extract_instances,
                    predictions[0], crop.shape[:2],
                )
                chip_anomaly_diagnostics = timed_call(
                    "efficientad_total_ms", inspector.inspect,
                    image, instances, crop_box, threshold=prepared.get("threshold"),
                )
                device = getattr(model, "device", None)
                if isinstance(device, torch.device) and device.type == "cuda":
                    note("gpu_memory", allocated_mb=round(torch.cuda.memory_allocated(device) / 1048576, 2),
                         reserved_mb=round(torch.cuda.memory_reserved(device) / 1048576, 2),
                         instances=len(instances), gpu_worker_count=self.gpu_worker_count)
            prepared.update(instances=instances, inference_ms=inference_ms,
                            chip_anomaly_diagnostics=chip_anomaly_diagnostics)
        finally:
            try:
                if stream is not None:
                    sync_started = time.perf_counter()
                    stream.synchronize()
                    record_duration("gpu_worker_stream_sync_ms", self.elapsed_ms(sync_started))
            finally:
                if return_resource:
                    self._gpu_worker_slots.put(resource)

    def prepare_silver_boundary(self, prepared: Dict[str, Any]):
        """Build the silver boundary from the existing ROI or template."""
        raw_boundary = (self.silver_config.get("overflow") or {}).get("boundary") or {}
        roi_inset = str(raw_boundary.get("mode", "")).strip().casefold() == "roi_inset"
        return _silver_inspection.prepare_boundary(
            self.silver_config, crop_box=prepared["crop_box"], circle=prepared["circle"],
            image_shape=prepared["crop"].shape[:2], image=prepared["image"],
            target_gray=None if roi_inset else self._prepared_gray_image(prepared),
            base_dir=self._silver_config_base_dir())

    @staticmethod
    def _prepared_gray_image(prepared: Dict[str, Any]) -> np.ndarray:
        # Silver SIFT and later 3D registration consume the same source gray.
        # Compute it once regardless of which parallel branch reaches it first.
        image = prepared.get("gray_image")
        if image is not None:
            return image
        lock = prepared.get("gray_lock")
        if lock is None:
            image = prepared["gray_image"] = cv2.cvtColor(prepared["image"], cv2.COLOR_BGR2GRAY)
            return image
        with lock:
            image = prepared.get("gray_image")
            if image is None:
                image = timed_call("cache_gray_image_ms", cv2.cvtColor,
                                   prepared["image"], cv2.COLOR_BGR2GRAY)
                prepared["gray_image"] = image
            return image

    def prepare_joint_piece(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        """Build immutable joint inputs after inference, before heavy diagnostics."""
        piece_id = prepared["piece_id"]
        image, crop, crop_box, circle, roi_boxes = (
            prepared[key] for key in ("image", "crop", "crop_box", "circle", "roi_boxes"))
        registration_mask = prepared["registration_mask"]
        instances = prepared["instances"]
        chip_anomaly_diagnostics = prepared["chip_anomaly_diagnostics"]
        wire_measurements = timed_call("wire_bending_ms", self._measure_2d_wire_instances,
            instances, roi_boxes, crop_box
        )
        wire_roi_checks = timed_call("wire_roi_presence_ms", _wire_roi_presence.check_wire_roi_presence,
            instances, roi_boxes, crop_box[:2]
        )
        chip_diagnostics = timed_call("chip_rotation_ms", _chip_alignment.measure_chip_alignment,
            instances, self.chip_alignment_config, crop_box[:2]
        )
        entry = {
            "piece_id": piece_id,
            "gray_image": self._prepared_gray_image(prepared),
            "image_shape": [int(image.shape[1]), int(image.shape[0])],
            "crop_box": tuple(int(value) for value in crop_box),
            "circle": circle,
            "roi_boxes": [tuple(int(value) for value in item) for item in roi_boxes],
            "registration_mask": registration_mask,
            "instances": instances,
            "wire_measurements": wire_measurements,
            "wire_roi_checks": wire_roi_checks,
            "chip_rotation_2d": chip_diagnostics,
            "chip_anomaly_2d": chip_anomaly_diagnostics,
            "received_at": time.time(),
        }
        prepared["joint_entry"] = entry
        return {**entry, "instances": [dict(item) for item in instances]}

    def finish_2d_piece(self, prepared: Dict[str, Any], *, publish_cache: bool = True,
                        on_joint_ready=None):
        """Publish full 2D results only after all independent branches finish."""
        start, board_id, piece_id = (prepared[key] for key in ("start", "board_id", "piece_id"))
        image, crop, crop_box, circle, roi_boxes = (
            prepared[key] for key in ("image", "crop", "crop_box", "circle", "roi_boxes"))
        read_decode_ms, circle_crop_ms, inference_ms = (
            prepared[key] for key in ("read_decode_ms", "circle_crop_ms", "inference_ms"))
        postprocess_start = time.perf_counter()
        if "joint_entry" not in prepared:
            self.prepare_joint_piece(prepared)
        entry = prepared["joint_entry"]
        instances = entry["instances"]
        wire_measurements, wire_roi_checks = entry["wire_measurements"], entry["wire_roi_checks"]
        chip_diagnostics, chip_anomaly_diagnostics = entry["chip_rotation_2d"], entry["chip_anomaly_2d"]
        if on_joint_ready is not None:
            on_joint_ready({**entry, "instances": [dict(item) for item in instances]})
        computed = prepared.get("cpu_2d_result")
        if computed is None:
            # Direct synchronous entry; formal streaming receives this work
            # from the dedicated model-free CPU process pool below.
            diagnostics = timed_call("silver_mask_checks_ms", _silver_inspection.inspect_masks,
                instances, crop.shape[:2], self.silver_config, crop_box=crop_box, circle=circle,
                image=image, base_dir=self._silver_config_base_dir(),
                prepared_boundary=prepared.get("silver_boundary"))
            silver_timings = current_timings()
            record_duration("silver_total_ms", silver_timings.get("silver_mask_checks_ms", 0.0)
                            + silver_timings.get("silver_prepare_phase_ms", 0.0))
            diagnostics = timed_call("silver_geometry_offset_ms", _silver_inspection.offset_geometry, diagnostics, crop_box[:2])
            postprocess_ms = self.elapsed_ms(postprocess_start)
        else:
            diagnostics = computed["diagnostics"]
            postprocess_ms = computed["mask_postprocess_ms"]

        entry.update(silver_overflow_2d=diagnostics["silver_overflow"],
                     silver_continuity_2d=diagnostics["disconnection"])
        replaced = False
        cache_state = {}
        if publish_cache:
            with self._piece_cache_lock:
                board_cache = self._piece_cache.setdefault(board_id, {})
                replaced = piece_id in board_cache
                board_cache[piece_id] = entry
                cache_state = self._cache_state_locked(board_id)
        public_instances = (computed["public_instances"] if computed is not None else
                            timed_call("public_2d_geometry_ms", self._public_2d_instances, instances, crop_box))
        total_ms = self.elapsed_ms(start)
        checks = [bool(item.get("is_ok")) for item in instances]
        checks.extend(bool(item.get("is_ok")) for item in wire_roi_checks)
        if chip_diagnostics.get("enabled") and chip_diagnostics.get("is_ok") is not True:
            checks.append(False)
        if (chip_anomaly_diagnostics.get("enabled")
                and chip_anomaly_diagnostics.get("is_ok") is not True):
            checks.append(False)
        for check in (diagnostics.get("silver_overflow"), diagnostics.get("disconnection")):
            if (
                diagnostics.get("enabled")
                and check
                and check.get("status") not in {None, "disabled"}
                and check.get("is_ok") is not True
            ):
                checks.append(False)
        response = {
            "ok": True,
            "cmd": "detect_2d_piece",
            "stage": "2d",
            "board_id": board_id,
            "piece_id": piece_id,
            "results_2d": {
                "is_ok": bool(checks) and all(checks),
                "circle": ({"x": circle[0], "y": circle[1], "radius": circle[2]}
                           if circle is not None else None),
                "crop_box": list(crop_box),
                "wire_rois_2d": [list(roi) for roi in roi_boxes],
                "rectangles": [
                    {"points": item.get("rectangle"),
                     "mask_ratio": item.get("rectangle_ratio", 0.0),
                     "is_ok": item.get("is_ok", False)}
                    for item in instances if item.get("rectangle") is not None
                ],
                "instances": public_instances,
                "wire_measurements": wire_measurements,
                "wire_roi_checks": wire_roi_checks,
                "wire_bending_2d": {
                    "enabled": True,
                    "is_ok": bool(wire_roi_checks) and all(
                        item["is_ok"] for item in wire_roi_checks
                    ),
                    "roi_checks": wire_roi_checks,
                    "measurements": wire_measurements,
                },
                "chip_rotation_2d": chip_diagnostics,
                "chip_anomaly_2d": chip_anomaly_diagnostics,
                "silver_overflow_2d": diagnostics["silver_overflow"],
                "silver_continuity_2d": diagnostics["disconnection"],
            },
            "cache_state": cache_state,
            "timings": {
                "shared_memory_read_decode_ms": read_decode_ms,
                "circle_crop_ms": circle_crop_ms,
                "yolo_inference_ms": inference_ms,
                "mask_postprocess_ms": postprocess_ms,
                "total_detect_ms": total_ms,
            },
            "elapsed_ms": total_ms,
            "message": (
                "2D piece segmented and cached"
                + (" (replaced previous piece)" if replaced else "")
            ),
        }
        return response if publish_cache else (response, entry)

    def handle_detect_3d_board(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Consume one 3D board and the ten cached 2D piece results."""
        start = time.perf_counter()
        try:
            board_id = self._stream_board_id(req)
        except ValueError as exc:
            return self._stream_failure("invalid_board_id", str(exc), board_id="")
        entries: Dict[int, Dict[str, Any]] = {}
        cache_count = 0
        try:
            with self._piece_cache_lock:
                board_cache = self._piece_cache.get(board_id, {})
                expected_ids = set(range(1, self.expected_piece_count + 1))
                missing = sorted(expected_ids - set(board_cache))
                if missing:
                    return self._stream_failure(
                        "missing_2d_pieces",
                        f"board {board_id!r} is missing 2D piece IDs: {missing}",
                        board_id=board_id,
                        missing_piece_ids=missing,
                    )
                # Hold references only for the duration of this request.  The
                # cache is removed in finally after the board is consumed.
                entries = {piece_id: board_cache[piece_id] for piece_id in expected_ids}
                cache_count = len(entries)

            read_start = time.perf_counter()
            depth = self._read_3d(req)
            read_decode_ms = self.elapsed_ms(read_start)
            pitch_start = time.perf_counter()
            depth, pitch_correction = self.prepare_3d_depth(depth)
            pitch_ms = self.elapsed_ms(pitch_start)

            segment_start = time.perf_counter()
            depth_pieces = segment_depth_image(depth, self.segmentation_config)
            segmentation_ms = self.elapsed_ms(segment_start)
            if len(depth_pieces) != self.expected_piece_count:
                raise PieceCountMismatchError(
                    self.expected_piece_count, len(depth_pieces)
                )

            align_start = time.perf_counter()
            results = []
            for index, piece in enumerate(depth_pieces):
                piece_id = index + 1
                results.append(self.measure_prepared_piece(piece_id, entries[piece_id], depth, piece, pitch_correction))
            alignment_ms = self.elapsed_ms(align_start)
            total_ms = self.elapsed_ms(start)
            return {
                "ok": True,
                "cmd": "detect_3d_board",
                "stage": "3d",
                "board_id": board_id,
                "batch_id": board_id,
                "results_board": {
                    "pitch_correction": pitch_correction,
                    "depth_coordinate_space": depth_coordinate_space(pitch_correction),
                    "registration_ok": all(
                        piece["registration"]["is_valid"] for piece in results
                    ),
                    "is_ok": all(
                        piece["registration"]["is_valid"] and piece["is_ok"]
                        for piece in results
                    ),
                    "pieces": results,
                },
                "released": {
                    "board_id": board_id,
                    "released_2d_count": cache_count,
                    "released_3d": True,
                },
                "elapsed_ms": total_ms,
                "timings": {
                    "shared_memory_read_decode_ms": read_decode_ms,
                    "smartray_pitch_ms": pitch_ms,
                    "segmentation_3d_ms": segmentation_ms,
                    "registration_and_mask_alignment_ms": alignment_ms,
                    "total_detect_ms": total_ms,
                },
                "message": "3D board aligned with cached 2D piece masks",
            }
        except PieceCountMismatchError as exc:
            return self._stream_failure(
                "piece_count_mismatch", str(exc), board_id=board_id
            )
        except (ValueError, OSError, RuntimeError) as exc:
            return self._stream_failure(
                "3d_board_failed", str(exc), board_id=board_id
            )
        finally:
            # Once all ten 2D entries were present, this board is terminal:
            # release them even when 3D decoding/alignment fails.  A missing
            # piece returns before entries are captured and remains retryable.
            if cache_count == self.expected_piece_count:
                with self._piece_cache_lock:
                    self._piece_cache.pop(board_id, None)

    def measure_prepared_piece(self, piece_id: int, entry: Dict[str, Any],
                               depth: np.ndarray, piece: Dict[str, Any],
                               pitch_correction: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Synchronous compatibility entry; streaming schedules these stages separately."""
        registration = self.register_prepared_piece(piece_id, entry, depth, piece, pitch_correction)
        return self.measure_registered_piece(piece_id, entry, depth, piece, registration, pitch_correction)

    def register_prepared_piece(self, piece_id: int, entry: Dict[str, Any],
                               depth: np.ndarray, piece: Dict[str, Any],
                               pitch_correction: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Fit one piece's matrix without waiting for mapping or height work."""
        from algorithm.registration_worker import register_arrays
        return register_arrays(entry["gray_image"], entry["registration_mask"],
            depth[int(piece["y1"]):int(piece["y2"])], self.registration_config,
            self.dynamic_affine_config, pitch_correction,
            operations=(platform_mask_3d, register_platform_masks, refine_affine_registration))

    def submit_registration(self, preparation_pool, inputs, piece_id, entry, depth, piece,
                            correction, timings, metadata, is_current):
        """Formal streaming registration always executes in model-free processes."""
        service = getattr(self, "registration_process", None)
        if service is None:
            raise RuntimeError("registration processes must be initialized before board detection")
        return service.submit(preparation_pool, inputs, piece_id, entry, depth, piece,
            correction, self.registration_config, self.dynamic_affine_config, timings,
            metadata, is_current, log=self.log)

    def submit_2d_postprocess(self, preparation_pool, prepared, timings, metadata, is_current):
        service = getattr(self, "cpu_2d_process", None)
        if service is None:
            raise RuntimeError("2D CPU processes must be initialized before streaming detection")
        return service.submit_piece(preparation_pool, prepared, self.silver_config,
            self._silver_config_base_dir(), timings, metadata, is_current, self.log)

    def submit_joint_measurement(self, preparation_pool, inputs, piece_id, entry, depth,
                                 piece, registration, correction, timings, metadata, is_current):
        service = getattr(self, "joint_measurement_process", None)
        if service is None:
            raise RuntimeError("3D measurement processes must be initialized before streaming detection")
        return service.submit_piece(preparation_pool, inputs, piece_id, entry, depth, piece,
            registration, correction, self.height_config, self.min_component_area,
            self.crop_roi, self.wire_rois, timings, metadata, is_current, self.log)


    @staticmethod
    def finalize_joint_piece(piece: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
        """Join completed diagnostics without mutating an in-flight 3D result."""
        diagnostics_ok = all(
            check.get("is_ok") is True
            for check in (
                entry.get("silver_overflow_2d", {}),
                entry.get("silver_continuity_2d", {}),
                entry.get("chip_anomaly_2d", {}),
                entry.get("chip_rotation_2d", {}),
            )
            if isinstance(check, dict) and check.get("status") not in {None, "disabled"}
        )
        diagnostics_present = any(
            check.get("status") not in {None, "disabled"}
            for check in (
                entry.get("silver_overflow_2d", {}),
                entry.get("silver_continuity_2d", {}),
                entry.get("chip_anomaly_2d", {}),
                entry.get("chip_rotation_2d", {}),
            )
            if isinstance(check, dict)
        )
        return {
            **piece,
            "is_ok": bool(piece["is_ok"] and (not diagnostics_present or diagnostics_ok)),
            "chip_anomaly_2d": entry.get("chip_anomaly_2d", {"status": "disabled", "is_ok": None}),
            "silver_overflow_2d": entry.get("silver_overflow_2d", {"status": "disabled", "is_ok": None}),
            "silver_continuity_2d": entry.get("silver_continuity_2d", {"status": "disabled", "is_ok": None}),
        }

    def handle_release_board(self, req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            board_id = self._stream_board_id(req)
        except ValueError as exc:
            return {
                "ok": False,
                "cmd": "release_board",
                "stage": "board",
                "error": "invalid_board_id",
                "message": str(exc),
            }
        with self._piece_cache_lock:
            removed = self._piece_cache.pop(board_id, {})
        return {
            "ok": True,
            "cmd": "release_board",
            "stage": "board",
            "board_id": board_id,
            "released": {
                "board_id": board_id,
                "released_2d_count": len(removed),
                "released_3d": False,
            },
            "message": "board cache released",
        }

    def _cache_state_locked(self, board_id: str) -> Dict[str, Any]:
        pieces = sorted(self._piece_cache.get(board_id, {}))
        return {
            "board_id": board_id,
            "expected_piece_count": self.expected_piece_count,
            "received_piece_count": len(pieces),
            "piece_ids": pieces,
            "ready_for_3d": len(pieces) == self.expected_piece_count,
        }

    def _stream_board_id(self, req: Dict[str, Any]) -> str:
        value = str(req.get("board_id") or req.get("batch_id") or "").strip()
        if not value:
            raise ValueError("board_id is required for streaming detection")
        return value

    def _silver_config_base_dir(self) -> Path | None:
        if not self.config_path:
            return None
        parent = Path(self.config_path).resolve().parent
        if parent.name == "2D":
            return parent
        if parent.name == "3D":
            return parent.parent / "2D"
        return parent / "2D"

    def _stream_piece_id(self, req: Dict[str, Any]) -> int:
        value = int(req.get("piece_id", 0))
        if not 1 <= value <= self.expected_piece_count:
            raise ValueError(
                f"piece_id must be in 1..{self.expected_piece_count}, received {value}"
            )
        return value

    @staticmethod
    def _optional_piece_id(req: Dict[str, Any]) -> int | None:
        try:
            value = int(req.get("piece_id", 0))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _stream_failure(
        self,
        error: str,
        message: str,
        *,
        board_id: str,
        piece_id: int | None = None,
        missing_piece_ids: list[int] | None = None,
    ) -> Dict[str, Any]:
        with self._piece_cache_lock:
            state = self._cache_state_locked(board_id)
        response: Dict[str, Any] = {
            "ok": False,
            "cmd": "detect_2d_piece" if piece_id is not None else "detect_3d_board",
            "stage": "2d" if piece_id is not None else "3d",
            "error": error,
            "message": message,
            "board_id": board_id,
            "cache_state": state,
        }
        if piece_id is not None:
            response["piece_id"] = piece_id
        if missing_piece_ids is not None:
            response["missing_piece_ids"] = missing_piece_ids
        return response

    def handle_detect(self, req: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            descriptors = self._validate_descriptors(req.get("images_2d"))
            depth_descriptor = _dict(req.get("image_3d"), "image_3d")

            read_start = time.perf_counter()
            crops = []
            registration_masks = []
            image_shapes = []
            crop_metadata = []
            gray_images = []
            for descriptor in descriptors:
                image = self._read_2d(descriptor)
                image_shapes.append([int(image.shape[1]), int(image.shape[0])])
                registration_masks.append(platform_mask_2d(image, self.registration_config))
                crop, crop_box, circle, roi_boxes = self._stream_crop(image)
                crops.append(crop.copy())
                crop_metadata.append((crop_box, circle, roi_boxes))
                gray_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            depth = self._read_3d(depth_descriptor)
            shared_memory_read_decode_ms = self.elapsed_ms(read_start)
            pitch_start = time.perf_counter()
            depth, pitch_correction = self.prepare_3d_depth(depth)
            pitch_ms = self.elapsed_ms(pitch_start)

            segment_start = time.perf_counter()
            depth_pieces = segment_depth_image(depth, self.segmentation_config)
            segmentation_ms = self.elapsed_ms(segment_start)

            if self.model is None:
                raise RuntimeError("YOLO model is not initialized")
            inference_start = time.perf_counter()
            self._synchronize_model()
            predictions = self._predict_batch(crops)
            self._synchronize_model()
            inference_ms = self.elapsed_ms(inference_start)
            if len(predictions) != self.expected_piece_count:
                raise RuntimeError(
                    f"YOLO returned {len(predictions)} results for "
                    f"{self.expected_piece_count} images"
                )

            align_start = time.perf_counter()
            results = []
            for index, (descriptor, crop, prediction, piece) in enumerate(
                zip(descriptors, crops, predictions, depth_pieces)
            ):
                y1, y2 = int(piece["y1"]), int(piece["y2"])
                local_depth = depth[y1:y2]
                registration = register_platform_masks(
                    registration_masks[index],
                    platform_mask_3d(local_depth, self.registration_config),
                    self.registration_config,
                )
                registration = refine_affine_registration(
                    gray_images[index], local_depth, registration, self.dynamic_affine_config,
                    context=affine_coordinate_context(gray_images[index].shape, local_depth.shape, pitch_correction),
                )
                crop_box, circle, roi_boxes = crop_metadata[index]
                instances = self._extract_instances(prediction, crop.shape[:2])
                aligned_masks = self._align_instances(instances, local_depth.shape, registration, crop_box)
                measurement = self._measure_piece(
                    local_depth, aligned_masks, registration,
                    crop_box=crop_box, circle=circle,
                )
                results.append(
                    height_points_to_board({
                        "piece_id": int(descriptor["piece_id"]),
                        "image_2d_size": image_shapes[index],
                        "crop_2d": list(crop_box),
                        "wire_rois_2d": [list(roi) for roi in roi_boxes],
                        "depth_start_row": y1,
                        "depth_end_row": y2,
                        "pitch_correction": pitch_correction,
                        "depth_coordinate_space": depth_coordinate_space(pitch_correction),
                        "registration": registration,
                        "is_ok": measurement["is_ok"],
                        "reference_planes": measurement["reference_planes"],
                        "wire_measurements": measurement["wire_measurements"],
                        "solder_measurements": measurement["solder_measurements"],
                        "masks": [self._public_instance(item) for item in aligned_masks],
                    })
                )
            alignment_ms = self.elapsed_ms(align_start)
            elapsed_ms = self.elapsed_ms(start)
            return {
                "ok": True,
                "cmd": "detect_board",
                "stage": "board",
                "batch_id": str(req.get("batch_id") or ""),
                "results_board": {
                    "pitch_correction": pitch_correction,
                    "depth_coordinate_space": depth_coordinate_space(pitch_correction),
                    "registration_ok": all(
                        piece["registration"]["is_valid"] for piece in results
                    ),
                    "is_ok": all(
                        piece["registration"]["is_valid"] and piece["is_ok"]
                        for piece in results
                    ),
                    "pieces": results,
                },
                "elapsed_ms": elapsed_ms,
                "timings": {
                    "shared_memory_read_decode_ms": shared_memory_read_decode_ms,
                    "smartray_pitch_ms": pitch_ms,
                    "segmentation_3d_ms": segmentation_ms,
                    "yolo_batch_inference_ms": inference_ms,
                    "registration_and_mask_alignment_ms": alignment_ms,
                    "total_detect_ms": elapsed_ms,
                },
                "message": "board masks aligned to 3D pieces",
            }
        except PieceCountMismatchError as exc:
            return self._failure("piece_count_mismatch", str(exc))
        except (DepthImageDecodeError, ValueError, OSError, RuntimeError) as exc:
            return self._failure("board_detection_failed", str(exc))

    def _validate_descriptors(self, value: Any) -> list[Dict[str, Any]]:
        if not isinstance(value, list):
            raise ValueError("images_2d must be a list")
        if len(value) != self.expected_piece_count:
            raise ValueError(
                f"expected {self.expected_piece_count} 2D images, received {len(value)}"
            )
        descriptors = [_dict(item, "images_2d item") for item in value]
        descriptors.sort(key=lambda item: int(item.get("piece_id", 0)))
        piece_ids = [int(item.get("piece_id", 0)) for item in descriptors]
        expected_ids = list(range(1, self.expected_piece_count + 1))
        if piece_ids != expected_ids:
            raise ValueError(f"2D piece_id values must be exactly {expected_ids}")
        return descriptors

    def _read_2d(self, descriptor: Dict[str, Any]) -> np.ndarray:
        size = int(descriptor.get("image_size", 0))
        name = str(descriptor.get("shm_name", ""))
        if not name or size <= 0:
            raise ValueError("each 2D image requires shm_name and image_size")
        data = timed_call("input_shared_memory_read_ms", read_shared_memory, name, size)
        image_format = str(descriptor.get("image_format", "encoded")).lower()
        if image_format in {"rgb8", "bgr8", "gray8", "raw"}:
            width = int(descriptor.get("image_width", 0))
            height = int(descriptor.get("image_height", 0))
            channels = int(descriptor.get("image_channels", 3) or 3)
            required = width * height * channels
            if width <= 0 or height <= 0 or channels not in (1, 3) or len(data) < required:
                raise ValueError("raw 2D image requires valid dimensions and byte length")
            image = np.frombuffer(data[:required], dtype=np.uint8).reshape(
                height, width, channels
            )
            if channels == 1:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image_format == "rgb8" else image.copy()
        image = timed_call("input_image_decode_ms", cv2.imdecode,
                           np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to decode 2D image for piece {descriptor.get('piece_id')}")
        return image

    def prepare_3d_depth(self, depth: np.ndarray):
        return timed_call("smartray_pitch_ms", correct_smartray_depth, depth, self.pitch_config,
                          tasks=self._pitch_tasks)

    @staticmethod
    def _read_3d(descriptor: Dict[str, Any]) -> np.ndarray:
        size = int(descriptor.get("data_size", 0))
        name = str(descriptor.get("shm_name", ""))
        if not name or size <= 0:
            raise ValueError("image_3d requires shm_name and data_size")
        data = timed_call("depth_shared_memory_read_ms", read_shared_memory, name, size)
        return timed_call("depth_image_decode_ms", decode_depth_png,
                          data, str(descriptor.get("data_format", "png")))

    def _fixed_crop(self, image: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.crop_roi
        height, width = image.shape[:2]
        if x2 > width or y2 > height:
            raise ValueError(
                f"crop_2d.roi {self.crop_roi} exceeds image shape {(height, width)}"
            )
        return image[y1:y2, x1:x2]

    @staticmethod
    def _parse_circle_crop_config(value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("crop must be an object")
        offsets = value.get("detection_roi_offsets", value.get("roi_offsets"))
        if not isinstance(offsets, (list, tuple)) or len(offsets) != 4:
            raise ValueError("crop.detection_roi_offsets must contain four numbers")
        search = value.get("circle_search_roi")
        if not isinstance(search, (list, tuple)) or len(search) != 4:
            raise ValueError("crop.circle_search_roi must contain four integers")
        search = tuple(int(item) for item in search)
        if search[2] <= search[0] or search[3] <= search[1]:
            raise ValueError("crop.circle_search_roi coordinates are invalid")
        raw_rois = value.get("rectangle_roi_offsets", value.get("roi_offsets_list", []))
        if not raw_rois:
            raw_rois = [offsets]
        if not isinstance(raw_rois, list):
            raise ValueError("crop.rectangle_roi_offsets must be a list")
        rois = []
        for item in raw_rois:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                raise ValueError("each rectangle ROI must contain four numbers")
            rois.append(tuple(float(number) for number in item))
        radius = int(value.get("circle_radius", 560))
        tolerance = float(value.get("radius_tolerance_pixels", 10))
        if radius <= 0 or tolerance < 0 or tolerance >= radius:
            raise ValueError("invalid circle radius/tolerance")
        return {
            "circle_search_roi": search,
            "detection_roi_offsets": tuple(float(number) for number in offsets),
            "rectangle_roi_offsets": rois,
            "circle_radius": radius,
            "radius_tolerance_pixels": tolerance,
        }

    def _stream_crop(self, image: np.ndarray):
        """Crop one streaming image using its runtime circle when configured."""
        if self.circle_crop_config is None:
            crop = self._fixed_crop(image)
            return crop, tuple(self.crop_roi), None, [tuple(item) for item in self.wire_rois]
        config = self.circle_crop_config
        circle = timed_call("circle_find_ms", _circle_roi.find_circle,
            image,
            fixed_radius=config["circle_radius"],
            radius_tolerance=config["radius_tolerance_pixels"] / config["circle_radius"],
            search_roi=config["circle_search_roi"],
        )
        crop, box = timed_call("circle_detection_crop_ms", _circle_roi.crop_from_config,
            image, circle, {"roi_offsets": config["detection_roi_offsets"]}
        )
        roi_boxes = []
        for offsets in config["rectangle_roi_offsets"]:
            _, full = timed_call("circle_wire_roi_ms", _circle_roi.crop_from_config,
                                 image, circle, {"roi_offsets": offsets})
            roi_boxes.append(
                (full[0] - box[0], full[1] - box[1], full[2] - box[0], full[3] - box[1])
            )
        return crop, box, circle, roi_boxes

    def _predict_batch(self, crops: Sequence[np.ndarray], model=None, lock=None):
        parameters = {**self.yolo_parameters, "verbose": False, "retina_masks": True}
        model = model if model is not None else self.model
        lock = lock if lock is not None else self._inference_lock
        with lock:
            return list(model.predict(source=list(crops), **parameters))

    def _synchronize_model(self, model=None, stream=None) -> None:
        if stream is not None:
            stream.synchronize()
            return
        model = model if model is not None else self.model
        device = getattr(model, "device", None)
        if isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _extract_instances(self, result: Any, crop_shape: tuple[int, int]):
        masks_object = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if masks_object is None or boxes is None:
            return []
        masks = masks_object.data
        if masks.shape[-2:] != crop_shape:
            masks = torch_functional.interpolate(
                masks.unsqueeze(1), size=crop_shape, mode="nearest"
            ).squeeze(1)
        masks = masks > 0.5
        # Convert all masks from this image together and perform one mask D2H
        # transfer. Conversion allocates new storage, so mul_ never modifies
        # the prediction tensor. NumPy slices below share disjoint CPU regions.
        masks_cpu = masks.to(torch.uint8).mul_(255).cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        names = getattr(result, "names", {}) or {}
        instances = []
        for index, mask in enumerate(masks_cpu):
            if int(np.count_nonzero(mask)) < self.min_component_area:
                continue
            class_id = int(classes[index]) if index < len(classes) else -1
            class_name = (
                names.get(class_id, names.get(str(class_id), class_id))
                if isinstance(names, dict)
                else class_id
            )
            instances.append(
                {
                    "instance_id": len(instances) + 1,
                    "class_id": class_id,
                    "class_name": str(class_name),
                    "confidence": float(confidences[index]) if index < len(confidences) else 1.0,
                    "mask": mask,
                    "rectangle": None,
                    "rectangle_ratio": 1.0,
                    "is_ok": True,
                }
            )
        return instances

    def _measure_2d_wire_instances(self, instances, roi_boxes, crop_box):
        """Compute bending for both ``wire`` and ``thin`` masks.

        The check is deliberately performed per YOLO instance so the result
        can be returned immediately with the single-piece response.  The 3D
        board path still builds connected wire regions for height alignment.
        """
        output = []
        offset = np.asarray(crop_box[:2], dtype=np.int32)
        for instance in instances:
            class_name = str(instance.get("class_name", "")).casefold()
            if class_name not in {"wire", "thin"}:
                continue
            mask = np.asarray(instance.get("mask"), dtype=np.uint8)
            cleaned, mask_offset = _geometry.clean_mask_crop(
                mask, min_area=self.min_component_area, kernel_size=5
            )
            bending = _geometry.measure_wire_bending(
                cleaned, roi_boxes, min_ratio=self.rectangle_ratio_threshold,
                mask_offset=mask_offset, output_offset=offset,
            )
            instance["bending_2d"] = bending
            instance["is_ok"] = bool(bending["is_ok"])
            instance["rectangle_ratio"] = float(bending["mask_ratio"])
            instance["rectangle"] = bending["rectangle"] or None
            output.append(
                {
                    "region_id": len(output) + 1,
                    "instance_id": int(instance["instance_id"]),
                    "class_name": str(instance["class_name"]),
                    "mask_2d": _mask_geometry(
                        cleaned, offset=offset + mask_offset, epsilon=2.0
                    ),
                    "bending_2d": bending,
                }
            )
        return output


    @staticmethod
    def _prepare_chip_regions(chips):
        return JointGeometry._prepare_chip_regions(chips, fill=_filled_external_region)


    @staticmethod
    def _matching_chip(bond, chips):
        return JointGeometry._matching_chip(bond, chips, fill=_filled_external_region)


    def _public_2d_instances(self, instances, crop_box):
        snapshots = [dict(item) for item in instances]
        return self._instance_tasks.map(
            lambda item: self._public_2d_instance(item, crop_box), snapshots,
            timing_stage="public_2d_geometry_work_ms")

    def _public_2d_instance(
        self, instance: Dict[str, Any], crop_box: Sequence[int] | None = None
    ) -> Dict[str, Any]:
        """Serialize a cached pre-alignment YOLO instance for the 2D reply."""
        return _instance_geometry.public_instance(instance, crop_box or self.crop_roi, geometry=_mask_geometry)

    @staticmethod
    def _failure(error: str, message: str) -> Dict[str, Any]:
        return {
            "ok": False,
            "cmd": "detect_board",
            "stage": "board",
            "error": error,
            "message": message,
            "results_board": {"registration_ok": False, "pieces": []},
        }

    def release(self) -> None:
        self._instance_tasks.close()
        self._pitch_tasks.close()
        for resource in self._gpu_worker_resources[1:]:
            resource["inspector"].release()
        self._gpu_worker_resources = []
        self._gpu_worker_slots = queue.Queue()
        self.model = None
        self.model_path = None
        self.model_paths = {}
        self.config_path = None
        self.config = {}
        self.circle_crop_config = None
        self.silver_config = {}
        self.chip_alignment_config = {}
        self.chip_anomaly_config = {"configured": False}
        self.dynamic_affine_config = {}
        self.pitch_config = {}
        if self.chip_anomaly_inspector is not None:
            self.chip_anomaly_inspector.release()
        self.chip_anomaly_inspector = None
        with self._piece_cache_lock:
            self._piece_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000.0


def _mask_geometry(
    mask: np.ndarray,
    offset: Sequence[int],
    epsilon: float,
) -> Dict[str, Any]:
    return _instance_geometry.mask_geometry(mask, offset, epsilon)


def _filled_external_region(mask: np.ndarray) -> np.ndarray:
    from algorithm.joint_geometry import _filled_external_region as fill
    return fill(mask)


def _dict(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _box(value: Any, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must contain four integers")
    box = tuple(int(item) for item in value)
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} is invalid: {box}")
    return box

