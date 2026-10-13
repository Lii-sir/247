"""Streaming board flow: circle-relative 2D pieces, 3D split and alignment."""
from __future__ import annotations

import gc
import io
import threading
import time
from contextlib import redirect_stdout
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
    fit_chip_plane,
    fit_local_platform_plane,
    fit_platform_plane,
    map_platform_rois,
    measure_mask_height,
    split_thin_mask,
)
from algorithm.registration import (
    RegistrationConfig,
    align_crop_mask_to_depth,
    platform_mask_2d,
    platform_mask_3d,
    refine_affine_registration,
    register_platform_masks,
)
from algorithm.perspective_registration import estimate_perspective
from transport.shared_memory import read_shared_memory


_segment = import_module("algorithm.3D.segment")
_geometry = import_module("algorithm.2D.defect_geometry")
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


class DetectBoard:
    def __init__(self, log):
        self.log = log
        self.model = None
        self.model_path = None
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
        self.perspective_registration_config: Dict[str, Any] = {}
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

    def handle_init(self, req: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        self.config_path = str(req.get("config_path") or "") or None
        raw_model_paths = req.get("model_path")
        model_paths = raw_model_paths if isinstance(raw_model_paths, dict) else {}
        self.model_path = str(model_paths.get("YOLO") or "") or None
        try:
            self._load_config(str(req.get("config_text") or ""))
            if not self.model_path:
                raise ValueError("model_path.YOLO is empty")
            self._validate_gpu_requirement()
            load_start = time.perf_counter()
            with redirect_stdout(io.StringIO()):
                from ultralytics import YOLO

                self.model = YOLO(self.model_path)
            model_load_ms = self.elapsed_ms(load_start)
            self.chip_anomaly_inspector = _chip_anomaly.ChipAnomalyInspector(
                self.chip_anomaly_config,
                {str(name): str(path) for name, path in model_paths.items()},
            )
            anomaly_load_timings = self.chip_anomaly_inspector.load()
        except Exception as exc:
            self.model = None
            self.chip_anomaly_inspector = None
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
            "model_load_timings": {"YOLO": model_load_ms, **anomaly_load_timings},
        }

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
        self.perspective_registration_config = dict(
            feature_parameters(config, "perspective_registration_3d") or {}
        )

    def _validate_gpu_requirement(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("board detection requires an available CUDA GPU")
        if str(self.yolo_parameters.get("device", "")).lower() == "cpu":
            raise ValueError("yolo.parameters.device must select a CUDA device")

    # ------------------------------------------------------------------
    # Streaming board protocol
    # ------------------------------------------------------------------
    def handle_detect_2d_piece(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Process and cache exactly one 2D image.

        The cache key is the explicit ``board_id`` and ``piece_id`` pair.  A
        repeated piece replaces the previous result atomically, which lets a
        caller retry a failed camera frame without leaving a half-written
        entry in the board cache.
        """
        start = time.perf_counter()
        try:
            board_id = self._stream_board_id(req)
        except ValueError as exc:
            return self._stream_failure("invalid_board_id", str(exc), board_id="")
        try:
            try:
                piece_id = self._stream_piece_id(req)
            except ValueError as exc:
                return self._stream_failure(
                    "invalid_piece_id", str(exc), board_id=board_id,
                    piece_id=self._optional_piece_id(req)
                )
            read_start = time.perf_counter()
            image = self._read_2d(req)
            circle_start = time.perf_counter()
            crop, crop_box, circle, roi_boxes = self._stream_crop(image)
            crop = crop.copy()
            circle_crop_ms = self.elapsed_ms(circle_start)
            registration_mask = platform_mask_2d(image, self.registration_config)
            read_decode_ms = self.elapsed_ms(read_start)

            if self.model is None:
                raise RuntimeError("board/stream 2D model is not initialized")
            inference_start = time.perf_counter()
            self._synchronize_model()
            predictions = self._predict_batch([crop])
            self._synchronize_model()
            inference_ms = self.elapsed_ms(inference_start)
            if len(predictions) != 1:
                raise RuntimeError(
                    f"YOLO returned {len(predictions)} results for one 2D image"
                )

            postprocess_start = time.perf_counter()
            instances = self._extract_instances(predictions[0], crop.shape[:2])
            wire_measurements = self._measure_2d_wire_instances(
                instances, roi_boxes, crop_box
            )
            wire_roi_checks = _wire_roi_presence.check_wire_roi_presence(
                instances, roi_boxes, crop_box[:2]
            )
            chip_diagnostics = _chip_alignment.measure_chip_alignment(
                instances, self.chip_alignment_config, crop_box[:2]
            )
            if self.chip_anomaly_inspector is None:
                raise RuntimeError("chip anomaly inspector is not initialized")
            chip_anomaly_diagnostics = self.chip_anomaly_inspector.inspect(
                image, instances, crop_box
            )
            diagnostics = _silver_inspection.inspect_masks(
                instances,
                crop.shape[:2],
                self.silver_config,
                crop_box=crop_box,
                circle=circle,
                image=image,
                base_dir=self._silver_config_base_dir(),
            )
            diagnostics = _silver_inspection.offset_geometry(diagnostics, crop_box[:2])
            postprocess_ms = self.elapsed_ms(postprocess_start)

            entry = {
                "piece_id": piece_id,
                # Retain only grayscale pixels needed for perspective fitting;
                # dropping the original BGR image avoids ~32 MB per piece.
                "gray_image": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
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
                "silver_overflow_2d": diagnostics["silver_overflow"],
                "silver_continuity_2d": diagnostics["disconnection"],
                "received_at": time.time(),
            }
            with self._piece_cache_lock:
                board_cache = self._piece_cache.setdefault(board_id, {})
                replaced = piece_id in board_cache
                board_cache[piece_id] = entry
                cache_state = self._cache_state_locked(board_id)

            total_ms = self.elapsed_ms(start)
            public_instances = [self._public_2d_instance(item, crop_box) for item in instances]
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
            return {
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
        except (ValueError, OSError, RuntimeError) as exc:
            return self._stream_failure(
                "2d_piece_failed",
                str(exc),
                board_id=board_id,
                piece_id=self._optional_piece_id(req),
            )

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

            segment_start = time.perf_counter()
            depth_pieces = segment_depth_image(depth, self.segmentation_config)
            segmentation_ms = self.elapsed_ms(segment_start)
            if len(depth_pieces) != self.expected_piece_count:
                raise PieceCountMismatchError(
                    self.expected_piece_count, len(depth_pieces)
                )

            align_start = time.perf_counter()
            results = []
            perspective_config = self.perspective_registration_config
            for index, piece in enumerate(depth_pieces):
                piece_id = index + 1
                entry = entries[piece_id]
                y1, y2 = int(piece["y1"]), int(piece["y2"])
                local_depth = depth[y1:y2]
                registration = register_platform_masks(
                    entry["registration_mask"],
                    platform_mask_3d(local_depth, self.registration_config),
                    self.registration_config,
                )
                registration = refine_affine_registration(
                    entry["gray_image"],
                    local_depth,
                    registration,
                    self.dynamic_affine_config,
                )
                registration = estimate_perspective(
                    entry["gray_image"], local_depth, registration, perspective_config
                )
                aligned_masks = [
                    self._align_instance(
                        item, local_depth.shape, registration, entry.get("crop_box", self.crop_roi)
                    )
                    for item in entry["instances"]
                ]
                measurement = self._measure_piece(
                    local_depth,
                    aligned_masks,
                    registration,
                    entry.get("roi_boxes", self.wire_rois),
                    crop_box=entry.get("crop_box", self.crop_roi),
                    circle=entry.get("circle"),
                )
                diagnostics_ok = all(
                    check.get("is_ok") is True
                    for check in (
                        entry.get("silver_overflow_2d", {}),
                        entry.get("silver_continuity_2d", {}),
                        entry.get("chip_anomaly_2d", {}),
                    )
                    if isinstance(check, dict) and check.get("status") not in {None, "disabled"}
                )
                diagnostics_present = any(
                    check.get("status") not in {None, "disabled"}
                    for check in (
                        entry.get("silver_overflow_2d", {}),
                        entry.get("silver_continuity_2d", {}),
                        entry.get("chip_anomaly_2d", {}),
                    )
                    if isinstance(check, dict)
                )
                results.append(
                    {
                        "piece_id": piece_id,
                        "image_2d_size": entry["image_shape"],
                        "crop_2d": list(entry.get("crop_box", self.crop_roi)),
                        "wire_rois_2d": [list(roi) for roi in entry.get("roi_boxes", self.wire_rois)],
                        "wire_roi_checks": entry.get("wire_roi_checks", []),
                        "depth_start_row": y1,
                        "depth_end_row": y2,
                        "registration": registration,
                        "is_ok": bool(
                            measurement["is_ok"]
                            and all(item.get("is_ok") is True for item in entry.get("wire_roi_checks", []))
                            and (not diagnostics_present or diagnostics_ok)
                        ),
                        "reference_planes": measurement["reference_planes"],
                        "wire_measurements": measurement["wire_measurements"],
                        "solder_measurements": measurement["solder_measurements"],
                        "chip_anomaly_2d": entry.get("chip_anomaly_2d", {"status": "disabled", "is_ok": None}),
                        "masks": [self._public_instance(item) for item in aligned_masks],
                    }
                )
            alignment_ms = self.elapsed_ms(align_start)
            total_ms = self.elapsed_ms(start)
            return {
                "ok": True,
                "cmd": "detect_3d_board",
                "stage": "3d",
                "board_id": board_id,
                "batch_id": board_id,
                "results_board": {
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
                    "segmentation_3d_ms": segmentation_ms,
                    "perspective_registration_and_mask_alignment_ms": alignment_ms,
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
                    gray_images[index], local_depth, registration, self.dynamic_affine_config
                )
                registration = estimate_perspective(
                    gray_images[index], local_depth, registration,
                    self.perspective_registration_config,
                )
                crop_box, circle, roi_boxes = crop_metadata[index]
                instances = self._extract_instances(prediction, crop.shape[:2])
                aligned_masks = [
                    self._align_instance(instance, local_depth.shape, registration, crop_box)
                    for instance in instances
                ]
                measurement = self._measure_piece(
                    local_depth, aligned_masks, registration,
                    wire_rois=roi_boxes, crop_box=crop_box, circle=circle,
                )
                results.append(
                    {
                        "piece_id": int(descriptor["piece_id"]),
                        "image_2d_size": image_shapes[index],
                        "crop_2d": list(crop_box),
                        "wire_rois_2d": [list(roi) for roi in roi_boxes],
                        "depth_start_row": y1,
                        "depth_end_row": y2,
                        "registration": registration,
                        "is_ok": measurement["is_ok"],
                        "reference_planes": measurement["reference_planes"],
                        "wire_measurements": measurement["wire_measurements"],
                        "solder_measurements": measurement["solder_measurements"],
                        "masks": [self._public_instance(item) for item in aligned_masks],
                    }
                )
            alignment_ms = self.elapsed_ms(align_start)
            elapsed_ms = self.elapsed_ms(start)
            return {
                "ok": True,
                "cmd": "detect_board",
                "stage": "board",
                "batch_id": str(req.get("batch_id") or ""),
                "results_board": {
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
        data = read_shared_memory(name, size)
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
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to decode 2D image for piece {descriptor.get('piece_id')}")
        return image

    @staticmethod
    def _read_3d(descriptor: Dict[str, Any]) -> np.ndarray:
        size = int(descriptor.get("data_size", 0))
        name = str(descriptor.get("shm_name", ""))
        if not name or size <= 0:
            raise ValueError("image_3d requires shm_name and data_size")
        data = read_shared_memory(name, size)
        return decode_depth_png(data, str(descriptor.get("data_format", "png")))

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
        circle = _circle_roi.find_circle(
            image,
            fixed_radius=config["circle_radius"],
            radius_tolerance=config["radius_tolerance_pixels"] / config["circle_radius"],
            search_roi=config["circle_search_roi"],
        )
        crop, box = _circle_roi.crop_from_config(
            image, circle, {"roi_offsets": config["detection_roi_offsets"]}
        )
        roi_boxes = []
        for offsets in config["rectangle_roi_offsets"]:
            _, full = _circle_roi.crop_from_config(image, circle, {"roi_offsets": offsets})
            roi_boxes.append(
                (full[0] - box[0], full[1] - box[1], full[2] - box[0], full[3] - box[1])
            )
        return crop, box, circle, roi_boxes

    def _predict_batch(self, crops: Sequence[np.ndarray]):
        parameters = {**self.yolo_parameters, "verbose": False, "retina_masks": True}
        with redirect_stdout(io.StringIO()):
            return list(self.model.predict(source=list(crops), **parameters))

    def _synchronize_model(self) -> None:
        device = getattr(self.model, "device", None)
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
        classes = boxes.cls.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        names = getattr(result, "names", {}) or {}
        instances = []
        for index, mask_tensor in enumerate(masks):
            mask = (mask_tensor.to(torch.uint8) * 255).cpu().numpy()
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
            if not np.any(mask):
                instance["bending_2d"] = {
                    "is_valid": False, "is_ok": False, "mask_ratio": 0.0, "rectangle": []
                }
                continue
            cleaned = _geometry.clean_mask(
                mask, min_area=self.min_component_area, kernel_size=5
            )
            rectangle = _geometry.constrained_rectangle(
                cleaned, roi_boxes, min_ratio=self.rectangle_ratio_threshold
            ) if np.any(cleaned) else None
            if rectangle is None:
                bending = {
                    "is_valid": False,
                    "is_ok": False,
                    "mask_ratio": 0.0,
                    "rectangle": [],
                }
            else:
                corners, ratio, is_ok = rectangle
                bending = {
                    "is_valid": True,
                    "is_ok": bool(is_ok),
                    "mask_ratio": float(ratio),
                    "rectangle": (corners + offset).astype(int).tolist(),
                }
            instance["bending_2d"] = bending
            instance["is_ok"] = bool(is_ok) if rectangle is not None else False
            instance["rectangle_ratio"] = float(bending["mask_ratio"])
            instance["rectangle"] = bending["rectangle"] or None
            output.append(
                {
                    "region_id": len(output) + 1,
                    "instance_id": int(instance["instance_id"]),
                    "class_name": str(instance["class_name"]),
                    "mask_2d": _mask_geometry(cleaned, offset=crop_box[:2], epsilon=2.0),
                    "bending_2d": bending,
                }
            )
        return output

    def _align_instance(
        self,
        instance: Dict[str, Any],
        depth_shape: tuple[int, int],
        registration: Dict[str, Any],
        crop_box: Sequence[int] | None = None,
    ) -> Dict[str, Any]:
        crop_mask = instance["mask"]
        crop_box = tuple(crop_box or self.crop_roi)
        aligned = align_crop_mask_to_depth(
            crop_mask, crop_box, depth_shape, registration
        )
        return {
            **instance,
            "mask_2d": _mask_geometry(crop_mask, offset=crop_box[:2], epsilon=2.0),
            "mask_3d_local": _mask_geometry(aligned, offset=(0, 0), epsilon=1.0),
            "_mask_2d_array": crop_mask,
            "_mask_3d_array": aligned,
        }

    def _mask_in_wire_rois(
        self, mask: np.ndarray, wire_rois: Sequence[Sequence[int]] | None = None
    ) -> np.ndarray:
        output = np.zeros_like(mask, dtype=np.uint8)
        height, width = output.shape
        for raw_roi in self.wire_rois if wire_rois is None else wire_rois:
            x1, y1, x2, y2 = (int(value) for value in raw_roi)
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 > x1 and y2 > y1:
                output[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return output

    def _measure_piece(
        self,
        depth: np.ndarray,
        instances: list[Dict[str, Any]],
        registration: Dict[str, Any],
        wire_rois: Sequence[Sequence[int]] | None = None,
        crop_box: Sequence[int] | None = None,
        circle: Sequence[float] | None = None,
    ) -> Dict[str, Any]:
        """Measure all 3D heights for one aligned piece.

        ``thin`` is both a wire and two solder joints: its first and last
        principal-axis 2 percent are measured as solder and the middle 96
        percent is measured as a wire.  Every reference plane is fitted after
        removing the complete aligned semantic mask union.  The chip itself
        is kept in the input region while fitting its own plane, but all
        other semantic masks are excluded.
        """
        crop_box = tuple(crop_box or self.crop_roi)
        semantic_exclusion = self._semantic_exclusion_mask(instances, depth.shape)
        platform_plane = None
        platform_diagnostics: Dict[str, Any] = {
            "status": "not_configured", "error": "platform_roi_offsets is empty",
            "boxes_2d": [], "polygons_3d_local": [],
            "sampling_mask_3d_local": _mask_geometry(
                np.zeros(depth.shape, dtype=np.uint8), offset=(0, 0), epsilon=0.0
            ),
        }
        try:
            if self.height_config.platform_roi_offsets:
                if circle is None:
                    raise ValueError("detected 2D circle is unavailable")
                if registration.get("is_valid") is False:
                    raise ValueError("2D/3D registration is invalid")
                if registration.get("perspective_valid") is not True:
                    raise ValueError("piece perspective registration is invalid")
                roi_mask, boxes, polygons = map_platform_rois(
                    circle, self.height_config.platform_roi_offsets,
                    registration, depth.shape,
                )
                sampling_mask = (roi_mask > 0) & (semantic_exclusion == 0)
                sampling_mask &= ~np.isin(depth, self.height_config.invalid_values)
                platform_diagnostics.update(
                    boxes_2d=boxes, polygons_3d_local=polygons,
                    roi_area_px=int(np.count_nonzero(roi_mask)),
                    candidate_points=int(np.count_nonzero(sampling_mask)),
                    sampling_mask_3d_local=_mask_geometry(
                        sampling_mask, offset=(0, 0), epsilon=0.0
                    ),
                )
                platform_plane = fit_platform_plane(
                    depth, self.height_config, roi_mask, exclude_mask=semantic_exclusion
                )
                platform_diagnostics.update(status="ok", error=None)
        except ValueError as exc:
            platform_diagnostics.update(status="invalid", error=str(exc))
        chips = [
            item for item in instances if item["class_name"].lower() == "chip"
        ]
        chip_planes: Dict[int, Dict[str, Any] | None] = {}
        for chip in chips:
            # Build this exclusion union without the current chip instance.
            # Resetting its pixels in an already-unioned mask would also
            # restore a bond/wire that overlaps the chip mask and bias the
            # chip reference plane upward.
            chip_exclusion = self._semantic_exclusion_mask(
                instances,
                depth.shape,
                skip_instance_id=chip["instance_id"],
            )
            try:
                chip_planes[chip["instance_id"]] = fit_chip_plane(
                    depth,
                    chip["_mask_3d_array"],
                    self.height_config,
                    exclude_mask=chip_exclusion,
                )
            except ValueError:
                chip_planes[chip["instance_id"]] = None

        wire_measurements = self._measure_wire_regions(
            depth,
            instances,
            registration,
            platform_plane,
            wire_rois,
            crop_box=crop_box,
        )
        wire_by_instance = {
            int(item["instance_id"]): item for item in wire_measurements
        }

        solder_measurements: list[Dict[str, Any]] = []
        for instance in instances:
            class_name = instance["class_name"].lower()
            if class_name == "thin":
                middle_2d, solder_2d = split_thin_mask(
                    instance["_mask_2d_array"],
                    self.height_config.thin_solder_end_ratio,
                )
                instance["_thin_wire_mask_2d"] = middle_2d
                instance["_thin_solder_masks_2d"] = solder_2d
                for endpoint, solder_mask_2d in zip(("start", "end"), solder_2d):
                    solder_mask_3d = align_crop_mask_to_depth(
                        solder_mask_2d,
                        crop_box,
                        depth.shape,
                        registration,
                    )
                    # The registration scale is not necessarily one.  Expand
                    # after mapping so this parameter is expressed in actual
                    # depth-image pixels, which is the search range needed to
                    # tolerate a very thin 2D endpoint missing the 3D ridge.
                    solder_mask_3d = dilate_mask(
                        solder_mask_3d,
                        self.height_config.thin_solder_search_dilation_px,
                    )
                    solder_measurements.append(
                        self._measure_solder(
                            depth,
                            solder_mask_2d,
                            solder_mask_3d,
                            instance,
                            endpoint,
                            chips,
                            chip_planes,
                            platform_plane,
                            semantic_exclusion,
                            crop_box,
                        )
                    )
                wire_result = wire_by_instance.get(int(instance["instance_id"]))
                instance["height_3d"] = (
                    wire_result["height_3d"] if wire_result is not None else None
                )
            elif class_name == "bond":
                solder_measurements.append(
                    self._measure_solder(
                        depth,
                        instance["_mask_2d_array"],
                        instance["_mask_3d_array"],
                        instance,
                        "bond",
                        chips,
                        chip_planes,
                        platform_plane,
                        semantic_exclusion,
                        crop_box,
                    )
                )
                solder_result = solder_measurements[-1]
                instance["height_3d"] = solder_result["height_3d"]
                instance["bond_location"] = solder_result["location"]
                instance["reference_instance_id"] = solder_result["reference_instance_id"]
                instance["chip_inside"] = solder_result["chip_inside"]
            elif class_name == "wire":
                wire_result = wire_by_instance.get(int(instance["instance_id"]))
                instance["height_3d"] = (
                    wire_result["height_3d"] if wire_result is not None else None
                )
            elif class_name == "chip":
                instance["height_3d"] = None
                instance["plane_3d"] = chip_planes.get(instance["instance_id"])
            else:
                instance["height_3d"] = None

        wire_ok = [
            bool(item["bending_2d"].get("is_ok"))
            and bool(item["height_3d"].get("is_ok"))
            for item in wire_measurements
        ]
        solder_ok = [bool(item["height_3d"].get("is_ok")) for item in solder_measurements]
        return {
            "is_ok": bool(wire_measurements or solder_measurements)
            and all(wire_ok + solder_ok),
            "wire_measurements": wire_measurements,
            "solder_measurements": solder_measurements,
            "reference_planes": {
                "wire_platform": platform_plane,
                "platform_roi": platform_diagnostics,
                "chips": [
                    {
                        "instance_id": chip["instance_id"],
                        "plane": chip_planes.get(chip["instance_id"]),
                    }
                    for chip in chips
                ],
            },
        }

    def _semantic_exclusion_mask(
        self,
        instances: Sequence[Dict[str, Any]],
        shape: tuple[int, int],
        skip_instance_id: int | None = None,
    ) -> np.ndarray:
        """Return a dilated union of all aligned YOLO masks."""
        union = np.zeros(shape, dtype=np.uint8)
        for item in instances:
            if skip_instance_id is not None and item.get("instance_id") == skip_instance_id:
                continue
            mask = np.asarray(item.get("_mask_3d_array"), dtype=np.uint8)
            if mask.shape == shape:
                union = np.maximum(union, mask)
        return dilate_mask(union, self.height_config.semantic_exclusion_dilation_px)

    def _measure_solder(
        self,
        depth: np.ndarray,
        mask_2d: np.ndarray,
        mask_3d: np.ndarray,
        instance: Dict[str, Any],
        endpoint: str,
        chips: list[Dict[str, Any]],
        chip_planes: Dict[int, Dict[str, Any] | None],
        platform_plane: Dict[str, Any] | None,
        semantic_exclusion: np.ndarray,
        crop_box: Sequence[int],
    ) -> Dict[str, Any]:
        chip, inside = self._matching_chip(
            {"_mask_2d_array": mask_2d}, chips
        )
        on_chip = chip is not None and inside
        if on_chip:
            plane = chip_planes.get(chip["instance_id"])
            reference = "chip"
            location = "chip"
            reference_instance_id = chip["instance_id"]
            max_height = self.height_config.max_solder_chip_height_mm
        else:
            try:
                plane = fit_local_platform_plane(
                    depth, mask_3d, semantic_exclusion, self.height_config
                )
            except ValueError:
                plane = None
            reference = "local_wire_platform" if plane is not None else "wire_platform"
            location = "wire_platform"
            reference_instance_id = None
            if plane is None:
                plane = platform_plane
            max_height = self.height_config.max_solder_platform_height_mm
        height = measure_mask_height(
            depth,
            mask_3d,
            plane,
            self.height_config,
            reference=reference,
            max_height_mm=max_height,
        )
        return {
            "measurement_id": int(instance.get("instance_id", 0)) * 3 + {
                "bond": 0,
                "start": 1,
                "end": 2,
            }.get(endpoint, 0),
            "source_instance_id": int(instance.get("instance_id", 0)),
            "class_name": str(instance.get("class_name", "")),
            "endpoint": endpoint,
            "location": location,
            "reference_instance_id": reference_instance_id,
            "chip_inside": bool(inside),
            "reference_plane": plane,
            "mask_2d": _mask_geometry(mask_2d, offset=crop_box[:2], epsilon=1.0),
            "mask_3d_local": _mask_geometry(mask_3d, offset=(0, 0), epsilon=1.0),
            "height_3d": height,
        }

    def _measure_wire_regions(
        self,
        depth: np.ndarray,
        instances: list[Dict[str, Any]],
        registration: Dict[str, Any],
        platform_plane: Dict[str, Any] | None,
        wire_rois: Sequence[Sequence[int]] | None = None,
        crop_box: Sequence[int] | None = None,
    ) -> list[Dict[str, Any]]:
        crop_box = tuple(crop_box or self.crop_roi)
        wires = [item for item in instances if item["class_name"].lower() in {"wire", "thin"}]
        if not wires:
            return []
        wire_rois = self.wire_rois if wire_rois is None else wire_rois
        output = []
        offset = np.asarray(crop_box[:2], dtype=np.int32)
        for wire in wires:
            class_name = wire["class_name"].lower()
            source_mask = np.asarray(wire["_mask_2d_array"], dtype=np.uint8)
            if class_name == "thin":
                source_mask, _ = split_thin_mask(
                    source_mask, self.height_config.thin_solder_end_ratio
                )
            if not np.any(self._mask_in_wire_rois(source_mask, wire_rois)):
                # Wires outside the configured 2D arc-height ROI are not measured.
                continue
            cleaned = _geometry.clean_mask(
                source_mask,
                min_area=self.min_component_area,
                kernel_size=5,
            )
            cleaned = self._mask_in_wire_rois(cleaned, wire_rois)
            if not np.any(cleaned):
                height = measure_mask_height(
                    depth, np.zeros_like(depth, dtype=np.uint8), platform_plane,
                    self.height_config, reference="wire_platform",
                    max_height_mm=self.height_config.max_wire_platform_height_mm,
                )
                height["is_ok"] = False
                output.append({
                    "region_id": len(output) + 1,
                    "instance_id": int(wire.get("instance_id", 0)),
                    "class_name": str(wire.get("class_name", "")),
                    "mask_2d": _mask_geometry(cleaned, offset=crop_box[:2], epsilon=1.0),
                    "mask_3d_local": _mask_geometry(np.zeros_like(depth, dtype=np.uint8), offset=(0, 0), epsilon=1.0),
                    "bending_2d": wire.get("bending_2d", {"is_valid": False, "is_ok": False}),
                    "height_3d": height,
                })
                continue
            # Bending is calculated once on the original YOLO instance during
            # the 2D stage.  Keep that result here; only the central thin mask
            # is changed for the independent 3D wire-height measurement.
            bending = wire.get("bending_2d")
            if not isinstance(bending, dict):
                rectangle = _geometry.constrained_rectangle(
                    cleaned,
                    [tuple(int(value) for value in roi) for roi in wire_rois],
                    min_ratio=self.rectangle_ratio_threshold,
                )
                if rectangle is None:
                    bending = {
                        "is_valid": False,
                        "is_ok": False,
                        "mask_ratio": 0.0,
                        "rectangle": [],
                    }
                else:
                    corners, ratio, is_ok = rectangle
                    bending = {
                        "is_valid": True,
                        "is_ok": bool(is_ok),
                        "mask_ratio": float(ratio),
                        "rectangle": (corners + offset).astype(int).tolist(),
                    }
            aligned = align_crop_mask_to_depth(
                cleaned.astype(np.uint8) * 255,
                crop_box,
                depth.shape,
                registration,
            )
            height = measure_mask_height(
                depth,
                aligned,
                platform_plane,
                self.height_config,
                reference="wire_platform",
                max_height_mm=self.height_config.max_wire_platform_height_mm,
            )
            output.append(
                {
                    "region_id": len(output) + 1,
                    "instance_id": int(wire.get("instance_id", 0)),
                    "class_name": str(wire.get("class_name", "")),
                    "mask_2d": _mask_geometry(
                        cleaned, offset=crop_box[:2], epsilon=2.0
                    ),
                    "mask_3d_local": _mask_geometry(
                        aligned, offset=(0, 0), epsilon=1.0
                    ),
                    "bending_2d": bending,
                    "height_3d": height,
                }
            )
        return output

    @staticmethod
    def _matching_chip(
        bond: Dict[str, Any],
        chips: list[Dict[str, Any]],
    ) -> tuple[Dict[str, Any] | None, bool]:
        bond_mask = bond["_mask_2d_array"] > 0
        ys, xs = np.nonzero(bond_mask)
        if len(xs) == 0 or not chips:
            return None, False
        center_x = int(round(float(xs.mean())))
        center_y = int(round(float(ys.mean())))
        best_chip = None
        best_area = None
        for chip in chips:
            chip_region = _filled_external_region(chip["_mask_2d_array"])
            if chip_region[center_y, center_x]:
                area = int(np.count_nonzero(chip_region))
                if best_area is None or area < best_area:
                    best_chip, best_area = chip, area
        return best_chip, best_chip is not None

    @staticmethod
    def _public_instance(instance: Dict[str, Any]) -> Dict[str, Any]:
        return {
            ("chip_rotation_2d" if key == "chip_alignment" else key): value
            for key, value in instance.items()
            if key != "mask" and not key.startswith("_")
        }

    def _public_2d_instance(
        self, instance: Dict[str, Any], crop_box: Sequence[int] | None = None
    ) -> Dict[str, Any]:
        """Serialize a cached pre-alignment YOLO instance for the 2D reply."""
        mask = instance.get("mask")
        offset = tuple(crop_box or self.crop_roi)[:2]
        return {
            "instance_id": int(instance.get("instance_id", 0)),
            "class_id": int(instance.get("class_id", -1)),
            "class_name": str(instance.get("class_name", "")),
            "confidence": float(instance.get("confidence", 0.0)),
            "mask_2d": _mask_geometry(
                mask,
                offset=offset,
                epsilon=1.0,
            ) if isinstance(mask, np.ndarray) else _mask_geometry(
                np.zeros((1, 1), dtype=np.uint8), (0, 0), 1.0
            ),
            "rectangle": instance.get("rectangle"),
            "rectangle_ratio": float(instance.get("rectangle_ratio", 0.0)),
            "bending_2d": instance.get("bending_2d"),
            "chip_rotation_2d": instance.get("chip_alignment"),
            "chip_anomaly_2d": instance.get("chip_anomaly_2d"),
            "is_ok": bool(instance.get("is_ok", True)),
        }

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
        self.model = None
        self.model_path = None
        self.config_path = None
        self.config = {}
        self.circle_crop_config = None
        self.silver_config = {}
        self.chip_alignment_config = {}
        self.chip_anomaly_config = {"configured": False}
        self.dynamic_affine_config = {}
        self.perspective_registration_config = {}
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
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    area = int(np.count_nonzero(binary))
    if area == 0:
        return {"area_px": 0, "bounding_box": [], "contours": []}
    ys, xs = np.nonzero(binary)
    ox, oy = int(offset[0]), int(offset[1])
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    serialized = []
    for contour in contours:
        approximate = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
        if len(approximate) >= 3:
            serialized.append(
                [[int(point[0]) + ox, int(point[1]) + oy] for point in approximate]
            )
    return {
        "area_px": area,
        "bounding_box": [
            int(xs.min()) + ox,
            int(ys.min()) + oy,
            int(xs.max()) + 1 + ox,
            int(ys.max()) + 1 + oy,
        ],
        "contours": serialized,
    }


def _filled_external_region(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    if contours:
        points = np.concatenate(contours, axis=0)
        cv2.fillConvexPoly(filled, cv2.convexHull(points), 255)
    return filled > 0


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

