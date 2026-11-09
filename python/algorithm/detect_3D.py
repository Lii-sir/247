import gc
import time
from importlib import import_module
from typing import Any, Dict

from algorithm.tilt_correction import correct_smartray_depth, production_pitch_parameters, depth_coordinate_space
from algorithm.config import feature_parameters, load_runtime_config
from algorithm.instance_tasks import InstanceTasks
from transport.shared_memory import read_shared_memory
from transport.depth_input import validate_depth_input, depth_from_uint16


# 具体算法存放在 3D 目录；数字目录名不能用于常规 from ... import 语法。
_segment = import_module("algorithm.3D.segment")
_wire_height = import_module("algorithm.3D.wire_height")
PieceCountMismatchError = _segment.PieceCountMismatchError
SegmentationConfig = _segment.SegmentationConfig
segment_depth_image = _segment.segment_depth_image
measure_depth = _wire_height.measure
configure_measurement = _wire_height.configure


class Detect3D:
    def __init__(self):
        self.model_path = None
        self.config_path = None
        self.segmentation_config = SegmentationConfig()
        self.pitch_config = {}
        self._pitch_tasks = InstanceTasks(thread_name_prefix="pitch")

    def handle_init(self, req: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        cmd = str(req.get("cmd", "init_algorithm"))

        self.model_path = req.get("model_path")
        self.config_path = req.get("config_path")

        config_text = req.get("config_text")
        self._pitch_tasks.close()
        self.pitch_config = {}
        try:
            configure_measurement(None)
            self.segmentation_config = SegmentationConfig()
            self.pitch_config = {}
        except Exception:
            pass

        if config_text:
            try:
                config = load_runtime_config(str(config_text), self.config_path)
                values = feature_parameters(config, "segmentation_3d")
                self.segmentation_config = SegmentationConfig.from_dict(values)
                measurement_values = feature_parameters(config, "wire_height_3d")
                configure_measurement(measurement_values)
                self.pitch_config = production_pitch_parameters(config,
                    float(measurement_values.get("gray_offset", 32768.)),
                    float(measurement_values.get("vertical_resolution_mm", .00105)))
                self._pitch_tasks.configure(self.pitch_config.get("projection_workers", 1))
            except Exception as exc:
                self._pitch_tasks.close()
                return {
                    "ok": False,
                    "cmd": cmd,
                    "stage": "3d",
                    "error": f"invalid 3D config: {exc}",
                }

        return {
            "ok": True,
            "cmd": cmd,
            "stage": "3d",
            "message": "3D mask centerline P95 measurement ready",
            "timings": {
                "total_init_ms": self.elapsed_ms(start_time),
            },
            "model_load_timings": {},
        }

    def handle_detect(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.detect(req.get("shm_name"), req.get("data_size"), req.get("data_format", "uint16"),
                           req.get("image_width"), req.get("image_height"), req.get("image_stride", 0))

    def detect(
        self,
        shm_name: str,
        data_size: int,
        data_format: str,
        image_width: int | None = None,
        image_height: int | None = None,
        image_stride: int = 0,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        try:
            shm_name, data_size, image_width, image_height = validate_depth_input(dict(
                shm_name=shm_name, data_size=data_size, data_format=data_format,
                image_width=image_width, image_height=image_height, image_stride=image_stride))
        except ValueError as exc:
            return {
                "ok": False,
                "cmd": "detect",
                "stage": "3d",
                "error": f"invalid 3D shared memory data: {exc}",
            }

        try:
            read_start = time.perf_counter()
            data = read_shared_memory(shm_name, data_size)
            shared_memory_read_ms = self.elapsed_ms(read_start)
        except Exception as exc:
            return {
                "ok": False,
                "cmd": "detect",
                "stage": "3d",
                "error": f"3D shared memory read failed: {exc}",
            }

        try:
            algorithm_start = time.perf_counter()
            view_start = time.perf_counter()
            depth = depth_from_uint16(data, image_width, image_height)
            view_ms = self.elapsed_ms(view_start)
            pitch_start = time.perf_counter()
            depth, pitch_correction = correct_smartray_depth(depth, self.pitch_config, tasks=self._pitch_tasks)
            pitch_ms = self.elapsed_ms(pitch_start)
            segment_start = time.perf_counter()
            pieces = segment_depth_image(depth, self.segmentation_config)
            segmentation_ms = self.elapsed_ms(segment_start)
            measurement_start = time.perf_counter()
            pieces = measure_depth(depth, pieces)
            results_3d = {"pitch_correction": pitch_correction,
                          "depth_coordinate_space": depth_coordinate_space(pitch_correction),
                          "is_ok": all(piece["is_ok"] for piece in pieces),
                          "pieces": pieces}
            measurement_ms = self.elapsed_ms(measurement_start)
            algorithm_detect_ms = self.elapsed_ms(algorithm_start)

        except PieceCountMismatchError as exc:
            return {
                "ok": False,
                "cmd": "detect",
                "stage": "3d",
                "error": "piece_count_mismatch",
                "message": str(exc),
                "results_3d": {"is_ok": False, "pieces": []},
            }
        except (ValueError, OSError) as exc:
            return {
                "ok": False,
                "cmd": "detect",
                "stage": "3d",
                "error": f"3D segmentation failed: {exc}",
                "results_3d": {"is_ok": False, "pieces": []},
            }

        elapsed_ms = self.elapsed_ms(start_time)

        return {
            "ok": True,
            "cmd": "detect",
            "stage": "3d",
            "results_3d": results_3d,
            "elapsed_ms": elapsed_ms,
            "timings": {
                "shared_memory_read_ms": shared_memory_read_ms,
                "algorithm_detect_ms": algorithm_detect_ms,
                "depth_array_view_ms": view_ms,
                "smartray_pitch_ms": pitch_ms,
                "segmentation_ms": segmentation_ms,
                "measurement_ms": measurement_ms,
                "received_bytes": float(len(data)),
                "total_detect_ms": elapsed_ms,
            },
            "message": (
                f"segmented {len(pieces)} pieces from {depth.shape[1]}x{depth.shape[0]} "
                f"raw depth, dtype={depth.dtype}"
            ),
        }

    def release(self):
        self._pitch_tasks.close()
        self.model_path = None
        self.config_path = None
        self.segmentation_config = SegmentationConfig()
        self.pitch_config = {}
        gc.collect()

    @staticmethod
    def elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000.0
