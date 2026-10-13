import gc
import time
from importlib import import_module
from typing import Any, Dict

from algorithm.config import feature_parameters, load_runtime_config
from transport.shared_memory import read_shared_memory


# 具体算法存放在 3D 目录；数字目录名不能用于常规 from ... import 语法。
_segment = import_module("algorithm.3D.segment")
_wire_height = import_module("algorithm.3D.wire_height")
DepthImageDecodeError = _segment.DepthImageDecodeError
PieceCountMismatchError = _segment.PieceCountMismatchError
SegmentationConfig = _segment.SegmentationConfig
decode_depth_png = _segment.decode_depth_png
segment_depth_image = _segment.segment_depth_image
measure_depth = _wire_height.measure
configure_measurement = _wire_height.configure


class Detect3D:
    def __init__(self):
        self.model_path = None
        self.config_path = None
        self.segmentation_config = SegmentationConfig()

    def handle_init(self, req: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        cmd = str(req.get("cmd", "init_algorithm"))

        self.model_path = req.get("model_path")
        self.config_path = req.get("config_path")

        config_text = req.get("config_text")
        try:
            configure_measurement(None)
            self.segmentation_config = SegmentationConfig()
        except Exception:
            pass

        if config_text:
            try:
                config = load_runtime_config(str(config_text), self.config_path)
                values = feature_parameters(config, "segmentation_3d")
                self.segmentation_config = SegmentationConfig.from_dict(values)
                measurement_values = feature_parameters(config, "wire_height_3d")
                configure_measurement(measurement_values)
            except Exception as exc:
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
        shm_name = str(req.get("shm_name", ""))
        data_size = int(req.get("data_size", req.get("image_size", 0)))
        data_format = str(req.get("data_format", "uint16"))
        return self.detect(shm_name, data_size, data_format)

    def detect(
        self,
        shm_name: str,
        data_size: int,
        data_format: str,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if not shm_name or data_size <= 0:
            return {
                "ok": False,
                "cmd": "detect",
                "stage": "3d",
                "error": "invalid 3D shared memory data",
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
            decode_start = time.perf_counter()
            depth = decode_depth_png(data, data_format)
            decode_ms = self.elapsed_ms(decode_start)
            segment_start = time.perf_counter()
            pieces = segment_depth_image(depth, self.segmentation_config)
            segmentation_ms = self.elapsed_ms(segment_start)
            measurement_start = time.perf_counter()
            pieces = measure_depth(depth, pieces)
            results_3d = {"is_ok": all(piece["is_ok"] for piece in pieces),
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
        except (DepthImageDecodeError, ValueError, OSError) as exc:
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
                "decode_ms": decode_ms,
                "segmentation_ms": segmentation_ms,
                "measurement_ms": measurement_ms,
                "received_bytes": float(len(data)),
                "total_detect_ms": elapsed_ms,
            },
            "message": (
                f"segmented {len(pieces)} pieces from {depth.shape[1]}x{depth.shape[0]} "
                f"depth PNG, dtype={depth.dtype}"
            ),
        }

    def release(self):
        self.model_path = None
        self.config_path = None
        self.segmentation_config = SegmentationConfig()
        gc.collect()

    @staticmethod
    def elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000.0
