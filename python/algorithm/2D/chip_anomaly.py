"""Rectify YOLO chip masks and inspect the resulting crops with EfficientAD."""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image

from algorithm.efficientad_models import EfficientAdModel


MULTISCALE_SCORE_METHOD = "masked_multiscale_normalized_topk_max"
SINGLE_SCALE_SCORE_METHOD = "masked_local_average_topk_mean"
TOP_SCORE_METHOD = "masked_pixel_max"
def validate_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return the compact, validated production configuration."""
    if config is None:
        return {
            "configured": False,
            "chip_class": "chip",
            "device": "auto",
            "aspect_ratio_models": [],
            "padding_ratio": 0.02,
            "max_rotation_deg": 5.0,
        }
    raw = dict(config)
    selection = raw.get("aspect_ratio_models")
    if not isinstance(selection, list) or not selection:
        raise ValueError(
            "chip_anomaly_2d.aspect_ratio_models must be a non-empty list of "
            "{range: [min, max], model: name} objects"
        )
    ranges = []
    for index, item in enumerate(selection):
        if not isinstance(item, dict):
            raise ValueError(
                f"chip_anomaly_2d.aspect_ratio_models[{index}] must be an object"
            )
        raw_range = item.get("range")
        # ``Detect2D`` and ``DetectBoard`` retain the validated form so it can
        # be exposed in diagnostics.  Keep validation idempotent when that
        # normalized object is handed to ``ChipAnomalyInspector``.
        if raw_range is None and "min" in item and "max" in item:
            raw_range = [item["min"], item["max"]]
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise ValueError(
                f"chip_anomaly_2d.aspect_ratio_models[{index}].range must be [min, max]"
            )
        minimum, maximum = (float(value) for value in raw_range)
        model = str(item.get("model", "")).strip()
        if not model:
            raise ValueError(f"chip anomaly model is empty at index {index}")
        if not (math.isfinite(minimum) and math.isfinite(maximum)):
            raise ValueError(f"chip anomaly range is not finite at index {index}")
        if minimum <= 0 or maximum < minimum:
            raise ValueError(f"chip anomaly range is invalid at index {index}")
        ranges.append({"min": minimum, "max": maximum, "model": model})
    ranges.sort(key=lambda item: (item["min"], item["max"]))
    for previous, current in zip(ranges, ranges[1:]):
        if current["min"] <= previous["max"]:
            raise ValueError(
                "chip_anomaly_2d aspect ratio ranges cannot overlap: "
                f"[{previous['min']}-{previous['max']}] and "
                f"[{current['min']}-{current['max']}]"
            )
    padding = float(raw.get("padding_ratio", 0.02))
    max_rotation = float(raw.get("max_rotation_deg", 5.0))
    if not math.isfinite(padding) or padding < 0 or padding > 0.5:
        raise ValueError("chip_anomaly_2d.padding_ratio must be in [0, 0.5]")
    if not math.isfinite(max_rotation) or not 0 <= max_rotation <= 45:
        raise ValueError("chip_anomaly_2d.max_rotation_deg must be in [0, 45]")
    return {
        "configured": True,
        "chip_class": str(raw.get("chip_class", "chip")),
        "device": str(raw.get("device", "auto")),
        "padding_ratio": padding,
        "max_rotation_deg": max_rotation,
        "aspect_ratio_models": ranges,
    }


def rectified_chip_crop(
    image: np.ndarray,
    points: np.ndarray,
    padding: float,
    max_rotation_deg: float,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Perspective-rectify one chip using the same geometry as test_chip_crops.py."""
    rect = cv2.minAreaRect(points.astype(np.float32))
    (cx, cy), (rw, rh), angle = rect
    if rw <= 1 or rh <= 1:
        raise ValueError("degenerate chip rectangle")
    corners = cv2.boxPoints(rect).astype(np.float32)
    raw_angle = float((angle + 45.0) % 90.0 - 45.0)
    correction = float(np.clip(raw_angle, -max_rotation_deg, max_rotation_deg))
    ratio = float(max(rw, rh) / max(1.0, min(rw, rh)))
    scale = 1.0 + max(0.0, padding) * 2.0

    order = np.argsort(corners[:, 1] * 10000 + corners[:, 0])
    top = corners[order[:2]][np.argsort(corners[order[:2], 0])]
    bottom = corners[order[2:]][np.argsort(corners[order[2:], 0])[::-1]]
    source = np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)
    center = np.array([cx, cy], dtype=np.float32)
    source = (source - center) * scale + center
    edge_width = float(np.linalg.norm(source[1] - source[0]))
    edge_height = float(np.linalg.norm(source[3] - source[0]))
    out_w, out_h = max(2, round(edge_width)), max(2, round(edge_height))
    destination = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    crop = cv2.warpPerspective(
        image, matrix, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE
    )
    return crop, ratio, correction, corners, matrix


def _chip_contour_points(mask: np.ndarray, offset: Sequence[int]) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if len(contour) >= 3]
    if not contours:
        raise ValueError("chip mask has fewer than three contour points")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    contour += np.asarray(offset[:2], dtype=np.float32)
    return contour


def _resolve_device(requested: str) -> torch.device:
    text = requested.strip().lower()
    if text == "auto":
        text = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif text.isdigit():
        text = f"cuda:{text}"
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("chip anomaly detection requires CUDA, but CUDA is unavailable")
    return device


def _preprocess(image_bgr: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = Image.fromarray(rgb).resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).unsqueeze(0)
    return tensor.to(device)


def _pooled_topk(
    anomaly_map: torch.Tensor, kernel: int, topk_ratio: float
) -> tuple[float, torch.Tensor]:
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("EfficientAD pool kernels must be positive odd integers")
    padding = kernel // 2
    pooled_sum = torch_functional.avg_pool2d(
        anomaly_map, kernel_size=kernel, stride=1, padding=padding,
        divisor_override=1,
    )
    valid = torch.ones_like(anomaly_map)
    pooled_count = torch_functional.avg_pool2d(
        valid, kernel_size=kernel, stride=1, padding=padding,
        divisor_override=1,
    )
    pooled = pooled_sum / pooled_count.clamp_min(1)
    values = pooled.reshape(-1)
    count = max(1, int(math.ceil(values.numel() * topk_ratio)))
    score = float(torch.topk(values, count, largest=True, sorted=False).values.mean())
    return score, pooled


def _scores_and_response(
    anomaly_map: torch.Tensor,
    calibration: Dict[str, Any],
    model_config: Dict[str, Any],
) -> tuple[Dict[str, float], np.ndarray, np.ndarray, list[int]]:
    method = calibration.get("score_method") or {}
    method_name = method.get("name")
    threshold = float(calibration["threshold"])
    topk_ratio = float(method.get("topk_ratio", model_config.get("score_topk_ratio", 0.001)))
    raw_map = anomaly_map[0, 0].detach().cpu().numpy().astype(np.float32, copy=True)

    if method_name == MULTISCALE_SCORE_METHOD:
        normalized_maps = []
        active_maps = []
        active_scales = []
        scores: Dict[str, float] = {"score_max": float(anomaly_map.max())}
        for value in method.get("pool_kernels", []):
            kernel = int(value)
            raw_score, pooled = _pooled_topk(anomaly_map, kernel, topk_ratio)
            reference = method["normalization"][str(kernel)]
            median = float(reference["median"])
            denominator = float(reference["denominator"])
            if denominator <= 0 or not math.isfinite(denominator):
                raise ValueError(f"invalid EfficientAD normalization for kernel {kernel}")
            normalized_score = max(0.0, (raw_score - median) / denominator)
            normalized_map = np.maximum(
                0.0,
                (pooled[0, 0].detach().cpu().numpy() - median) / denominator,
            ).astype(np.float32)
            scores[f"score_kernel_{kernel}"] = raw_score
            scores[f"score_normalized_kernel_{kernel}"] = normalized_score
            normalized_maps.append(normalized_map)
            if normalized_score > threshold:
                active_maps.append(normalized_map)
                active_scales.append(kernel)
        if not normalized_maps:
            raise ValueError("EfficientAD multiscale checkpoint has no pool kernels")
        score = max(scores[key] for key in scores if key.startswith("score_normalized_kernel_"))
        response = (
            np.maximum.reduce(active_maps) if active_maps
            else np.maximum.reduce(normalized_maps)
        )
        scores.update(score=score, score_topk=score)
        binary = response > max(0.0, threshold) if score > threshold else np.zeros_like(response, bool)
        return scores, response, binary, active_scales

    if method_name == SINGLE_SCALE_SCORE_METHOD:
        kernel = int(method.get("pool_kernel", model_config.get("score_pool_kernel", 21)))
        score, pooled = _pooled_topk(anomaly_map, kernel, topk_ratio)
        response = pooled[0, 0].detach().cpu().numpy().astype(np.float32)
        scores = {"score": score, "score_topk": score, "score_max": float(anomaly_map.max())}
        binary = response > max(0.0, threshold) if score > threshold else np.zeros_like(response, bool)
        return scores, response, binary, [kernel] if score > threshold else []

    if method_name not in (None, TOP_SCORE_METHOD):
        raise ValueError(f"unsupported EfficientAD score method: {method_name}")
    score = float(anomaly_map.max())
    binary = raw_map > max(0.0, threshold) if score > threshold else np.zeros_like(raw_map, bool)
    return {"score": score, "score_topk": score, "score_max": score}, raw_map, binary, [1] if score > threshold else []


def _component_boxes(binary: np.ndarray, params: Dict[str, Any]) -> list[list[int]]:
    if not np.any(binary):
        return []
    kernel_size = int(params.get("morph_kernel", 3))
    if kernel_size < 1 or kernel_size % 2 == 0:
        kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = binary.astype(np.uint8) * 255
    opened = int(params.get("open_iterations", 1))
    closed = int(params.get("close_iterations", 1))
    if opened:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=opened)
    if closed:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=closed)
    count, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    min_area = max(1, int(math.ceil(binary.size * float(params.get("min_area_ratio", 0.00002)))))
    boxes = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < min_area:
            continue
        x, y = int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP])
        w, h = int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT])
        boxes.append([x, y, x + w, y + h])
    return boxes


def _map_box_to_image(
    box: Sequence[int],
    response_shape: Sequence[int],
    crop_shape: Sequence[int],
    inverse: np.ndarray,
) -> tuple[list[int], list[list[int]]]:
    rh, rw = int(response_shape[0]), int(response_shape[1])
    ch, cw = int(crop_shape[0]), int(crop_shape[1])
    x1, y1, x2, y2 = (int(value) for value in box)
    crop_box = [
        int(round(x1 * cw / rw)), int(round(y1 * ch / rh)),
        int(round(x2 * cw / rw)), int(round(y2 * ch / rh)),
    ]
    points = np.array([[
        [crop_box[0], crop_box[1]], [crop_box[2], crop_box[1]],
        [crop_box[2], crop_box[3]], [crop_box[0], crop_box[3]],
    ]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(points, inverse)[0]
    return crop_box, np.rint(mapped).astype(int).tolist()


class _LoadedPredictor:
    def __init__(self, name: str, checkpoint: str, device: torch.device):
        self.name = name
        self.checkpoint = str(Path(checkpoint).resolve())
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 1 or not saved.get("calibration"):
            raise ValueError(f"{name} is not a calibrated EfficientAD model.pt")
        config = dict(saved["config"])
        config.setdefault("model_size", "small")
        config.setdefault("backbone", f"pdn_{config['model_size']}")
        config.setdefault("resnet_architecture_version", 1)
        config.setdefault("resnet_feature_mode", "native")
        config.setdefault("resnet_teacher_output_activation", "relu")
        image_size = config.get("image_size")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
            raise ValueError(f"{name} checkpoint has invalid image_size")
        self.config = config
        self.calibration = dict(saved["calibration"])
        self.category = str(saved.get("manifest", {}).get("category", name))
        self.image_size = image_size
        self.device = device
        self.model = EfficientAdModel(
            model_size=config["model_size"],
            backbone=config["backbone"],
            hard_loss_mode=config.get("hard_loss_mode", "global"),
            resnet_architecture_version=int(config.get("resnet_architecture_version", 1)),
            resnet_feature_mode=config.get("resnet_feature_mode"),
            resnet_teacher_output_activation=config.get("resnet_teacher_output_activation"),
        )
        self.model.load_state_dict(saved["model_state"], strict=True)
        self.model.to(device).eval()
        self._lock = threading.RLock()

    def predict(self, crop: np.ndarray) -> tuple[Dict[str, Any], Dict[str, Any]]:
        tensor = _preprocess(crop, self.image_size, self.device)
        with self._lock, torch.inference_mode():
            prediction = self.model(tensor)
            scores, response, binary, active_scales = _scores_and_response(
                prediction.anomaly_map, self.calibration, self.config
            )
        threshold = float(self.calibration["threshold"])
        score = float(scores["score"])
        boxes = _component_boxes(binary, self.config.get("localization", {}))
        return ({
            "model": self.name,
            "category": self.category,
            "score": score,
            "score_topk": float(scores["score_topk"]),
            "score_max": float(scores["score_max"]),
            "threshold": threshold,
            "prediction": "NG" if score > threshold else "OK",
            "decision_rule": "score > threshold",
            "is_ok": bool(score <= threshold),
            "active_scales": active_scales,
        }, {
            "response": response,
            "binary": binary,
            "boxes": boxes,
            "display_max": float(self.calibration.get("display_max", max(1.0, threshold))),
        })


class ChipAnomalyInspector:
    """Own both aspect-ratio-selected models and inspect all YOLO chip instances."""

    def __init__(self, config: Dict[str, Any] | None, model_paths: Dict[str, str]):
        self.config = validate_config(config)
        self.model_paths = dict(model_paths)
        self.device = _resolve_device(self.config["device"])
        self.predictors: Dict[str, _LoadedPredictor] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("configured"))

    def load(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
        timings = {}
        names = dict.fromkeys(rule["model"] for rule in self.config["aspect_ratio_models"])
        for name in names:
            path = self.model_paths.get(name)
            if not path:
                raise ValueError(f"model_path.{name} is required by chip_anomaly_2d")
            started = time.perf_counter()
            self.predictors[name] = _LoadedPredictor(name, path, self.device)
            timings[name] = (time.perf_counter() - started) * 1000.0
        return timings

    def _model_rule(self, aspect_ratio: float) -> Dict[str, Any]:
        for rule in self.config["aspect_ratio_models"]:
            if rule["min"] <= aspect_ratio <= rule["max"]:
                return rule
        configured = ", ".join(
            f"[{rule['min']}-{rule['max']}]={rule['model']}"
            for rule in self.config["aspect_ratio_models"]
        )
        raise ValueError(
            f"chip aspect ratio {aspect_ratio:.4f} does not match a configured "
            f"EfficientAD model range ({configured})"
        )

    def _model_name(self, aspect_ratio: float) -> str:
        return str(self._model_rule(aspect_ratio)["model"])

    def inspect(
        self,
        image: np.ndarray,
        instances: list[Dict[str, Any]],
        crop_box: Sequence[int],
        *,
        include_visualization: bool = False,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "status": "disabled", "is_ok": None, "chips": []}
        chips = [
            item for item in instances
            if str(item.get("class_name", "")).casefold() == self.config["chip_class"].casefold()
            and isinstance(item.get("mask"), np.ndarray)
        ]
        if not chips:
            return {"enabled": True, "status": "no_chip", "is_ok": False, "chips": []}

        results = []
        visualizations = []
        for instance in chips:
            points = _chip_contour_points(instance["mask"], crop_box[:2])
            crop, ratio, correction, corners, matrix = rectified_chip_crop(
                image, points, self.config["padding_ratio"], self.config["max_rotation_deg"]
            )
            model_rule = self._model_rule(ratio)
            model_name = str(model_rule["model"])
            predictor = self.predictors.get(model_name)
            if predictor is None:
                raise RuntimeError(f"EfficientAD model {model_name!r} is not loaded")
            prediction, artifacts = predictor.predict(crop)
            inverse = np.linalg.inv(matrix)
            anomaly_boxes = []
            for box in artifacts["boxes"]:
                crop_xyxy, polygon = _map_box_to_image(
                    box, artifacts["response"].shape, crop.shape[:2], inverse
                )
                anomaly_boxes.append({
                    "crop_xyxy": crop_xyxy,
                    "image_polygon": polygon,
                })
            result = {
                "instance_id": int(instance.get("instance_id", 0)),
                "class_name": str(instance.get("class_name", "")),
                "model": model_name,
                "aspect_ratio": float(ratio),
                "model_aspect_ratio_range": [
                    float(model_rule["min"]), float(model_rule["max"])
                ],
                "rectification_angle_deg": float(correction),
                "rectangle": np.rint(corners).astype(int).tolist(),
                "rectified_size": [int(crop.shape[1]), int(crop.shape[0])],
                "model_input_size": [predictor.image_size, predictor.image_size],
                "anomaly_boxes": anomaly_boxes,
                **prediction,
            }
            instance["chip_anomaly_2d"] = result
            instance["is_ok"] = bool(instance.get("is_ok", True) and result["is_ok"])
            results.append(result)
            if include_visualization:
                visualizations.append({
                    "instance_id": result["instance_id"],
                    "crop": crop,
                    "response": artifacts["response"],
                    "binary": artifacts["binary"],
                    "boxes": artifacts["boxes"],
                    "display_max": artifacts["display_max"],
                    "result": result,
                })
        output = {
            "enabled": True,
            "status": "ok" if all(item["is_ok"] for item in results) else "anomaly",
            "is_ok": all(item["is_ok"] for item in results),
            "ratio_definition": "long_side / short_side",
            "aspect_ratio_models": [
                {
                    "range": [float(rule["min"]), float(rule["max"])],
                    "model": str(rule["model"]),
                }
                for rule in self.config["aspect_ratio_models"]
            ],
            "chips": results,
        }
        if include_visualization:
            output["_visualizations"] = visualizations
        return output

    def release(self) -> None:
        self.predictors.clear()
