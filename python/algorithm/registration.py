"""Fixed-scale registration from a full 2D image to one local 3D depth piece."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Sequence

import cv2
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class RegistrationConfig:
    scale: float = 0.306
    threshold_2d: int = 50
    roi_3d: tuple[int, int, int, int] = (120, 50, 480, 720)
    search_2d_scaled: tuple[int, int, int, int] = (250, 0, 950, 900)
    morphology_kernel_size: int = 5
    minimum_score: float = 0.7

    @classmethod
    def from_dict(cls, values: Dict[str, Any] | None) -> "RegistrationConfig":
        values = values or {}
        roi_3d = _four_ints(values.get("roi_3d", cls.roi_3d), "registration.roi_3d")
        search = _four_ints(
            values.get("search_2d_scaled", cls.search_2d_scaled),
            "registration.search_2d_scaled",
        )
        config = cls(
            scale=float(values.get("scale", cls.scale)),
            threshold_2d=int(values.get("threshold_2d", cls.threshold_2d)),
            roi_3d=roi_3d,
            search_2d_scaled=search,
            morphology_kernel_size=int(
                values.get("morphology_kernel_size", cls.morphology_kernel_size)
            ),
            minimum_score=float(values.get("minimum_score", cls.minimum_score)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.scale <= 0:
            raise ValueError("registration.scale must be greater than zero")
        if not 0 <= self.threshold_2d <= 255:
            raise ValueError("registration.threshold_2d must be in [0, 255]")
        if self.morphology_kernel_size <= 0:
            raise ValueError("registration.morphology_kernel_size must be positive")
        if not -1 <= self.minimum_score <= 1:
            raise ValueError("registration.minimum_score must be in [-1, 1]")
        _validate_box(self.roi_3d, "registration.roi_3d")
        _validate_box(self.search_2d_scaled, "registration.search_2d_scaled")


def platform_mask_2d(image: np.ndarray, config: RegistrationConfig) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("2D registration image must be a BGR image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray,
        None,
        fx=config.scale,
        fy=config.scale,
        interpolation=cv2.INTER_AREA,
    )
    mask = (small > config.threshold_2d).astype(np.uint8) * 255
    kernel = np.ones(
        (config.morphology_kernel_size, config.morphology_kernel_size),
        dtype=np.uint8,
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def platform_mask_3d(depth_piece: np.ndarray, config: RegistrationConfig) -> np.ndarray:
    if depth_piece.ndim != 2 or depth_piece.size == 0:
        raise ValueError("3D registration piece must be a non-empty depth image")
    mask = (depth_piece > 0).astype(np.uint8) * 255
    kernel = np.ones(
        (config.morphology_kernel_size, config.morphology_kernel_size),
        dtype=np.uint8,
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def register_platform_masks(
    mask_2d_scaled: np.ndarray,
    mask_3d: np.ndarray,
    config: RegistrationConfig,
) -> Dict[str, Any]:
    x0, y0, x1, y1 = config.roi_3d
    sx0, sy0, sx1, sy1 = config.search_2d_scaled
    template_bottom = min(y1, mask_3d.shape[0] - 20)
    if x1 > mask_3d.shape[1] or template_bottom <= y0:
        raise ValueError(
            f"registration.roi_3d does not fit depth piece shape {mask_3d.shape}"
        )
    if sx1 > mask_2d_scaled.shape[1] or sy1 > mask_2d_scaled.shape[0]:
        raise ValueError(
            "registration.search_2d_scaled does not fit the scaled 2D image: "
            f"search={config.search_2d_scaled}, shape={mask_2d_scaled.shape}"
        )

    template = mask_3d[y0:template_bottom, x0:x1]
    search = mask_2d_scaled[sy0:sy1, sx0:sx1]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        raise ValueError("2D registration search area is smaller than the 3D template")
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (match_x, match_y) = cv2.minMaxLoc(response)
    tx = float(x0 - (sx0 + match_x))
    ty = float(y0 - (sy0 + match_y))
    # Keep the historical 2x3 matrix for callers that consume tx/ty, but also
    # publish a homogeneous matrix.  The latter is intentionally used by the
    # streaming board path with warpPerspective so the registration contract
    # can be upgraded to a true projective transform without changing the
    # cached 2D result format.
    homography_full = np.asarray(
        [
            [config.scale, 0.0, tx + config.scale * 0.0],
            [0.0, config.scale, ty + config.scale * 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return {
        "scale": config.scale,
        "tx": tx,
        "ty": ty,
        "score": float(score),
        "is_valid": bool(score >= config.minimum_score),
        "matrix_2d_to_3d": [
            [config.scale, 0.0, tx],
            [0.0, config.scale, ty],
        ],
        "homography_2d_to_3d": homography_full.tolist(),
    }


def refine_affine_registration(
    image_2d_gray: np.ndarray,
    depth_piece: np.ndarray,
    coarse_registration: Dict[str, Any],
    values: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Dynamically fit one six parameter affine transform for one piece.

    The optimizer follows the supplied ``register_affine_desktop_test.py``
    calibration logic, but starts from the current coarse match and keeps all
    parameters in memory.  It compares several spatial zones so a large flat
    substrate cannot dominate the result.
    """
    values = values or {}
    result = dict(coarse_registration)
    result["affine_valid"] = False
    result["affine_score_before"] = None
    result["affine_score_after"] = None
    result["affine_zone_losses"] = []
    result["affine_scoring_region_ids"] = []
    result.update(affine_function_evaluations=0, affine_score_calls=0,
                  affine_score_total_ms=0., affine_score_mean_ms=0.,
                  affine_optimizer_success=None, affine_optimizer_status=None,
                  affine_stop_reason="not_started")

    if image_2d_gray.ndim != 2 or depth_piece.ndim != 2:
        result["affine_error"] = "affine registration requires grayscale 2D and 2D depth arrays"
        return result
    if image_2d_gray.size == 0 or depth_piece.size == 0:
        result["affine_error"] = "affine registration received an empty image"
        return result

    preview_scale = float(values.get("preview_scale", 0.5))
    if not 0 < preview_scale <= 1:
        result["affine_error"] = "dynamic_affine.preview_scale must be in (0, 1]"
        return result
    threshold = int(values.get("threshold_2d", 50))
    # Keep this at one by default to match the reference calibration. A
    # caller can opt into a small closing kernel for noisier cameras.
    kernel_size = int(values.get("morphology_kernel_size", 1))
    if kernel_size <= 0:
        result["affine_error"] = "dynamic_affine.morphology_kernel_size must be positive"
        return result

    source_mask = (image_2d_gray > threshold).astype(np.uint8)
    source_mask = cv2.resize(
        source_mask,
        None,
        fx=preview_scale,
        fy=preview_scale,
        interpolation=cv2.INTER_NEAREST,
    )
    target_mask = (depth_piece > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    source_mask = cv2.morphologyEx(source_mask, cv2.MORPH_CLOSE, kernel)
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)
    target_height, target_width = target_mask.shape

    regions = _affine_regions(values.get("rois"), target_width, target_height,
                              values.get("selected_regions"))
    rois = [box for _, box in regions]
    result["affine_scoring_region_ids"] = [region_id for region_id, _ in regions]
    if not rois:
        result["affine_error"] = f"no valid affine scoring zones for depth shape {target_mask.shape}"
        return result
    target_rois = tuple(
        target_mask[y0:y1, x0:x1] for x0, y0, x1, y1 in rois
    )
    # Native warpAffine is the production default. ROI sampling preserves
    # pixels but its coordinate arrays can cost more than the pixels saved.
    score_mode = values.get("score_mode", "full")
    if score_mode not in {"roi", "full"}:
        result["affine_error"] = "dynamic_affine.score_mode must be roi or full"
        return result
    # remap uses signed-short coordinates. Preserve the former path for
    # exceptional image sizes beyond that API's supported range.
    if max(*source_mask.shape, target_height, target_width) >= 32767:
        score_mode = "full"
    if (cv2.__version__.split('.')[0] != '5' or not cv2.useOptimized()
            or '*AVX2' not in cv2.getCPUFeaturesLine().split()):
        # Only enable the optimized kernel on the validated CPU backend.
        score_mode = "full"
    grids = [_affine_roi_grid(box) for box in rois] if score_mode == "roi" else None
    result["affine_score_mode"] = score_mode
    score_calls = 0
    score_seconds = 0.

    initial_matrix = _registration_matrix(coarse_registration)
    p0 = _affine_parameters_from_matrix(initial_matrix, preview_scale)
    translation_delta = values.get("translation_delta", [35.0, 25.0])
    if not isinstance(translation_delta, (list, tuple)) or len(translation_delta) != 2:
        translation_delta = [35.0, 25.0]
    scale_delta = float(values.get("scale_delta", 0.02))
    shear_limit = float(values.get("shear_limit", 0.02))
    if scale_delta <= 0 or shear_limit <= 0:
        result["affine_error"] = "dynamic affine bounds must be positive"
        return result
    tx_delta, ty_delta = float(translation_delta[0]), float(translation_delta[1])
    if tx_delta <= 0 or ty_delta <= 0:
        result["affine_error"] = "dynamic_affine.translation_delta must be positive"
        return result

    lower = np.asarray(
        [
            max(0.05, p0[0] - scale_delta),
            max(0.05, p0[1] - scale_delta),
            p0[2] - shear_limit,
            p0[3] - shear_limit,
            p0[4] - tx_delta,
            p0[5] - ty_delta,
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            p0[0] + scale_delta,
            p0[1] + scale_delta,
            p0[2] + shear_limit,
            p0[3] + shear_limit,
            p0[4] + tx_delta,
            p0[5] + ty_delta,
        ],
        dtype=np.float64,
    )
    penalty_weight = float(values.get("penalty_weight", 0.005))

    def score(parameters: np.ndarray, details: bool = False):
        nonlocal score_calls, score_seconds
        started = time.perf_counter()
        scaled = _scaled_affine(parameters)
        if score_mode == "roi":
            patches = _warp_affine_rois(source_mask, scaled, rois, grids, target_width)
        else:
            warped = cv2.warpAffine(source_mask, scaled, (target_width, target_height),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            patches = (warped[y0:y1, x0:x1] for x0, y0, x1, y1 in rois)
        losses = [
            float(np.count_nonzero(patch != target_roi) / target_roi.size)
            for patch, target_roi in zip(patches, target_rois)
        ]
        # Penalize shear more than translation, matching the reference
        # experiment while allowing real lens/perspective anisotropy.
        penalty = penalty_weight * (
            (parameters[2] / max(shear_limit, 1e-6)) ** 2
            + (parameters[3] / max(shear_limit, 1e-6)) ** 2
        )
        value = float(np.mean(losses) + penalty)
        score_calls += 1
        score_seconds += time.perf_counter() - started
        if details:
            return value, losses
        return value

    before, _ = score(p0, details=True)
    optimizer_started = time.perf_counter()
    try:
        optimized = minimize(
            score,
            p0,
            method="Powell",
            bounds=list(zip(lower, upper)),
            options={
                "maxiter": int(values.get("maxiter", 35)),
                "xtol": float(values.get("xtol", 1e-4)),
                "ftol": float(values.get("ftol", 1e-4)),
            },
        )
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        result["affine_error"] = str(exc)
        result["affine_score_before"] = before
        result.update(affine_stop_reason=str(exc), affine_optimizer_success=False,
            affine_function_evaluations=max(0, score_calls - 1), affine_score_calls=score_calls,
            affine_score_total_ms=score_seconds * 1000,
            affine_score_mean_ms=score_seconds * 1000 / max(1, score_calls))
        return result

    result["affine_optimizer_ms"] = (time.perf_counter() - optimizer_started) * 1000
    result["affine_function_evaluations"] = int(getattr(optimized, "nfev", score_calls - 1))
    result["affine_optimizer_success"] = bool(optimized.success) if hasattr(optimized, "success") else None
    result["affine_optimizer_status"] = int(optimized.status) if hasattr(optimized, "status") else None
    result["affine_stop_reason"] = str(getattr(optimized, "message", "optimizer did not report a stop reason"))
    after, losses = score(optimized.x, details=True)
    result.update(affine_score_calls=score_calls, affine_score_total_ms=score_seconds * 1000,
                  affine_score_mean_ms=score_seconds * 1000 / max(1, score_calls))
    matrix = _full_affine(optimized.x, preview_scale)
    result["affine_valid"] = bool(
        np.all(np.isfinite(matrix))
        and float(after) <= float(values.get("max_score", 0.25))
    )
    result["affine_score_before"] = float(before)
    result["affine_score_after"] = float(after)
    result["affine_zone_losses"] = [float(value) for value in losses]
    result["affine_iterations"] = int(getattr(optimized, "nit", 0) or 0)
    result["affine_matrix_2d_to_3d"] = matrix.tolist()
    result["homography_2d_to_3d"] = np.vstack(
        [matrix, np.asarray([0.0, 0.0, 1.0])]
    ).tolist()
    result["matrix_2d_to_3d"] = matrix.tolist()
    result["scale"] = float(np.mean([matrix[0, 0], matrix[1, 1]]))
    result["tx"] = float(matrix[0, 2])
    result["ty"] = float(matrix[1, 2])
    # The coarse template score remains available as ``coarse_score``. The
    # final validity is decided by the dynamically fitted affine quality.
    result["is_valid"] = bool(result["affine_valid"])
    return result


def _affine_roi_grid(box):
    x0, y0, x1, y1 = box
    return np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32)


def _warp_affine_rois(source, matrix, rois, grids, canvas_width):
    """Sample only scoring pixels using GLOBAL destination coordinates.

    OpenCV 5's uint8 nearest kernel uses float32 row arithmetic followed by
    SIMD FMA in X; its scalar tail has left-associated arithmetic instead.
    Use global X/Y and the ORIGINAL canvas width to preserve that boundary.
    See OpenCV modules/imgproc/src/warp_kernels.simd.hpp (nearest 8UC1).
    """
    inverse = cv2.invertAffineTransform(matrix).astype(np.float32)
    features = cv2.getCPUFeaturesLine().split()
    fma = cv2.useOptimized() and '*AVX2' in features
    unroll = 16 if fma else 8
    vector_end = canvas_width // unroll * unroll
    for (x0, y0, x1, y1), (x, y) in zip(rois, grids):
        coordinates = np.empty((y1-y0, x1-x0, 2), np.int16)
        for axis in range(2):
            row = y * inverse[axis, 1] + inverse[axis, 2]
            if fma:
                # float64 represents a product of two float32 values exactly;
                # one cast reproduces fused multiply-add rounding.
                positions = (x.astype(np.float64)[None, :] * float(inverse[axis, 0])
                             + row.astype(np.float64)[:, None]).astype(np.float32)
            else:
                positions = x[None, :] * inverse[axis, 0] + row[:, None]
            tail = max(0, vector_end - x0)
            if tail < len(x):
                positions[:, tail:] = ((x[None, tail:] * inverse[axis, 0]
                    + y[:, None] * inverse[axis, 1]) + inverse[axis, 2])
            coordinates[:, :, axis] = np.clip(np.rint(positions), -32768, 32767)
        yield cv2.remap(source, coordinates, None, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _registration_matrix(registration: Dict[str, Any]) -> np.ndarray:
    raw = registration.get("homography_2d_to_3d")
    if raw is not None:
        matrix = np.asarray(raw, dtype=np.float64)
        if matrix.shape == (3, 3) and abs(float(matrix[2, 2])) > 1e-12:
            return (matrix / matrix[2, 2])[:2]
    return np.asarray(
        [
            [float(registration.get("scale", 1.0)), 0.0, float(registration.get("tx", 0.0))],
            [0.0, float(registration.get("scale", 1.0)), float(registration.get("ty", 0.0))],
        ],
        dtype=np.float64,
    )


def _affine_parameters_from_matrix(matrix: np.ndarray, preview_scale: float) -> np.ndarray:
    sx = float(matrix[0, 0]) / preview_scale
    sy = float(matrix[1, 1]) / preview_scale
    kxy = float(matrix[0, 1]) / preview_scale
    kyx = float(matrix[1, 0]) / preview_scale
    cx = float(matrix[0, 2]) + sx * 1000.0 + kxy * 400.0
    cy = float(matrix[1, 2]) + kyx * 1000.0 + sy * 400.0
    return np.asarray([sx, sy, kxy, kyx, cx, cy], dtype=np.float64)


def _scaled_affine(parameters: np.ndarray) -> np.ndarray:
    sx, sy, kxy, kyx, cx, cy = parameters
    return np.asarray(
        [
            [sx, kxy, cx - sx * 1000.0 - kxy * 400.0],
            [kyx, sy, cy - kyx * 1000.0 - sy * 400.0],
        ],
        dtype=np.float64,
    )


def _full_affine(parameters: np.ndarray, preview_scale: float) -> np.ndarray:
    matrix = _scaled_affine(parameters)
    matrix[:, :2] *= preview_scale
    return matrix


def _affine_regions(
    raw: Any,
    width: int,
    height: int,
    selected_regions: Sequence[int] | None = None,
) -> list[tuple[int, tuple[int, int, int, int]]]:
    if raw is None:
        raw = [
            [165, 40, 520, max(40, height - 44)],
            [520, 40, 950, max(40, height - 44)],
            [950, 40, 1380, max(40, height - 44)],
        ]
    if not isinstance(raw, (list, tuple)):
        return []
    rois = []
    for region_id, item in enumerate(raw, start=1):
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue
        x0, y0, x1, y1 = (int(value) for value in item)
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        if x1 > x0 and y1 > y0:
            rois.append((region_id, (x0, y0, x1, y1)))
    if selected_regions is not None:
        if (not isinstance(selected_regions, (list, tuple)) or not selected_regions
                or any(isinstance(i, bool) or not isinstance(i, int) for i in selected_regions)
                or len(set(selected_regions)) != len(selected_regions)):
            raise ValueError("selected_regions须为非空、不重复的区域编号列表，如[1,2]")
        available = {region_id for region_id, _ in rois}
        if not set(selected_regions).issubset(available):
            raise ValueError("所选仿射评分区不存在或超出深度片范围，请检查区域编号和ROI")
        rois = [(region_id, box) for region_id, box in rois if region_id in selected_regions]
    return rois


def _affine_rois(raw: Any, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Compatibility wrapper returning every valid scoring box."""
    return [box for _, box in _affine_regions(raw, width, height)]


def align_crop_mask_to_depth(
    crop_mask: np.ndarray,
    crop_roi: Sequence[int],
    depth_shape: tuple[int, int],
    registration: Dict[str, Any],
) -> np.ndarray:
    """Warp a fixed-ROI mask into local coordinates of its matched depth piece."""
    x0, y0, _, _ = (int(value) for value in crop_roi)
    scale = float(registration["scale"])
    tx = float(registration["tx"])
    ty = float(registration["ty"])
    # A caller may provide a calibrated projective transform.  It is defined
    # in full-image coordinates, so convert it to crop-local coordinates
    # before warping the cached 2D mask.  The default remains the historical
    # scale/translation registration, represented as a 3x3 homography.
    full_homography = registration.get("homography_2d_to_3d")
    if full_homography is not None:
        full_homography = np.asarray(full_homography, dtype=np.float64)
        if full_homography.shape != (3, 3):
            raise ValueError("registration.homography_2d_to_3d must be a 3x3 matrix")
        crop_to_full = np.asarray(
            [[1.0, 0.0, float(x0)], [0.0, 1.0, float(y0)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        transform = full_homography @ crop_to_full
    else:
        transform = np.asarray(
            [[scale, 0.0, tx + scale * x0],
             [0.0, scale, ty + scale * y0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    height, width = depth_shape
    return cv2.warpPerspective(
        (np.asarray(crop_mask) > 0).astype(np.uint8) * 255,
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _four_ints(values: Any, name: str) -> tuple[int, int, int, int]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{name} must contain four integers")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def _validate_box(box: Sequence[int], name: str) -> None:
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} is invalid: {tuple(box)}")
