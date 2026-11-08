"""Fixed-scale registration from a full 2D image to one local 3D depth piece."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Sequence

import cv2
import numpy as np
from scipy.optimize import minimize, OptimizeResult


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
    *, context: Dict[str, Any] | None = None,
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
    result["is_valid"] = False
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
    result["affine_score_mode"] = score_mode

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

    cache_enabled = values.get("cache_enabled", False)
    if type(cache_enabled) is not bool:
        raise ValueError("dynamic_affine.cache_enabled must be boolean")
    score = _AffineScore(source_mask, target_mask, rois, penalty_weight, shear_limit,
                         score_mode=score_mode, cache_enabled=cache_enabled)

    before, _ = score(p0, details=True)
    calibration = values.get("calibration", {})
    if calibration is None:calibration = {}
    if not isinstance(calibration, dict) or type(calibration.get("enabled", False)) is not bool:
        raise ValueError("dynamic_affine.calibration must contain a boolean enabled")
    start = p0.copy()
    initial_score = before
    initialization = dict(enabled=calibration.get("enabled", False), source="coarse", reason="disabled",
                          coarse_score=before, candidate_score=None, profile_valid=False)
    if initialization["enabled"]:
        try:
            candidate = _calibration_seed(calibration.get("profile"), context, p0, preview_scale, lower, upper)
        except ValueError as exc:
            initialization["reason"] = str(exc)
        else:
            candidate_score = score(candidate)
            initialization.update(profile_valid=True, candidate_score=float(candidate_score) if np.isfinite(candidate_score) else None)
            if np.isfinite(candidate_score) and candidate_score < before:
                start, initial_score = candidate, candidate_score
                initialization.update(source="calibration_linear", reason="candidate_improved_full_score")
            else:
                initialization["reason"] = "candidate_did_not_improve_full_score"
    result.update(affine_initialization=initialization, affine_initial_parameters=start.tolist(),
                  affine_initial_score=float(initial_score), affine_coarse_parameters=p0.tolist())
    optimizer_calls_before = score.calls
    optimizer_started = time.perf_counter()
    stages, objectives = [], [score]
    try:
        optimized, stages, objectives, search_info = _search_affine(
            score, start, p0, lower, upper, values, initial_score)
        if search_info.get("fatal_error"):
            raise ValueError(search_info["fatal_error"])
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        result["affine_error"] = str(exc)
        result["affine_score_before"] = before
        result.update(affine_stop_reason=str(exc), affine_optimizer_success=False,
            affine_function_evaluations=sum(stage["nfev"] for stage in stages) if stages else max(0, score.calls-optimizer_calls_before))
        result.update(_score_diagnostics(objectives))
        result["affine_stages"] = stages
        return result

    result["affine_optimization_pipeline_ms"] = (time.perf_counter()-optimizer_started)*1000
    result["affine_stages"] = stages
    result["affine_multiscale"] = search_info
    result["affine_budget"] = search_info.get("budget")
    result["affine_optimizer_ms"] = sum(stage["optimizer_ms"] for stage in stages)
    result["affine_function_evaluations"] = sum(stage["nfev"] for stage in stages)
    result["affine_optimizer_success"] = bool(optimized.success) if hasattr(optimized,"success") else None
    result["affine_optimizer_status"] = int(optimized.status) if hasattr(optimized,"status") else None
    result["affine_stop_reason"] = str(getattr(optimized,"message","optimizer did not report a stop reason"))
    after, losses = score(optimized.x, details=True)
    result.update(_score_diagnostics(objectives))
    matrix = _full_affine(optimized.x, preview_scale)
    result["affine_valid"] = bool(
        np.all(np.isfinite(matrix))
        and float(after) <= float(values.get("max_score", 0.25))
    )
    result["affine_score_before"] = float(before)
    result["affine_score_after"] = float(after)
    result["affine_zone_losses"] = [float(value) for value in losses]
    result["affine_iterations"] = sum(stage["iterations"] for stage in stages)
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


def affine_coordinate_context(image_shape, depth_shape, correction=None, image_role="independent_camera"):
    """Geometry contract shared by formal detection and the registration lab."""
    correction = correction or {}
    prep = dict(model=correction.get("model", "smartray_pitch"), image_role=image_role,
                state="applied" if correction.get("applied") else correction.get("reason", "disabled"))
    if correction.get("applied"):
        for key in ("angle_deg", "scan_axis", "scan_pitch_mm", "transform_direction",
                    "z_step_mm", "gray_offset", "z_origin_mm"):
            prep[key] = correction.get(key)
    return dict(source_space="independent_2d_image" if image_role == "independent_camera" else "smartray_intensity_depth_piece",
                target_space="depth_piece_local", image_size=[int(image_shape[1]), int(image_shape[0])],
                depth_size=[int(depth_shape[1]), int(depth_shape[0])], preprocessing=prep)


def _calibration_seed(profile, context, coarse, preview_scale, lower, upper):
    """Reuse only linear4. Current coarse anchor position is never replaced."""
    if (not isinstance(profile, dict) or profile.get("format") != "registration_lab_matrix"
            or type(profile.get("version")) is not int or profile["version"] != 1):
        raise ValueError("calibration_requires_registration_lab_matrix_profile")
    saved = profile.get("context")
    if not isinstance(saved, dict) or not isinstance(context, dict):
        raise ValueError("calibration_coordinate_context_missing")
    for key in ("source_space", "target_space", "preprocessing"):
        if key not in saved or saved[key] != context.get(key):
            raise ValueError("calibration_coordinate_or_pitch_mismatch")
    if context.get("source_space") != "independent_2d_image":
        raise ValueError("calibration_requires_independent_2d_image")
    for owner in (saved, context):
        for key in ("image_size", "depth_size"):
            size = owner.get(key)
            if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
                raise ValueError("calibration_invalid_image_size")
    if saved["image_size"] != context["image_size"] or saved["depth_size"][0] != context["depth_size"][0]:
        raise ValueError("calibration_image_resolution_mismatch")
    raw = profile.get("matrix")
    if (not isinstance(raw, list) or len(raw) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in raw)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for row in raw for v in row)):
        raise ValueError("calibration_matrix_must_be_numeric_3x3")
    matrix = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(matrix).all() or abs(matrix[2,2]) < 1e-12:
        raise ValueError("calibration_nonfinite_or_invalid_matrix")
    matrix = matrix / matrix[2,2]
    if not np.allclose(matrix[2,:2],0,rtol=0,atol=1e-12) or np.linalg.det(matrix[:2,:2]) <= 0:
        raise ValueError("calibration_requires_orientation_preserving_affine")
    candidate = _affine_parameters_from_matrix(matrix[:2], preview_scale)
    candidate[4:] = coarse[4:]
    if np.any(candidate < lower) or np.any(candidate > upper):
        raise ValueError("calibration_candidate_outside_original_bounds")
    return candidate


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
    *,
    binary_255: bool = False,
) -> np.ndarray:
    """Warp a crop mask; binary_255 trusts a producer's uint8 0/255 contract.

    Unmarked compatibility inputs retain positive-value binarization. Neither
    path modifies its source, including read-only shared-memory views.
    """
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
    source = np.asarray(crop_mask)
    if not binary_255 or source.dtype != np.uint8:
        source = (source > 0).astype(np.uint8) * 255
    return cv2.warpPerspective(
        source,
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


class _AffineScore:
    """One image/one scoring resolution. Exact keys; no cross-image cache."""
    def __init__(self, source, target, rois, penalty_weight, shear_limit,
                 *, score_mode="full", cache_enabled=False, ratios=(1.,1.,1.,1.), scale=1.):
        self.source, self.target, self.rois = source, target, rois
        self.ratios, self.scale = ratios, scale
        self.penalty_weight, self.shear_limit = penalty_weight, shear_limit
        self.score_mode, self.cache_enabled = score_mode, cache_enabled
        self.grids = [_affine_roi_grid(box) for box in rois] if score_mode == "roi" else None
        self.targets = [target[y0:y1,x0:x1] for x0,y0,x1,y1 in rois]
        # A scorer belongs to one piece/resolution and Powell calls it serially.
        # Keep these private: different pieces and multiscale levels must not
        # overwrite each other's warp or comparison output.
        self._warp_buffer = np.empty(target.shape, dtype=source.dtype) if score_mode == "full" else None
        self._difference_buffers = [np.empty(item.shape, dtype=np.bool_) for item in self.targets]
        self.cache = {}
        self.calls = self.hits = self.misses = self.warps = 0
        self.seconds = self.compute_seconds = 0.

    def __call__(self, parameters, details=False):
        started = time.perf_counter()
        self.calls += 1
        key = tuple(float(v) for v in parameters)
        if self.cache_enabled and key in self.cache:
            self.hits += 1
            value, zones = self.cache[key]
        else:
            self.misses += 1
            compute_started = time.perf_counter()
            matrix = _scaled_affine(parameters)
            if self.ratios != (1.,1.,1.,1.):
                sx,sy,tx,ty = self.ratios
                matrix[:,0] /= sx
                matrix[:,1] /= sy
                matrix[0,:] *= tx
                matrix[1,:] *= ty
            if self.score_mode == "roi":
                patches = _warp_affine_rois(self.source, matrix, self.rois, self.grids, self.target.shape[1])
            else:
                warped = cv2.warpAffine(self.source, matrix, (self.target.shape[1],self.target.shape[0]),
                    dst=self._warp_buffer, flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                patches = (warped[y0:y1,x0:x1] for x0,y0,x1,y1 in self.rois)
            zones = tuple(float(np.count_nonzero(np.not_equal(patch, target, out=difference)) / target.size)
                          for patch,target,difference in zip(patches,self.targets,self._difference_buffers))
            penalty = self.penalty_weight * ((parameters[2]/max(self.shear_limit,1e-6))**2
                                            +(parameters[3]/max(self.shear_limit,1e-6))**2)
            value = float(np.mean(zones)+penalty)
            self.warps += 1
            self.compute_seconds += time.perf_counter()-compute_started
            if self.cache_enabled and np.isfinite(value):
                self.cache[key] = (value,zones)
        self.seconds += time.perf_counter()-started
        return (value,list(zones)) if details else value

    def diagnostics(self):
        return dict(affine_cache_enabled=self.cache_enabled,affine_cache_hits=self.hits,
            affine_cache_misses=self.misses,affine_cache_hit_rate=self.hits/max(1,self.calls),
            affine_warp_evaluations=self.warps,affine_score_calls=self.calls,
            affine_score_total_ms=self.seconds*1000,affine_score_mean_ms=self.seconds*1000/max(1,self.calls),
            affine_score_compute_ms=self.compute_seconds*1000,
            affine_score_compute_mean_ms=self.compute_seconds*1000/max(1,self.misses))

    def resized(self, scale):
        if self.score_mode != "full":
            raise ValueError("multiscale requires native full score_mode")
        sh,sw = self.source.shape
        th,tw = self.target.shape
        ss = (max(1,round(sw*scale)),max(1,round(sh*scale)))
        ts = (max(1,round(tw*scale)),max(1,round(th*scale)))
        rx,ry = ts[0]/tw,ts[1]/th
        rois = [(max(0,int(np.floor(x0*rx))),max(0,int(np.floor(y0*ry))),
                 min(ts[0],int(np.ceil(x1*rx))),min(ts[1],int(np.ceil(y1*ry))))
                for x0,y0,x1,y1 in self.rois]
        if any(x1<=x0 or y1<=y0 for x0,y0,x1,y1 in rois):
            raise ValueError("multiscale produced an empty scoring region")
        return _AffineScore(cv2.resize(self.source,ss,interpolation=cv2.INTER_NEAREST),
            cv2.resize(self.target,ts,interpolation=cv2.INTER_NEAREST),rois,self.penalty_weight,self.shear_limit,
            cache_enabled=self.cache_enabled,ratios=(ss[0]/sw,ss[1]/sh,rx,ry),scale=scale)


def _score_diagnostics(objectives):
    # The full objective may be reused by refine/validation/fallback. Sum it once.
    unique = {id(objective):objective for objective in objectives}.values()
    rows = [objective.diagnostics() for objective in unique]
    totals = {key:sum(row[key] for row in rows) for key in (
        "affine_cache_hits","affine_cache_misses","affine_warp_evaluations","affine_score_calls",
        "affine_score_total_ms","affine_score_compute_ms")}
    totals.update(affine_cache_enabled=all(row["affine_cache_enabled"] for row in rows),
        affine_cache_hit_rate=totals["affine_cache_hits"]/max(1,totals["affine_score_calls"]),
        affine_score_mean_ms=totals["affine_score_total_ms"]/max(1,totals["affine_score_calls"]),
        affine_score_compute_mean_ms=totals["affine_score_compute_ms"]/max(1,totals["affine_cache_misses"]))
    return totals


def _optional_evaluation_limit(value, name):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(name + " must be null or a positive integer")
    return value


def _run_affine_level(score, start, lower, upper, values, name, maxiter, *, maxfev=None, budget=None):
    started = time.perf_counter()
    before_stats = score.diagnostics()
    calls_before = score.calls
    error = None
    limit = _optional_evaluation_limit(maxfev,"maxfev")
    remaining = None if budget is None or budget["limit"] is None else max(0,budget["limit"]-budget["used"])
    if remaining is not None:
        limit = remaining if limit is None else min(limit,remaining)
    best_value,best_params = np.inf,None
    def objective(parameters):
        nonlocal best_value,best_params
        value = score(parameters)
        point = np.asarray(parameters,dtype=float)
        if (np.isfinite(value) and np.isfinite(point).all() and np.all(point>=lower)
                and np.all(point<=upper) and value<best_value):
            best_value,best_params = float(value),point.copy()
        return value
    skipped = limit == 0
    if skipped:
        optimized = OptimizeResult(x=start.copy(),nit=0,nfev=0,status=1,success=False,
                                   message="total evaluation budget exhausted; stage not run")
    else:
        options = dict(maxiter=int(maxiter),xtol=float(values.get("xtol",1e-4)),ftol=float(values.get("ftol",1e-4)))
        if limit is not None:options["maxfev"] = limit
        try:
            # Best-trial tracking is needed only for an active evaluation cap;
            # avoid its extra per-call checks on the default unlimited path.
            optimized = minimize(objective if limit is not None else score,start,
                                 method="Powell",bounds=list(zip(lower,upper)),options=options)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            error = str(exc)
            optimized = OptimizeResult(x=start.copy(),nit=0,nfev=score.calls-calls_before,
                                       status=-1,success=False,message=error)
    observed_calls = score.calls-calls_before
    if budget is not None:budget["used"] += observed_calls
    exhausted = skipped or getattr(optimized,"status",None) == 1 or (limit is not None and observed_calls>=limit)
    returned = np.asarray(optimized.x,dtype=float).copy()
    use_best = exhausted and best_params is not None
    if use_best:
        optimized = OptimizeResult(x=best_params.copy(),fun=best_value,nit=getattr(optimized,"nit",0),
            nfev=getattr(optimized,"nfev",observed_calls),status=getattr(optimized,"status",1),
            success=getattr(optimized,"success",False),message=getattr(optimized,"message","evaluation budget exhausted"))
    stage = dict(name=name,scale=score.scale,source_size=list(score.source.shape[::-1]),
        target_size=list(score.target.shape[::-1]),nfev=int(getattr(optimized,"nfev",score.calls-calls_before)),
        iterations=int(getattr(optimized,"nit",0) or 0),optimizer_ms=(time.perf_counter()-started)*1000,
        success=bool(optimized.success) if hasattr(optimized,"success") else None,
        status=int(optimized.status) if hasattr(optimized,"status") else None,
        stop_reason=str(getattr(optimized,"message","not reported")),error=error,
        initial_parameters=start.tolist(),bounds=np.column_stack((lower,upper)).tolist(),
        maxfev=limit,objective_calls=observed_calls,budget_exhausted=bool(exhausted),skipped=skipped,
        optimizer_returned_parameters=returned.tolist(),used_best_observed=bool(use_best),
        best_observed_score=float(best_value) if np.isfinite(best_value) else None)
    own_score = getattr(optimized,"fun",None)
    stage["objective_score_after"] = float(own_score) if own_score is not None and np.isfinite(own_score) else None
    after_stats = score.diagnostics()
    stage.update(cache_hits=after_stats["affine_cache_hits"]-before_stats["affine_cache_hits"],
        cache_misses=after_stats["affine_cache_misses"]-before_stats["affine_cache_misses"],
        warp_evaluations=after_stats["affine_warp_evaluations"]-before_stats["affine_warp_evaluations"],
        score_compute_ms=after_stats["affine_score_compute_ms"]-before_stats["affine_score_compute_ms"])
    return optimized,stage


def _search_affine(full, start, coarse, lower, upper, values, initial_score):
    settings = values.get("multiscale", {})
    if settings is None:settings = {}
    if not isinstance(settings,dict) or type(settings.get("enabled",False)) is not bool:
        raise ValueError("dynamic_affine.multiscale must contain a boolean enabled")
    enabled = settings.get("enabled",False)
    info = dict(enabled=enabled,applied=False,fallback_used=False,fallback_reason=None,solution_source="full")
    limit = _optional_evaluation_limit(values.get("maxfev"),"maxfev")
    budget = dict(limit=_optional_evaluation_limit(values.get("total_maxfev"),"total_maxfev"),used=0)
    stages,objectives = [],[full]
    if not enabled:
        optimized,stage = _run_affine_level(full,start,lower,upper,values,"full",values.get("maxiter",35),maxfev=limit,budget=budget)
        stages.append(stage)
        info["fatal_error"] = stage["error"]
        info["budget"] = dict(budget,exhausted=budget["limit"] is not None and budget["used"]>=budget["limit"])
        return optimized,stages,objectives,info
    scale = float(settings.get("search_scale",.5))
    ratio = float(settings.get("refine_window_ratio",.25))
    if not np.isfinite(scale) or not 0 < scale < 1 or not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("multiscale search_scale must be (0,1); refine_window_ratio must be (0,1]")
    low = full.resized(scale)
    objectives.append(low)
    info.update(applied=True,solution_source="refine",search_scale=scale)
    searched,search_stage = _run_affine_level(low,start,lower,upper,values,"search",settings.get("search_maxiter",15),
        maxfev=settings.get("search_maxfev") if settings.get("search_maxfev") is not None else limit,budget=budget)
    stages.append(search_stage)
    reason = "search_error" if search_stage["error"] else None
    candidate = np.asarray(searched.x,dtype=float)
    refined = searched
    candidates = [(float(initial_score),start.copy(),"initial")]
    if not reason and np.isfinite(candidate).all() and np.all(candidate>=lower) and np.all(candidate<=upper):
        search_stage["full_score_after"] = full(candidate)
        radius = (upper-lower)*.5*ratio
        fine_lower,fine_upper = np.maximum(lower,candidate-radius),np.minimum(upper,candidate+radius)
        refined,refine_stage = _run_affine_level(full,candidate,fine_lower,fine_upper,values,
            "refine",settings.get("refine_maxiter",10),
            maxfev=settings.get("refine_maxfev") if settings.get("refine_maxfev") is not None else limit,budget=budget)
        stages.append(refine_stage)
        parameters = np.asarray(refined.x,dtype=float)
        if refine_stage["error"] or not np.isfinite(parameters).all():
            reason = "refine_error"
        else:
            after = full(parameters)
            refine_stage["full_score_after"] = after
            if np.isfinite(after):candidates.append((float(after),parameters.copy(),"refine"))
            margin = np.maximum((upper-lower)*1e-3,float(values.get("xtol",1e-4))*2)
            hit = ((parameters-fine_lower <= margin)&(fine_lower>lower+1e-12)) | ((fine_upper-parameters<=margin)&(fine_upper<upper-1e-12))
            if not np.isfinite(after):reason="nonfinite_full_score"
            elif after > initial_score+1e-12:reason="full_score_worse_than_initial"
            elif after > float(values.get("max_score",.25)):reason="full_score_above_threshold"
            elif np.any(hit):reason="refine_window_boundary"
    elif not reason:
        reason = "invalid_search_parameters"
    if reason:
        info.update(fallback_used=True,fallback_reason=reason)
        refined,fallback_stage = _run_affine_level(full,coarse,lower,upper,values,"fallback",values.get("maxiter",35),maxfev=limit,budget=budget)
        stages.append(fallback_stage)
        if fallback_stage["skipped"]:
            info.update(fallback_used=False,fallback_not_run="total_evaluation_budget_exhausted")
        if fallback_stage["error"] and len(candidates) == 1:
            info["fatal_error"] = fallback_stage["error"]
        if not fallback_stage["error"] and np.isfinite(refined.x).all():
            value = full(refined.x)
            fallback_stage["full_score_after"] = value
            if np.isfinite(value):candidates.append((float(value),np.asarray(refined.x).copy(),"fallback"))
    # Retain the best of actual full-resolution candidate/seed checks. A low
    # resolution loss is never compared with any full-resolution loss.
    _,parameters,source = min(candidates,key=lambda item:item[0])
    info["solution_source"] = source
    info["budget"] = dict(budget,exhausted=budget["limit"] is not None and budget["used"]>=budget["limit"])
    refined = OptimizeResult(x=parameters,nit=getattr(refined,"nit",0),nfev=getattr(refined,"nfev",0),
        success=getattr(refined,"success",None),status=getattr(refined,"status",0),message=getattr(refined,"message","not reported"))
    return refined,stages,objectives,info
