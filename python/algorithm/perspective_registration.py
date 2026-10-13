"""Left-contour perspective correction ported from the supplied experiment."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares


def edges(binary: np.ndarray) -> np.ndarray:
    """Extract only the physical left lead-slot contours used for fitting."""
    dark = (binary == 0).astype(np.uint8) * 255
    edge = cv2.morphologyEx(
        dark, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )
    keep = np.zeros_like(edge)
    keep[120:345, 150:560] = 255
    keep[380:520, 150:560] = 255
    keep[515:635, 150:560] = 255
    # The second slot's lower 3D boundary can be missing; do not fit it.
    keep[605:635, :] = 0
    return cv2.bitwise_and(edge, keep)


def samples(edge_image: np.ndarray, limit: int = 1100) -> np.ndarray:
    yy, xx = np.where(edge_image > 0)
    xy = np.column_stack((xx, yy)).astype(np.float64)
    if len(xy) > limit:
        xy = xy[np.linspace(0, len(xy) - 1, limit).astype(int)]
    return xy


def matrix(parameters: np.ndarray) -> np.ndarray:
    """Bounded projective correction normalized near (700, 380)."""
    p = parameters
    return np.asarray(
        [
            [1 + p[0], p[1], p[2] - 700 * p[0] - 380 * p[1]],
            [p[3], 1 + p[4], p[5] - 700 * p[3] - 380 * p[4]],
            [
                p[6] * 1e-5,
                p[7] * 1e-5,
                1 - 700 * p[6] * 1e-5 - 380 * p[7] * 1e-5,
            ],
        ],
        dtype=np.float64,
    )


def apply(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    denominator = homogeneous[:, 2:3]
    safe = np.where(np.abs(denominator) < 1e-12, np.nan, denominator)
    return homogeneous[:, :2] / safe


def estimate_perspective(
    image_2d: np.ndarray,
    depth_piece: np.ndarray,
    registration: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a bounded perspective correction over the existing coarse affine.

    This follows ``left_contour_perspective_depth.py``: warp the 2D grayscale
    image by the initial affine, fit sampled physical slot contours against a
    3D edge distance field, optionally add five circular-aperture landmarks,
    and return ``H_correction @ H_initial`` as a full 3x3 homography.
    """
    config = config or {}
    base = _base_homography(registration)
    result = dict(registration)
    result["homography_2d_to_3d"] = base.tolist()
    result["perspective_valid"] = False
    result["perspective_inliers"] = 0
    result["perspective_error_px"] = None
    result["perspective_circle_error_px"] = None

    if depth_piece.ndim != 2 or depth_piece.size == 0:
        return result
    gray = (
        cv2.cvtColor(image_2d, cv2.COLOR_BGR2GRAY)
        if image_2d.ndim == 3
        else image_2d
    )
    if gray.ndim != 2:
        return result

    height, width = depth_piece.shape
    warped = cv2.warpAffine(
        gray,
        base[:2].astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
    )
    edge_2d = edges((warped > int(config.get("threshold_2d", 50))).astype(np.uint8))
    edge_3d = edges((depth_piece > 0).astype(np.uint8))
    source = samples(edge_2d, int(config.get("max_edge_samples", 1100)))
    if len(source) < 8 or cv2.countNonZero(edge_3d) == 0:
        return result

    center_2d, circle_3d = _circle_centers(gray, depth_piece, config)
    distance = cv2.distanceTransform(
        (edge_3d == 0).astype(np.uint8), cv2.DIST_L2, 3
    )
    initial_guess = np.zeros(8, dtype=np.float64)

    if center_2d is not None and circle_3d is not None:
        center = apply(base, np.asarray([center_2d[:2]], dtype=np.float64))[0]
        radius_2d = float(center_2d[2])
        radius_3d = float(circle_3d[2])
        circle_source = np.asarray(
            [
                [center[0], center[1]],
                [center[0] - radius_2d * base[0, 0], center[1]],
                [center[0] + radius_2d * base[0, 0], center[1]],
                [center[0], center[1] - radius_2d * base[1, 1]],
                [center[0], center[1] + radius_2d * base[1, 1]],
            ],
            dtype=np.float64,
        )
        circle_target = np.asarray(
            [
                [circle_3d[0], circle_3d[1]],
                [circle_3d[0] - radius_3d, circle_3d[1]],
                [circle_3d[0] + radius_3d, circle_3d[1]],
                [circle_3d[0], circle_3d[1] - radius_3d],
                [circle_3d[0], circle_3d[1] + radius_3d],
            ],
            dtype=np.float64,
        )
    else:
        circle_source = None
        circle_target = None

    try:
        correction = _fit_correction(
            source,
            distance,
            (width, height),
            circle_source,
            circle_target,
            config,
        )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
        return result

    total = correction @ base
    mapped_source = apply(correction, source)
    x = np.clip(mapped_source[:, 0].astype(np.float32), 0, width - 1)
    y = np.clip(mapped_source[:, 1].astype(np.float32), 0, height - 1)
    error = cv2.remap(
        distance,
        x.reshape(1, -1),
        y.reshape(1, -1),
        cv2.INTER_LINEAR,
    ).reshape(-1)

    result["homography_2d_to_3d"] = total.tolist()
    result["matrix_2d_to_3d"] = total[:2].tolist()
    result["perspective_valid"] = True
    result["perspective_inliers"] = int(len(source))
    result["perspective_error_px"] = float(np.mean(error))
    if circle_source is not None and circle_target is not None:
        mapped_circle = apply(correction, circle_source)
        result["perspective_circle_error_px"] = float(
            np.linalg.norm(mapped_circle[0] - circle_target[0])
        )
    return result


def _fit_correction(
    source: np.ndarray,
    distance: np.ndarray,
    shape: tuple[int, int],
    circle_source: np.ndarray | None,
    circle_target: np.ndarray | None,
    config: dict[str, Any],
) -> np.ndarray:
    width, height = shape

    def residual(parameters: np.ndarray) -> np.ndarray:
        correction = matrix(parameters)
        mapped = apply(correction, source)
        mx = np.clip(mapped[:, 0].astype(np.float32), 0, width - 1)
        my = np.clip(mapped[:, 1].astype(np.float32), 0, height - 1)
        chamfer = cv2.remap(
            distance,
            mx.reshape(1, -1),
            my.reshape(1, -1),
            cv2.INTER_LINEAR,
        ).ravel()
        chamfer = np.minimum(chamfer, 12) * 0.22
        regular = np.asarray(
            [
                parameters[0] * 20,
                parameters[1] * 20,
                parameters[2] * 0.15,
                parameters[3] * 20,
                parameters[4] * 30,
                parameters[5] * 0.15,
                parameters[6] * 2,
                parameters[7] * 4,
            ],
            dtype=np.float64,
        )
        if circle_source is None or circle_target is None:
            return np.r_[chamfer, regular]
        landmark = (apply(correction, circle_source) - circle_target).ravel()
        return np.r_[chamfer, landmark * 0.8, regular]

    lower = np.asarray([-0.025, -0.015, -16, -0.015, -0.012, -16, -2, -1.5])
    upper = np.asarray([0.025, 0.015, 16, 0.015, 0.012, 16, 2, 1.5])
    result = least_squares(
        residual,
        np.zeros(8, dtype=np.float64),
        bounds=(lower, upper),
        max_nfev=int(config.get("max_nfev", 45)),
    )
    return matrix(result.x)


def _circle_centers(
    gray_2d: np.ndarray,
    depth_piece: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    scale = float(config.get("circle_search_scale", 0.25))
    small = cv2.resize(
        gray_2d, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    circles_2d = cv2.HoughCircles(
        cv2.GaussianBlur(small, (0, 0), 2),
        cv2.HOUGH_GRADIENT,
        1.2,
        80,
        param1=80,
        param2=25,
        minRadius=int(config.get("circle_2d_min_radius", 115)),
        maxRadius=int(config.get("circle_2d_max_radius", 165)),
    )
    candidates_2d = (
        []
        if circles_2d is None
        else [
            circle / scale
            for circle in circles_2d[0]
            if 3800 < circle[0] / scale < 4800
            and 900 < circle[1] / scale < 1800
            and 530 < circle[2] / scale < 585
        ]
    )
    center_2d = (
        min(
            candidates_2d,
            key=lambda circle: abs(circle[2] - 558) + abs(circle[1] - 1330) / 5,
        )
        if candidates_2d
        else None
    )

    normalized = cv2.normalize(
        depth_piece, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    circles_3d = cv2.HoughCircles(
        cv2.GaussianBlur(normalized, (0, 0), 5),
        cv2.HOUGH_GRADIENT,
        1.2,
        100,
        param1=30,
        param2=20,
        minRadius=int(config.get("circle_3d_min_radius", 135)),
        maxRadius=int(config.get("circle_3d_max_radius", 205)),
    )
    candidates_3d = (
        []
        if circles_3d is None
        else [
            circle
            for circle in circles_3d[0]
            if 1080 < circle[0] < 1250
            and 220 < circle[1] < 500
            and 150 < circle[2] < 190
        ]
    )
    center_3d = (
        min(
            candidates_3d,
            key=lambda circle: abs(circle[2] - 168) + abs(circle[1] - 360) / 4,
        )
        if candidates_3d
        else None
    )
    return center_2d, center_3d


def _base_homography(registration: dict[str, Any]) -> np.ndarray:
    raw = registration.get("homography_2d_to_3d")
    if raw is not None:
        homography = np.asarray(raw, dtype=np.float64)
        if homography.shape == (3, 3) and abs(float(homography[2, 2])) > 1e-12:
            return homography / homography[2, 2]
    scale = float(registration.get("scale", 1.0))
    tx = float(registration.get("tx", 0.0))
    ty = float(registration.get("ty", 0.0))
    return np.asarray(
        [[scale, 0.0, tx], [0.0, scale, ty], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
