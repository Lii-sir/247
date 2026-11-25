"""Assemble per-piece inspection responses."""
import time

from algorithm.runtime.defect_plan import CHECKS_3D, unrun_checks
from transport.depth_input import depth_coordinate_space


def join_piece_diagnostics(piece, entry):
    """Combine registered 3D results and final 2D diagnostics."""
    public_names = ("chip_anomaly_2d", "silver_overflow_2d", "silver_continuity_2d")
    checks = (entry.get(name) for name in (*public_names, "chip_rotation_2d"))
    diagnostics_ok = all(
        check.get("is_ok") is True for check in checks
        if isinstance(check, dict) and check.get("status") not in {None, "disabled", "skipped"}
    )
    result = {
        **piece,
        "is_ok": bool(piece["is_ok"] and diagnostics_ok),
        **{name: entry.get(name, {"status": "disabled", "is_ok": None}) for name in public_names},
    }
    if entry.get("first_ng_check") is not None:
        result.update(first_ng_check=entry["first_ng_check"], is_ok=False)
    return result


def skipped_3d_piece(piece, pitch_correction, reason, entry=None, *, height_order=None):
    entry = entry or {}
    disabled = reason == "no_3d_checks"
    result = {
        "piece_id": int(piece["piece_id"]),
        "depth_start_row": int(piece["y1"]),
        "depth_end_row": int(piece["y2"]),
        "pitch_correction": pitch_correction,
        "depth_coordinate_space": piece.get("depth_coordinate_space") or depth_coordinate_space(pitch_correction),
        "skip_3d_reason": reason,
        "is_ok": disabled,
        "registration": {"is_valid": False, "status": "disabled" if disabled else "skipped"},
        "masks": [],
        "wire_measurements": [],
        "solder_measurements": [],
    }
    order = entry.get("height_order", height_order)
    if order is not None:
        result["defect_checks_3d"] = unrun_checks(order, CHECKS_3D)
    for target, source in (("image_2d_size", "image_shape"), ("crop_2d", "crop_box"),
                           ("wire_rois_2d", "roi_boxes"), ("wire_roi_checks", "wire_roi_checks"),
                           ("first_ng_check", "first_ng_check")):
        if source in entry:
            result[target] = entry[source]
    return result


def piece_3d_response(board_id, piece, started, timings):
    elapsed_ms = (time.perf_counter() - started) * 1000
    return dict(ok=True, cmd="detect_3d_piece", stage="3d", board_id=board_id,
                piece_id=piece["piece_id"], results_piece=piece, elapsed_ms=elapsed_ms,
                timings={**timings, "total_detect_ms": elapsed_ms})
