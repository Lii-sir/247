"""Assemble per-piece inspection results into the current whole-board response."""
import time

from transport.depth_input import depth_coordinate_space


def join_piece_diagnostics(piece, entry):
    """Combine registered 3D results and final 2D diagnostics."""
    public_names = ("chip_anomaly_2d", "silver_overflow_2d", "silver_continuity_2d")
    checks = (entry.get(name) for name in (*public_names, "chip_rotation_2d"))
    diagnostics_ok = all(
        check.get("is_ok") is True for check in checks
        if isinstance(check, dict) and check.get("status") not in {None, "disabled", "skipped"}
    )
    return {
        **piece,
        "is_ok": bool(piece["is_ok"] and diagnostics_ok),
        **{name: entry.get(name, {"status": "disabled", "is_ok": None}) for name in public_names},
    }


def skipped_3d_piece(piece, pitch_correction, reason, entry=None):
    entry = entry or {}
    result = {
        "piece_id": int(piece["piece_id"]),
        "depth_start_row": int(piece["y1"]),
        "depth_end_row": int(piece["y2"]),
        "pitch_correction": pitch_correction,
        "depth_coordinate_space": piece.get("depth_coordinate_space") or depth_coordinate_space(pitch_correction),
        "skip_3d_reason": reason,
        "is_ok": False,
        "registration": {"is_valid": False, "status": "skipped"},
        "masks": [],
        "wire_measurements": [],
        "solder_measurements": [],
    }
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


def board_results(pieces, pitch_correction):
    registration_ok = all(piece["registration"]["is_valid"] for piece in pieces)
    return {
        "pitch_correction": pitch_correction,
        "depth_coordinate_space": depth_coordinate_space(pitch_correction),
        "registration_ok": registration_ok,
        "is_ok": registration_ok and all(piece["is_ok"] for piece in pieces),
        "pieces": pieces,
    }
