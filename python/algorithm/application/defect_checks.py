"""Adapters for per-piece 2D defect checks."""
from algorithm.runtime.timing import timed_call


def wire_bending_result(measurements, roi_checks):
    is_ok = bool(roi_checks) and all(item["is_ok"] for item in roi_checks)
    is_ok = is_ok and all(item["bending_2d"]["is_ok"] for item in measurements)
    return {"enabled": True, "status": "ok" if is_ok else "ng", "is_ok": is_ok,
            "roi_checks": roi_checks, "measurements": measurements}


def silver_check(name, config, instances, shape, *, crop_box, circle, image, base_dir):
    """Invoke one silver check so later checks (including boundary preparation) stay lazy."""
    from algorithm.vision2d import silver_inspection as silver
    settings_key, result_key = ("overflow", "silver_overflow") if name == "silver_overflow_2d" else (
        "continuity", "disconnection")
    diagnostics = timed_call("silver_mask_checks_ms", silver.inspect_masks,
                             instances, shape, {settings_key: config}, crop_box=crop_box,
                             circle=circle, image=image, base_dir=base_dir)
    return {"enabled": True, **silver.offset_geometry(diagnostics[result_key], crop_box[:2])}
