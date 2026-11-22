"""Per-piece 2D check order and short-circuit rules."""
from algorithm.runtime.timing import timed_call


CHECKS_2D = (
    "wire_bending_2d",
    "chip_rotation_2d",
    "chip_anomaly_2d",
    "silver_overflow_2d",
    "silver_continuity_2d",
)


def detection_order(value):
    """An empty list keeps the parallel path; unlisted checks follow in default order."""
    if not isinstance(value, list) or any(name not in CHECKS_2D for name in value):
        raise ValueError("common_2d.parameters.defect_priority must list 2D check names: "
                         + ", ".join(CHECKS_2D))
    if len(set(value)) != len(value):
        raise ValueError("common_2d.parameters.defect_priority contains duplicate checks")
    if not value:
        return ()
    return (*value, *(name for name in CHECKS_2D if name not in value))


def evaluate_in_order(order, checks):
    """Each enabled check is a callable. A check without a passing result is NG."""
    results = {}
    first_ng = None
    for name in order:
        run = checks[name]
        if run is None:
            results[name] = {"enabled": False, "status": "disabled", "is_ok": None}
        elif first_ng is not None:
            results[name] = {"enabled": True, "status": "skipped", "is_ok": None}
        else:
            result = results[name] = run()
            if result.get("status") != "disabled" and result.get("is_ok") is not True:
                first_ng = name
    return results, first_ng


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
