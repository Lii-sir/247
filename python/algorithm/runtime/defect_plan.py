"""Shared configuration and execution rules for per-piece defect checks."""


CHECKS_2D = (
    "wire_bending_2d",
    "chip_rotation_2d",
    "chip_anomaly_2d",
    "silver_overflow_2d",
    "silver_continuity_2d",
)
WIRE_HEIGHT = "wire_height_3d"
PLATFORM_SOLDER_HEIGHT = "solder_platform_height_3d"
CHIP_SOLDER_HEIGHT = "solder_chip_height_3d"
CHECKS_3D = (WIRE_HEIGHT, PLATFORM_SOLDER_HEIGHT, CHIP_SOLDER_HEIGHT)


def defect_orders(settings):
    """None means legacy behavior; an empty stage tuple means no checks selected."""
    sequence = settings.get("defect_sequence")
    if sequence is None:
        return detection_order(settings.get("defect_priority", [])) or None, None
    checks = CHECKS_2D + CHECKS_3D
    if not isinstance(sequence, list) or any(name not in checks for name in sequence):
        raise ValueError("inspection.defect_sequence must be a list of: "
                         + ", ".join(checks))
    if len(set(sequence)) != len(sequence):
        raise ValueError("inspection.defect_sequence contains duplicate checks")
    order_2d = tuple(name for name in sequence if name in CHECKS_2D)
    order_3d = tuple(name for name in sequence if name in CHECKS_3D)
    return order_2d, order_3d


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


def evaluate_in_order(order, checks, *, short_circuit=True):
    """Each enabled check is a callable. A check without a passing result is NG."""
    results = unrun_checks(tuple(name for name in order if checks[name] is not None), checks)
    first_ng = None
    for name in order:
        run = checks[name]
        if run is not None and (first_ng is None or not short_circuit):
            result = results[name] = run()
            if first_ng is None and failed_check(result):
                first_ng = name
    return results, first_ng


def failed_check(result):
    return result.get("status") not in {"disabled", "skipped"} and result.get("is_ok") is not True


def unrun_checks(order, names):
    return {name: {"enabled": name in order, "status": "skipped" if name in order else "disabled",
                   "is_ok": None} for name in names}


def require_configured_checks(order, settings):
    missing = [name for name in order if name in settings and not settings[name]]
    if missing:
        raise ValueError("Selected defect checks require configuration: " + ", ".join(missing))
