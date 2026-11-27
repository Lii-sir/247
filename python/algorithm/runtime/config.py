import json
from pathlib import Path
from typing import Any, Dict, List


_PROJECT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"

DEFAULT_ALGORITHM_PARAMETERS: Dict[str, Any] = {
    # YOLO 默认参数；模型配置里的 algorithm.parameters 会覆盖这些值。
    "imgsz": 640,
    "conf": 0.25,
}


def default_stage_config_path(stage: str) -> Path:
    manifest = json.loads((_PROJECT_CONFIG_DIR / "config.json").read_text(encoding="utf-8-sig"))
    return (_PROJECT_CONFIG_DIR / manifest[stage]).resolve()


def _combine_feature_settings(parameters: Dict[str, Any], thresholds: Dict[str, Any],
                              path: str) -> Dict[str, Any]:
    combined = dict(parameters)
    for key, value in thresholds.items():
        if key in combined:
            setting_path = f"{path}.{key}"
            if not isinstance(combined[key], dict) or not isinstance(value, dict):
                raise ValueError(f"{setting_path} is defined in both parameters and thresholds")
            value = _combine_feature_settings(combined[key], value, setting_path)
        combined[key] = value
    return combined


def feature_parameters(
        config: Dict[str, Any],
        name: str,
        *,
        required: bool = True) -> Dict[str, Any] | None:
    """Read one production feature block.

    Runtime configuration uses one consistent shape for every algorithm
    feature::

        "feature_name": {
            "description": "...",
            "thresholds": { ... },
            "parameters": { ... }
        }

    There is deliberately no ``enabled`` switch.  A required production
    feature must be present; optional callers can use ``required=False`` and
    treat an absent block as disabled.
    """
    if name not in config:
        if required:
            raise ValueError(f"{name} feature block is required")
        return None
    block = config[name]
    if not isinstance(block, dict):
        raise ValueError(f"{name} must be an object")
    description = block.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{name}.description must be a non-empty string")
    thresholds = block.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"{name}.thresholds must be an object")
    parameters = block.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{name}.parameters must be an object")
    if "enabled" in block or "enabled" in parameters or "enabled" in thresholds:
        raise ValueError(
            f"{name} no longer accepts enabled; remove the feature block "
            "only in an optional integration"
        )
    return _combine_feature_settings(parameters, thresholds, name)


def load_runtime_config(config_text: str, config_path: str | None = None, *,
                        config_3d_path: str | None = None) -> Dict[str, Any]:
    """Load either a stage file or the small project manifest.

    The manifest points at the algorithm parameters and the inspection policy.
    Callers that assemble the request themselves can include an inline inspection object.
    """
    parsed = parse_config_text(config_text) if config_text else {}
    if not isinstance(parsed, dict):
        raise ValueError("runtime config must be an object")
    path_2d = parsed.get("config_2d")
    path_3d = parsed.get("config_3d")
    if not isinstance(path_2d, str) or not isinstance(path_3d, str):
        _load_affine_initializer_profile(parsed, config_3d_path or config_path,
                                         stage_3d=bool(config_3d_path))
        return parsed
    base = (
        Path(config_path).expanduser().resolve().parent
        if config_path
        else _PROJECT_CONFIG_DIR
    )
    first = json.loads((base / path_2d).resolve().read_text(encoding="utf-8-sig"))
    second = json.loads((base / path_3d).resolve().read_text(encoding="utf-8-sig"))
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError("stage config files must contain objects")
    merged = dict(first)
    merged.update({key: value for key, value in second.items() if key != "model_path"})
    if "config_inspection" in parsed:
        policy_path = parsed["config_inspection"]
        if not isinstance(policy_path, str) or not policy_path.strip():
            raise ValueError("config_inspection must be a non-empty path")
        merged["inspection"] = json.loads((base / policy_path).resolve().read_text(encoding="utf-8-sig"))
    _load_affine_initializer_profile(merged, str((base / path_3d).resolve()), stage_3d=True)
    return merged


def inspection_parameters(config, legacy_2d):
    """Read the independent policy, with support for older inline stage settings."""
    settings = config.get("inspection", legacy_2d)
    if not isinstance(settings, dict):
        raise ValueError("inspection configuration must be an object")
    return settings


def _load_affine_initializer_profile(config: Dict[str, Any], config_path: str | None, *,
                                     stage_3d: bool) -> None:
    """Resolve the optional coarse-relative affine profile from profile_path.

    A known 3D config path fixes the base directory, including for product
    manifests. Older inline requests carrying only a 2D path retain their
    existing search paths.
    """
    dynamic = config.get("dynamic_affine_3d")
    if not isinstance(dynamic, dict):
        return
    parameters = dynamic.get("parameters")
    if not isinstance(parameters, dict):
        return
    calibration = parameters.get("calibration")
    if not isinstance(calibration, dict):
        return
    profile_path = calibration.get("profile_path")
    if profile_path is None:
        return
    if not isinstance(profile_path, str) or not profile_path.strip():
        raise ValueError("dynamic_affine_3d.calibration.profile_path must be a non-empty string")

    raw = Path(profile_path).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if config_path:
            stage = Path(config_path).expanduser().resolve()
            candidates.append(stage.parent / raw)
            if not stage_3d and stage.parent.name.casefold() == "2d":
                candidates.append(stage.parent.parent / "3D" / raw)
        if not stage_3d:
            candidates.append(default_stage_config_path("config_3d").parent / raw)

    selected = next((path.resolve() for path in candidates if path.is_file()), None)
    if selected is None:
        searched = ", ".join(str(path.resolve()) for path in candidates)
        raise FileNotFoundError(
            f"dynamic_affine_3d.calibration.profile_path does not exist: {profile_path}; searched {searched}"
        )
    try:
        profile = json.loads(selected.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid affine initializer profile JSON: {selected}") from exc
    if not isinstance(profile, dict):
        raise ValueError("affine initializer profile must be a JSON object")
    calibration["profile"] = profile


def load_model_config(
        model_item: Dict[str, Any],
        defaults: Dict[str, Any],
        log) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    algorithm_parameters = dict(defaults)

    if not isinstance(model_item, dict):
        log("model config is empty, use default algorithm/filter parameters")
        return algorithm_parameters, {}

    config_text = model_item.get("config_text")
    config_path = model_item.get("config_path")

    if not config_text:
        log(f"model config is empty, path={config_path}, use default algorithm/filter parameters")
        return algorithm_parameters, {}

    try:
        config = parse_config_text(config_text)
    except Exception:
        import traceback

        log(f"parse model config failed, path={config_path}, use default algorithm/filter parameters")
        log(traceback.format_exc())
        return algorithm_parameters, {}

    if not isinstance(config, dict):
        log(f"model config is not a dict, path={config_path}, use default algorithm/filter parameters")
        return algorithm_parameters, {}

    extracted_algorithm_parameters = extract_algorithm_parameters(config)
    if extracted_algorithm_parameters:
        algorithm_parameters.update(extracted_algorithm_parameters)

    defect_filters = extract_defect_filters(config)

    log(
        "model config loaded"
        f", path={config_path}"
        f", algorithm_parameters={algorithm_parameters}"
        f", defect_filters={len(defect_filters)}"
    )

    return algorithm_parameters, defect_filters


def parse_config_text(config_text: str) -> Dict[str, Any]:
    # 优先按 JSON/YAML 解析；失败时兼容历史的简化配置格式。
    try:
        parsed = json.loads(config_text)
        return parsed or {}
    except Exception:
        pass

    try:
        import yaml

        parsed = yaml.safe_load(config_text)
        return parsed or {}
    except Exception:
        return parse_config_text_fallback(config_text)


def parse_config_text_fallback(config_text: str) -> Dict[str, Any]:
    # 轻量 fallback 只覆盖当前需要的 defect_filters 和 algorithm 段。
    config: Dict[str, Any] = {
        "defect_filters": [],
        "algorithm": [],
    }

    section = None
    current_item = None
    in_parameters = False

    for raw_line in config_text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue

        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if indent == 0 and stripped.endswith(":"):
            section = stripped[:-1]
            current_item = None
            in_parameters = False
            continue

        if section in ("defect_filters", "algorithm") and stripped.startswith("- "):
            current_item = {}
            config[section].append(current_item)
            in_parameters = False
            key_value = stripped[2:]
            if ":" in key_value:
                key, value = key_value.split(":", 1)
                current_item[key.strip()] = parse_scalar(value.strip())
            continue

        if current_item is None or ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        if section == "algorithm" and key == "parameters":
            current_item["parameters"] = {}
            in_parameters = True
            continue

        if section == "algorithm" and in_parameters:
            current_item.setdefault("parameters", {})[key] = parse_scalar(value)
            continue

        current_item[key] = parse_scalar(value)

    return config


def parse_scalar(value: str):
    if value == "":
        return {}

    text = value.strip().strip("\"'")
    lower = text.lower()

    if lower == "true":
        return True

    if lower == "false":
        return False

    try:
        if "." in text:
            return float(text)

        return int(text)
    except ValueError:
        return text


def extract_algorithm_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    algorithm = config.get("algorithm") or []

    if not isinstance(algorithm, list):
        return {}

    for step in algorithm:
        if not isinstance(step, dict):
            continue

        if step.get("enabled", True) is False:
            continue

        parameters = step.get("parameters") or {}

        if not isinstance(parameters, dict):
            continue

        allowed_keys = {
            # 白名单限制直接传给 YOLO predict 的参数，避免配置误传无关字段。
            "imgsz",
            "conf",
            "iou",
            "max_det",
            "classes",
            "agnostic_nms",
            "half",
            "device",
        }

        return {
            key: value
            for key, value in parameters.items()
            if key in allowed_keys
        }

    return {}


def extract_defect_filters(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    filters = config.get("defect_filters") or []
    output: Dict[str, Dict[str, Any]] = {}

    if not isinstance(filters, list):
        return output

    for item in filters:
        if not isinstance(item, dict):
            continue

        defect_code = item.get("defect_code")

        if defect_code is None:
            continue

        output[str(defect_code)] = item

    return output


def apply_defect_filters(
        boxes: List[Dict[str, Any]],
        defect_filters: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    # defect filter 是检测后的业务过滤，不影响 YOLO 原始推理结果。
    if not defect_filters:
        return boxes

    output = []

    for box in boxes:
        defect_filter = defect_filters.get(str(box.get("class_id")))

        if defect_filter is None:
            output.append(box)
            continue

        if defect_filter.get("is_enabled", True) is False:
            continue

        width = float(box["x2"]) - float(box["x1"])
        height = float(box["y2"]) - float(box["y1"])
        area = width * height
        confidence = float(box.get("confidence", 0.0))

        if width < float(defect_filter.get("min_width", 0) or 0):
            continue

        if height < float(defect_filter.get("min_height", 0) or 0):
            continue

        if area < float(defect_filter.get("min_area", 0) or 0):
            continue

        if confidence < float(defect_filter.get("custom_threshold", 0) or 0):
            continue

        output.append(box)

    return output
