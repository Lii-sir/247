"""人工配对标定的 JSON 存取；不在磁盘上存放 Python 对象。"""

import json
import os
from pathlib import Path

from part_segmentation.image_io import read_image

from .geometry import fit_template_mapping
from .models import Calibration, MappingSettings


def calibration_document(calibration: Calibration, relative_to: Path | None = None) -> dict:
    def path_text(path):
        path = Path(path).resolve()
        if relative_to is not None:
            try:
                return Path(os.path.relpath(path, relative_to)).as_posix()
            except ValueError:  # Windows 不同盘符，保留绝对路径。
                pass
        return str(path)

    document = {"schema_version": 1, "coordinate_system": "decoded image pixels; x right, y down",
            "template_a": path_text(calibration.template_a), "template_b": path_text(calibration.template_b),
            "points_a": [list(point) for point in calibration.points_a],
            "points_b": [list(point) for point in calibration.points_b]}
    if calibration.radii_a or calibration.radii_b:
        document["radii_a"] = list(calibration.radii_a or (None,) * len(calibration.points_a))
        document["radii_b"] = list(calibration.radii_b or (None,) * len(calibration.points_b))
    return document


def validate_calibration(calibration: Calibration, settings: MappingSettings | None = None):
    a, b = read_image(calibration.template_a), read_image(calibration.template_b)
    return fit_template_mapping(calibration.points_a, calibration.points_b, a.shape, b.shape, settings)


def save_calibration(path: Path, calibration: Calibration, settings: MappingSettings | None = None) -> None:
    validate_calibration(calibration, settings)
    path = Path(path).resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("标定文件必须使用 .json 扩展名")
    if path in (calibration.template_a.resolve(), calibration.template_b.resolve()):
        raise ValueError("不能覆盖模板图片")
    text = json.dumps(calibration_document(calibration, path.parent), ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_calibration(path: Path) -> Calibration:
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("不支持的标定文件版本")

    def resolve(key):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"标定缺少有效的 {key} 路径")
        value = Path(value).expanduser()
        return (path.parent / value).resolve() if not value.is_absolute() else value.resolve()

    try:
        points_a = tuple(tuple(float(v) for v in point) for point in data["points_a"])
        points_b = tuple(tuple(float(v) for v in point) for point in data["points_b"])
        radii = []
        for key, points in (("radii_a", points_a), ("radii_b", points_b)):
            if key not in data:
                radii.append(())
                continue
            values = data[key]
            if not isinstance(values, list) or len(values) != len(points):
                raise ValueError("圆半径列表必须与点列表等长")
            radii.append(tuple(None if value is None else float(value) for value in values))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("标定中的对应点格式无效") from exc
    calibration = Calibration(resolve("template_a"), resolve("template_b"), points_a, points_b, *radii)
    # 加载只做格式/范围验证，RANSAC 使用运行时的 MappingSettings。
    from .geometry import validate_points
    if len(points_a) != len(points_b):
        raise ValueError("A/B 模板点数不一致")
    validate_points(points_a, read_image(calibration.template_a).shape, "A 模板")
    validate_points(points_b, read_image(calibration.template_b).shape, "B 模板")
    return calibration

