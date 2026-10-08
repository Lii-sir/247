"""Explicit user calibration; never infer an allowed region from silver itself."""

import json
from dataclasses import dataclass
from pathlib import Path

from .geometry import Boundary


@dataclass(frozen=True)
class Calibration:
    template_path: Path
    boundary: Boundary

    def summary(self):
        return {"schema_version": 1, "template_path": str(self.template_path.resolve()),
                "boundary_mode": self.boundary.mode, "template_points": self.boundary.points}


def load_calibration(path) -> Calibration:
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema_version", 1) != 1:
        raise ValueError("不支持的标定文件格式")
    try:
        template = Path(data["template_path"]).expanduser()
        boundary = Boundary(data["template_points"], data.get("boundary_mode", "polygon"))
    except (KeyError, TypeError) as exc:
        raise ValueError("标定文件需要 template_path 和 template_points") from exc
    if not template.is_absolute():
        template = path.parent / template
    return Calibration(template.resolve(), boundary)


def save_calibration(path, calibration: Calibration, protected=()):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("标定文件必须保存为 .json")
    if path in {calibration.template_path.resolve(), *(Path(p).resolve() for p in protected)}:
        raise ValueError("不能覆盖输入图片、权重或其他受保护的输入")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.summary(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
