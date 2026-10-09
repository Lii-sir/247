"""同步推理源码到独立交付目录，不移动原文件，不复制生产权重或训练数据。"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_DIR / "delivery" / "efficientad"
RUNTIME_FILES = (
    "ccd_efficientad/__init__.py", "ccd_efficientad/paths.py",
    "ccd_efficientad/inference.py", "ccd_efficientad/cli.py", "ccd_efficientad/data.py",
    "ccd_efficientad/localization.py", "ccd_efficientad/report.py", "ccd_efficientad/mask.py",
    "ccd_efficientad/models/__init__.py", "ccd_efficientad/models/backbones.py",
    "ccd_efficientad/models/lightning_model.py", "ccd_efficientad/models/torch_model.py",
    "ccd_efficientad/models/resnet_autoencoder.py", "ccd_efficientad/models/README.md",
    "ccd_efficientad/models/COPY_INFO.md",
)
ENVIRONMENT_FILES = ("pyproject.toml", "uv.lock", ".python-version")


def build() -> Path:
    copies = [(relative, "src/" + relative) for relative in RUNTIME_FILES]
    copies += [(relative, relative) for relative in ENVIRONMENT_FILES]
    for source, _ in copies:
        if not (PROJECT_DIR / source).is_file():
            raise FileNotFoundError(f"原项目缺少文件：{PROJECT_DIR / source}")
    # 所有写入固定在本项目的 delivery/efficientad；不用任何递归删除。
    destination = DESTINATION.resolve()
    if not destination.is_relative_to(PROJECT_DIR.resolve()) or DESTINATION.is_symlink():
        raise ValueError("交付目录必须位于当前项目内，且不能是符号链接。")
    for _, relative in copies:
        if not (destination / relative).resolve().is_relative_to(destination):
            raise ValueError(f"交付文件不能通过链接指向目录外：{relative}")
    records = []
    for source_relative, output_relative in copies:
        source, target = PROJECT_DIR / source_relative, destination / output_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append({
            "source": source_relative, "destination": output_relative,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        })
    (destination / "source_manifest.json").write_text(json.dumps({
        "description": "原项目源码与环境文件的 SHA256 快照；不包含生产权重、mask 或训练数据。",
        "files": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已同步 {len(records)} 个文件：{destination}")
    print("现有 weights/、masks/、configs/、examples/、results/ 不受影响。")
    return destination


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    build()
