"""图片发现、读写；支持 Windows 中文路径，不依赖推理和界面。"""

from pathlib import Path

import cv2 as cv
import numpy as np

IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def collect_images(source: str | Path, recursive: bool = False) -> list[Path]:
    source = Path(source).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不支持的图片格式：{source.suffix}")
        return [source]
    if not source.is_dir():
        raise ValueError(f"图片或文件夹不存在：{source}")
    entries = source.rglob("*") if recursive else source.iterdir()
    images = sorted(p for p in entries if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"文件夹内没有支持的图片：{source}")
    return images


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv.imdecode(data, cv.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"无法解码图片：{path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"不支持的输出格式：{path.suffix}")
    ok, data = cv.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"图片编码失败：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(path)

