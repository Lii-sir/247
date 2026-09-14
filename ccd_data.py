"""扫描 CCD 数据并生成只读快照；不会创建、移动或修改原始数据。"""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
TEMP_EXTENSIONS = {".tmp", ".temp", ".part", ".partial", ".download", ".crdownload", ".aria2"}
DEFAULT_DATA_ROOTS = (
    Path(r"D:\datasets\20260909_ccd1-6_ok+v5ng"),
    Path(r"D:\datasets\20260909\_ccd1-6\_ok+v5ng"),
)


def resolve_data_root(path: str | Path | None = None) -> Path:
    """显式路径必须存在；未指定时才尝试已知的默认目录。"""
    if path is not None:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"数据集目录不存在或不是目录：{root}")
        return root
    for candidate in DEFAULT_DATA_ROOTS:
        if candidate.is_dir():
            return candidate.resolve()
    candidates = "、".join(str(candidate) for candidate in DEFAULT_DATA_ROOTS)
    raise FileNotFoundError(f"未找到默认数据集目录，请显式指定 --data-root。已检查：{candidates}")


def list_categories(root: str | Path) -> list[str]:
    """返回有 train 或 test 目录的相机名称；正在下载的空相机不参与。"""
    root = resolve_data_root(root)
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and ((child / "train").is_dir() or (child / "test").is_dir())
    )


def _skip(skipped: list[dict[str, Any]], path: Path, reason: str, detail: str, **extra: Any) -> None:
    """统一记录跳过原因，方便命令行统计和事后核查。"""
    skipped.append({"path": str(path.resolve()), "reason": reason, "detail": detail, **extra})


def _read_record(
    path: Path,
    label: int,
    defect_type: str,
    skipped: list[dict[str, Any]],
    cutoff_ns: int,
    verify_images: bool,
) -> dict[str, Any] | None:
    """检查文件稳定性和图片完整性，同时计算去重所需的内容哈希。"""
    if path.name.startswith(("~$", ".~")) or path.suffix.lower() in TEMP_EXTENSIONS:
        _skip(skipped, path, "temporary_file", "下载临时文件或编辑器临时文件")
        return None
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        _skip(skipped, path, "unsupported_extension", "不是支持的图片格式")
        return None
    try:
        before = path.stat()
        if before.st_mtime_ns > cutoff_ns:
            _skip(skipped, path, "too_recent", "修改时间过近，可能仍在下载")
            return None
        if verify_images:
            with Image.open(path) as image:
                image.verify()
        # verify 不保证像素数据均可解码，必须重新打开并完整加载。
        with Image.open(path) as image:
            if verify_images:
                image.load()
            width, height = image.size
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            _skip(skipped, path, "changed_during_scan", "检查期间文件大小或修改时间发生变化")
            return None
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        _skip(skipped, path, "unreadable_image", f"图片损坏或无法读取：{error}")
        return None
    return {
        "path": str(path.resolve()),
        "label": label,
        "defect_type": defect_type,
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "width": width,
        "height": height,
        "sha256": digest.hexdigest(),
    }


def _scan_directory(
    directory: Path,
    label: int,
    defect_type: str,
    skipped: list[dict[str, Any]],
    cutoff_ns: int,
    verify_images: bool,
) -> list[dict[str, Any]]:
    """按路径排序后递归扫描，让相同文件集合产生可复现的划分。"""
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file():
            continue
        record = _read_record(path, label, defect_type, skipped, cutoff_ns, verify_images)
        if record is not None:
            records.append(record)
    return records


def _deduplicate(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """保留扫描到的全部记录；仅保留该函数作为兼容入口。"""
    return train, test


def _find_duplicate_content(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> dict[str, Any]:
    """一次分组并报告全部字节内容相同的图片，不在这里删除任何记录。"""
    by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, records in (("train", train), ("test", test)):
        for record in records:
            by_digest[record["sha256"]].append({
                "path": record["path"],
                "split": split,
                "label": record["label"],
                "defect_type": record["defect_type"],
            })

    groups = []
    for digest, files in sorted(by_digest.items()):
        if len(files) < 2:
            continue
        files.sort(key=lambda item: item["path"].casefold())
        splits = sorted({item["split"] for item in files})
        labels = sorted({item["label"] for item in files})
        groups.append({
            "sha256": digest,
            "file_count": len(files),
            "extra_copy_count": len(files) - 1,
            "splits": splits,
            "cross_train_test": splits == ["test", "train"],
            "label_conflict": len(labels) > 1,
            "files": files,
        })

    return {
        "hash_type": "sha256_file_bytes",
        "group_count": len(groups),
        "file_count": sum(group["file_count"] for group in groups),
        "extra_copy_count": sum(group["extra_copy_count"] for group in groups),
        "cross_train_test_group_count": sum(group["cross_train_test"] for group in groups),
        "label_conflict_group_count": sum(group["label_conflict"] for group in groups),
        "groups": groups,
    }


def prepare_manifest(
    root: Path,
    category: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    min_age_seconds: float = 60.0,
    verify_images: bool = True,
) -> dict[str, Any]:
    """生成快照：仅 train/good 参与训练与正常验证，test 只用于最终测试。

    验证集至少保留两张不同的良品图片，供 EfficientAD 的分位数校准使用；
    训练集也至少保留两张。SHA256 仅作为记录字段，不据此剔除图片。
    文件年龄和扫描前后状态检查只能降低下载干扰；快照生成后应保留源文件。
    没有提供像素标注时仅记录图像标签，不构造虚假的缺陷 mask。
    """
    if not math.isfinite(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError("val_ratio 必须大于 0 且小于 1")
    if not math.isfinite(min_age_seconds) or min_age_seconds < 0:
        raise ValueError("min_age_seconds 必须是非负有限数")
    root = resolve_data_root(root)
    if not category or category in {".", ".."} or Path(category).name != category or "/" in category or "\\" in category:
        raise ValueError("category 必须是数据集根目录下的相机目录名称")
    category_root = root / category
    train_good = category_root / "train" / "good"
    if not train_good.is_dir():
        raise FileNotFoundError(f"训练良品目录不存在：{train_good}")

    skipped: list[dict[str, Any]] = []
    cutoff_ns = time.time_ns() - int(min_age_seconds * 1_000_000_000)
    train = _scan_directory(train_good, 0, "good", skipped, cutoff_ns, verify_images)
    test: list[dict[str, Any]] = []
    test_root = category_root / "test"
    if test_root.is_dir():
        for defect_directory in sorted(test_root.iterdir(), key=lambda item: item.name.casefold()):
            if defect_directory.is_dir():
                label = 0 if defect_directory.name == "good" else 1
                test.extend(_scan_directory(defect_directory, label, defect_directory.name, skipped, cutoff_ns, verify_images))
            elif defect_directory.is_file():
                _skip(skipped, defect_directory, "missing_label_directory", "测试图片必须位于 test/good 或 test/缺陷类别 子目录")
    train, test = _deduplicate(train, test, skipped)
    duplicate_report = _find_duplicate_content(train, test)
    if len(train) < 4:
        raise ValueError(
            f"{category} 可用的训练良品只有 {len(train)} 张，至少需要 4 张"
            f"（训练、验证各至少 2 张）；已跳过 {len(skipped)} 个文件。"
            "请等待下载完成，或检查文件年龄和损坏图片。"
        )
    random.Random(seed).shuffle(train)
    val_count = min(len(train) - 2, max(2, round(len(train) * val_ratio)))
    val, train = train[:val_count], train[val_count:]
    summary = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "test_good": sum(record["label"] == 0 for record in test),
        "test_anomaly": sum(record["label"] == 1 for record in test),
        "test_defect_types": dict(sorted(Counter(record["defect_type"] for record in test).items())),
        "skipped_count": len(skipped),
        "skipped_by_reason": dict(sorted(Counter(record["reason"] for record in skipped).items())),
        "duplicate_groups": duplicate_report["group_count"],
        "duplicate_files": duplicate_report["file_count"],
        "duplicate_extra_copies": duplicate_report["extra_copy_count"],
        "duplicate_cross_train_test_groups": duplicate_report["cross_train_test_group_count"],
        "duplicate_label_conflict_groups": duplicate_report["label_conflict_group_count"],
    }
    return {
        "schema_version": 1,
        "root": str(root),
        "category": category,
        "seed": seed,
        "val_ratio": val_ratio,
        "min_age_seconds": min_age_seconds,
        "verify_images": verify_images,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train": train,
        "val": val,
        "test": test,
        "skipped": skipped,
        "duplicate_report": duplicate_report,
        "summary": summary,
    }
