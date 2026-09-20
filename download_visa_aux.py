"""Download VisA and prepare its normal training images as an ImageFolder.

The EfficientAD training script keeps the historical ``--imagenette-dir``
argument name, but the directory is consumed by torchvision's ImageFolder.
This utility therefore prepares a VisA directory that can be passed to that
argument directly.

Only ``train`` samples labelled ``normal``/``good`` are copied.  The archive
is read without extracting unselected members, and archive paths are never
used as output paths.  This avoids both unnecessary temporary disk usage and
path traversal through a malicious archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
DEFAULT_ARCHIVE = Path("assets") / "VisA_20220922.tar"
NORMAL_LABELS = frozenset({"normal", "good"})
NORMAL_DIR_NAMES = frozenset({"normal", "good"})
TRAIN_SPLITS = frozenset({"train", "training"})
IMAGE_SUFFIXES = frozenset(
    {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)
PATH_COLUMN_NAMES = ("imagepath", "image", "path", "imgpath", "filepath", "file")
SPLIT_COLUMN_NAMES = ("split", "partition", "set")
LABEL_COLUMN_NAMES = ("label", "classlabel", "anomalylabel", "type")
CATEGORY_COLUMN_NAMES = ("object", "category", "classname", "objectname", "product")


@dataclass(frozen=True)
class SelectedImage:
    """One archive member selected for the auxiliary ImageFolder."""

    category: str
    member_key: tuple[str, ...]


@dataclass(frozen=True)
class SplitTable:
    member_key: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    path_column: str
    split_column: str
    label_column: str | None
    category_column: str | None


def _normalise_field_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _normalise_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().casefold()


def _normalise_member_name(name: str) -> tuple[str, ...] | None:
    """Return a safe POSIX member path, or None for an unsafe path."""

    normalised = name.replace("\\", "/")
    path = PurePosixPath(normalised)
    if path.is_absolute():
        return None

    parts = tuple(part for part in path.parts if part not in {"", "."})
    if not parts or any(part == ".." for part in parts):
        return None
    # A drive-qualified first component can be interpreted specially on
    # Windows even when PurePosixPath considers it relative.
    if ":" in parts[0]:
        return None
    return parts


def _is_image_member(member_key: tuple[str, ...]) -> bool:
    return len(member_key) > 0 and Path(member_key[-1]).suffix.casefold() in IMAGE_SUFFIXES


def _is_under(path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(path) > len(prefix) and path[: len(prefix)] == prefix


def _is_normal_path(path: tuple[str, ...]) -> bool:
    directories = {part.casefold() for part in path[:-1]}
    return bool(directories & NORMAL_DIR_NAMES) and "anomaly" not in directories


def _member_string(member_key: tuple[str, ...]) -> str:
    return "/".join(member_key)


def _build_member_index(
    archive: tarfile.TarFile,
) -> tuple[dict[tuple[str, ...], tarfile.TarInfo], dict[str, tuple[str, ...]]]:
    """Index regular archive files using only safe, normalised names."""

    members: dict[tuple[str, ...], tarfile.TarInfo] = {}
    lower_names: dict[str, tuple[str, ...]] = {}
    for member in archive.getmembers():
        if not member.isfile():
            continue
        member_key = _normalise_member_name(member.name)
        if member_key is None:
            continue
        members.setdefault(member_key, member)
        lower_names.setdefault(_member_string(member_key).casefold(), member_key)
    if not members:
        raise RuntimeError("VisA 归档中没有可读取的普通文件。")
    return members, lower_names


def _is_normal_image_under(
    member_key: tuple[str, ...], prefix: tuple[str, ...]
) -> bool:
    """Recognise normal image samples in either supported VisA layout."""

    if not _is_under(member_key, prefix) or not _is_image_member(member_key):
        return False
    relative = member_key[len(prefix) :]
    if len(relative) >= 5 and tuple(part.casefold() for part in relative[1:3]) == ("data", "images"):
        # Official layout: <category>/Data/Images/Normal/<file>.
        return _is_normal_path(relative[3:])
    if prefix and prefix[-1].casefold() == "images" and len(relative) >= 3:
        # Alternate layout: Data/Images/<category>/Normal/<file>.
        return relative[0].casefold() not in NORMAL_DIR_NAMES and _is_normal_path(relative[1:])
    return False


def _find_images_prefix(member_keys: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    """Find the root above categories, including all category-first image trees."""

    candidates: set[tuple[str, ...]] = set()
    file_keys = tuple(member_keys)
    for member_key in file_keys:
        for index, part in enumerate(member_key[:-1]):
            if part.casefold() == "images":
                candidates.add(member_key[: index + 1])
                if index >= 2 and member_key[index - 1].casefold() == "data":
                    candidates.add(member_key[: index - 2])

    scored: list[tuple[tuple[int, int], tuple[str, ...]]] = []
    for candidate in candidates:
        normal_count = sum(
            _is_normal_image_under(member_key, candidate) for member_key in file_keys
        )
        if not normal_count:
            continue
        scored.append(((normal_count, len(candidate)), candidate))

    if not scored:
        raise RuntimeError("无法在 VisA 归档中找到包含 Normal 图片的 Data/Images 目录。")
    return max(scored)[1]


def _lookup_member(
    member_key: tuple[str, ...],
    members: dict[tuple[str, ...], tarfile.TarInfo],
    lower_names: dict[str, tuple[str, ...]],
) -> tuple[str, ...] | None:
    if member_key in members:
        return member_key
    return lower_names.get(_member_string(member_key).casefold())


def _resolve_image_member(
    raw_path: str,
    images_prefix: tuple[str, ...],
    members: dict[tuple[str, ...], tarfile.TarInfo],
    lower_names: dict[str, tuple[str, ...]],
) -> tuple[str, ...] | None:
    """Resolve a CSV path against common VisA archive layouts."""

    raw_parts = _normalise_member_name(raw_path.strip())
    if raw_parts is None:
        return None

    # CSV paths can be <object>/Data/Images/... or Data/Images/<object>/...,
    # while the tar can have an additional top-level VisA_20220922/ directory.
    candidates: list[tuple[str, ...]] = [raw_parts]
    for start in range(len(raw_parts)):
        candidates.append(images_prefix + raw_parts[start:])
    for candidate in candidates:
        resolved = _lookup_member(candidate, members, lower_names)
        if resolved is not None and _is_under(resolved, images_prefix):
            return resolved
    return None


def _find_column(fieldnames: Iterable[str] | None, aliases: Iterable[str]) -> str | None:
    if fieldnames is None:
        return None
    by_normalised_name = {
        _normalise_field_name(fieldname): fieldname
        for fieldname in fieldnames
        if fieldname is not None
    }
    for alias in aliases:
        result = by_normalised_name.get(_normalise_field_name(alias))
        if result is not None:
            return result
    return None


def _read_csv_table(archive: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[list[str], list[dict[str, str]]]:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"无法读取 split CSV：{member.name}")
    with extracted:
        text = extracted.read().decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        return [], []
    return reader.fieldnames, list(reader)


def _split_csv_candidates(
    member_keys: Iterable[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    csv_keys = [
        key
        for key in member_keys
        if len(key) > 0 and Path(key[-1]).suffix.casefold() == ".csv"
    ]

    def priority(key: tuple[str, ...]) -> tuple[int, int, str]:
        filename = key[-1].casefold()
        parent_is_split = int(any(part.casefold() == "split_csv" for part in key[:-1]))
        if filename == "1cls.csv":
            filename_priority = 0
        elif filename == "2cls.csv":
            filename_priority = 1
        elif "split" in filename:
            filename_priority = 2
        else:
            filename_priority = 3
        return (filename_priority, -parent_is_split, _member_string(key).casefold())

    return sorted(csv_keys, key=priority)


def _load_split_table(
    archive: tarfile.TarFile,
    members: dict[tuple[str, ...], tarfile.TarInfo],
) -> SplitTable | None:
    for member_key in _split_csv_candidates(members):
        fieldnames, rows = _read_csv_table(archive, members[member_key])
        path_column = _find_column(fieldnames, PATH_COLUMN_NAMES)
        split_column = _find_column(fieldnames, SPLIT_COLUMN_NAMES)
        if path_column is None or split_column is None:
            continue
        return SplitTable(
            member_key=member_key,
            rows=tuple(rows),
            path_column=path_column,
            split_column=split_column,
            label_column=_find_column(fieldnames, LABEL_COLUMN_NAMES),
            category_column=_find_column(fieldnames, CATEGORY_COLUMN_NAMES),
        )
    return None


def _validate_category(category: str) -> str:
    category = category.strip()
    invalid_characters = set('<>:"/\\|?*')
    if not category or category in {".", ".."} or invalid_characters.intersection(category):
        raise RuntimeError(f"VisA 类别名无法作为 Windows 文件夹名：{category!r}")
    return category


def _select_from_split(
    table: SplitTable,
    images_prefix: tuple[str, ...],
    members: dict[tuple[str, ...], tarfile.TarInfo],
    lower_names: dict[str, tuple[str, ...]],
) -> list[SelectedImage]:
    selected: list[SelectedImage] = []
    seen_members: set[tuple[str, ...]] = set()
    skipped_missing = 0
    for row in table.rows:
        if _normalise_value(row.get(table.split_column)) not in TRAIN_SPLITS:
            continue

        raw_path = row.get(table.path_column, "")
        member_key = _resolve_image_member(raw_path, images_prefix, members, lower_names)
        if member_key is None:
            skipped_missing += 1
            continue
        relative = member_key[len(images_prefix) :]
        if not _is_normal_image_under(member_key, images_prefix):
            continue

        label = _normalise_value(row.get(table.label_column)) if table.label_column else ""
        if label and label not in NORMAL_LABELS:
            continue
        if member_key in seen_members:
            continue

        category_value = row.get(table.category_column, "") if table.category_column else ""
        category = _validate_category(category_value or relative[0])
        selected.append(SelectedImage(category=category, member_key=member_key))
        seen_members.add(member_key)

    if skipped_missing:
        print(
            f"警告：split CSV 中有 {skipped_missing} 条图片路径无法在归档中解析，已跳过。",
            file=sys.stderr,
        )
    return selected


def _select_from_normal_directories(
    images_prefix: tuple[str, ...],
    members: dict[tuple[str, ...], tarfile.TarInfo],
) -> list[SelectedImage]:
    """Fallback for archives that omit split CSV files."""

    selected: list[SelectedImage] = []
    for member_key in sorted(members):
        if not _is_normal_image_under(member_key, images_prefix):
            continue
        relative = member_key[len(images_prefix) :]
        selected.append(SelectedImage(category=_validate_category(relative[0]), member_key=member_key))
    return selected


def _filter_selected(
    selected: Iterable[SelectedImage],
    excluded_categories: Iterable[str],
    max_per_class: int | None,
) -> list[SelectedImage]:
    excluded = {category.casefold() for category in excluded_categories}
    grouped: dict[str, list[SelectedImage]] = defaultdict(list)
    for image in selected:
        key = image.category.casefold()
        if key in excluded:
            continue
        grouped[key].append(image)

    result: list[SelectedImage] = []
    for key in sorted(grouped):
        images = sorted(grouped[key], key=lambda item: _member_string(item.member_key).casefold())
        if max_per_class is not None:
            images = images[:max_per_class]
        result.extend(images)
    if not result:
        raise RuntimeError("筛选后没有可用的 VisA 训练正常图片，请检查归档、排除类别和参数。")
    return result


def _prepare_output(output: Path, clean: bool) -> None:
    output = output.resolve()
    if output.exists():
        if not output.is_dir():
            raise RuntimeError(f"输出路径不是目录：{output}")
        if any(output.iterdir()):
            if not clean:
                raise RuntimeError(
                    f"输出目录不为空：{output}；如需重新整理，请确认后添加 --clean。"
                )
            if output == Path.cwd().resolve() or len(output.parts) <= 2:
                raise RuntimeError("拒绝使用 --clean 删除过于宽泛的输出目录。")
            shutil.rmtree(output)
        elif clean:
            output.rmdir()
    output.mkdir(parents=True, exist_ok=True)


def _download_archive(url: str, archive: Path, force: bool) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.is_file() and archive.stat().st_size > 0 and not force:
        print(f"复用已有 VisA 归档：{archive}")
        return

    temporary = archive.with_name(f"{archive.name}.part")
    print(f"下载 VisA 归档：{url}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "visa-aux-downloader/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as target:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 / total
                    print(f"\r已下载 {downloaded / 1024**3:.2f}/{total / 1024**3:.2f} GiB ({percent:5.1f}%)", end="", flush=True)
            if total and downloaded != total:
                raise RuntimeError(f"下载大小不完整：{downloaded} / {total} bytes")
        print()
        temporary.replace(archive)
    except (OSError, urllib.error.URLError, RuntimeError) as error:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(
            "VisA 下载失败。可以先手动下载归档，再使用 --archive 指定本地 .tar 文件。"
            f" 原因：{error}"
        ) from error


def _safe_output_filename(member_key: tuple[str, ...]) -> str:
    filename = member_key[-1]
    if filename in {"", ".", ".."} or any(character in filename for character in '<>:"/\\|?*'):
        raise RuntimeError(f"VisA 图片文件名无法写入 Windows 文件系统：{filename!r}")
    return filename


def _copy_selected_images(
    archive: tarfile.TarFile,
    members: dict[tuple[str, ...], tarfile.TarInfo],
    selected: Iterable[SelectedImage],
    output: Path,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for image in selected:
        destination_dir = output / image.category
        destination_dir.mkdir(parents=True, exist_ok=True)
        filename = _safe_output_filename(image.member_key)
        destination = destination_dir / filename
        if destination.exists():
            source_tag = _member_string(image.member_key).encode("utf-8")
            digest = hashlib.sha1(source_tag).hexdigest()[:10]
            source_path = Path(filename)
            destination = destination_dir / f"{source_path.stem}__{digest}{source_path.suffix}"
        if destination.exists():
            raise RuntimeError(f"输出文件名冲突：{destination}")

        extracted = archive.extractfile(members[image.member_key])
        if extracted is None:
            raise RuntimeError(f"无法读取归档成员：{_member_string(image.member_key)}")
        with extracted, destination.open("wb") as target:
            shutil.copyfileobj(extracted, target, length=1024 * 1024)
        counts[image.category] += 1
    return dict(sorted(counts.items()))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="下载 VisA，并整理 train/normal 图片为 EfficientAD 辅助 ImageFolder。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出 ImageFolder 目录，例如 D:/datasets/visa_aux",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"VisA tar 保存或读取路径，默认 {DEFAULT_ARCHIVE}",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="VisA tar 下载地址")
    parser.add_argument(
        "--exclude-class",
        action="append",
        default=[],
        metavar="CLASS",
        help="排除一个 VisA 类别；可重复指定",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        help="每个类别最多保留多少张图片；默认保留全部",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="整理前删除已有的非空输出目录；请确认路径后使用",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="忽略已有归档并重新下载",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_per_class is not None and args.max_per_class < 1:
        raise ValueError("--max-per-class 必须为正整数。")

    output = args.output.expanduser().resolve()
    archive_path = args.archive.expanduser().resolve()
    if output == archive_path or output in archive_path.parents:
        raise ValueError("--archive 不能放在 --output 目录内，请使用输出目录外的归档路径。")

    _download_archive(args.url, archive_path, args.redownload)
    print(f"读取 VisA 归档：{archive_path}")
    with tarfile.open(archive_path, mode="r:*") as archive:
        members, lower_names = _build_member_index(archive)
        images_prefix = _find_images_prefix(members)
        split_table = _load_split_table(archive, members)
        if split_table is not None:
            selected = _select_from_split(split_table, images_prefix, members, lower_names)
            split_csv = _member_string(split_table.member_key)
            selection_mode = "split_csv_train_normal"
        else:
            selected = _select_from_normal_directories(images_prefix, members)
            split_csv = None
            selection_mode = "normal_directories_fallback"
            print(
                "警告：未找到可解析的 split CSV，将使用所有 Normal/Good 目录图片；"
                "请确认归档结构。",
                file=sys.stderr,
            )

        selected = _filter_selected(selected, args.exclude_class, args.max_per_class)
        _prepare_output(output, args.clean)
        counts = _copy_selected_images(archive, members, selected, output)

    manifest = {
        "dataset": "VisA",
        "format": "torchvision.datasets.ImageFolder",
        "source": {
            "url": args.url,
            "archive": str(archive_path),
            "archive_size_bytes": archive_path.stat().st_size,
        },
        "selection": {
            "mode": selection_mode,
            "images_root": _member_string(images_prefix) or ".",
            "split_csv": split_csv,
            "splits": ["train"],
            "labels": ["normal", "good"],
            "excluded_classes": list(args.exclude_class),
            "max_per_class": args.max_per_class,
        },
        "classes": counts,
        "total_images": sum(counts.values()),
    }
    temporary_manifest = output / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(output / "manifest.json")

    print(f"整理完成：{output}")
    print(f"类别数：{len(counts)}；图片数：{sum(counts.values())}")
    print(f"可直接用于训练：--imagenette-dir \"{output}\"")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, tarfile.TarError) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1) from error
