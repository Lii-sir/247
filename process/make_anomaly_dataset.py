#!/usr/bin/env python3
"""Create a train/good, test/good and synthetic test/ng dataset.

The input directory is treated as a directory of normal images.  Every source
image is assigned to exactly one of the three output groups.  Images assigned
to ``test/ng`` are never copied as good images; instead, a defect is rendered
onto them and the rendered image is written to ``test/ng``.

Example:
    uv run python make_anomaly_dataset.py \
        --input-dir D:/data/normal \
        --output-dir D:/data/ccd_dataset \
        --train-ratio 0.70 \
        --test-good-ratio 0.15 \
        --test-ng-ratio 0.15 \
        --defect-types occlusion,dirt \
        --seed 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
OUTPUT_DIRS = (Path("train") / "good", Path("test") / "good", Path("test") / "ng")
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class SplitCounts:
    train_good: int
    test_good: int
    test_ng: int


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将一个正常图像文件夹随机划分为 train/good、test/good、test/ng；"
            "test/ng 使用遮挡或合成脏污生成。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="输入正常图像文件夹。")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出数据集根目录。")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="分配到 train/good 的比例。")
    parser.add_argument("--test-good-ratio", type=float, default=0.15, help="分配到 test/good 的比例。")
    parser.add_argument("--test-ng-ratio", type=float, default=0.15, help="分配到 test/ng 的源图比例。")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子。")
    parser.add_argument("--defect-types", default="occlusion,dirt", help="可选：occlusion,dirt，逗号分隔。")
    parser.add_argument("--min-defects", type=int, default=1, help="每张 NG 图最少叠加的缺陷数量。")
    parser.add_argument("--max-defects", type=int, default=2, help="每张 NG 图最多叠加的缺陷数量。")
    parser.add_argument("--min-size", type=float, default=0.04, help="单个缺陷相对短边的最小尺寸比例。")
    parser.add_argument("--max-size", type=float, default=0.22, help="单个缺陷相对短边的最大尺寸比例。")
    parser.add_argument("--occlusion-opacity", type=float, default=0.82, help="遮挡缺陷的不透明度，范围 0 到 1。")
    parser.add_argument("--dirt-dir", type=Path, default=None, help="可选的脏污贴图文件夹；未提供时使用合成脏污。")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="是否递归读取输入文件夹。")
    parser.add_argument("--overwrite", action="store_true", help="允许清空本脚本生成的输出子目录后重新生成。")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.input_dir.is_dir():
        raise ValueError(f"输入文件夹不存在：{args.input_dir}")
    if args.output_dir.resolve() == args.input_dir.resolve():
        raise ValueError("output-dir 不能与 input-dir 相同。")
    try:
        args.output_dir.resolve().relative_to(args.input_dir.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output-dir 不能放在 input-dir 内，否则会把已生成文件再次读入。")
    if args.input_dir.resolve().is_relative_to(args.output_dir.resolve()):
        raise ValueError("output-dir 不能是 input-dir 的上级目录。")
    if args.dirt_dir is not None:
        dirt = args.dirt_dir.resolve()
        output = args.output_dir.resolve()
        if dirt.is_relative_to(output) or output.is_relative_to(dirt):
            raise ValueError("dirt-dir 与 output-dir 不得相互包含。")

    ratios = (args.train_ratio, args.test_good_ratio, args.test_ng_ratio)
    if any(not math.isfinite(r) or r < 0 for r in ratios):
        raise ValueError("三个比例都必须大于等于 0。")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train/test-good/test-ng 三个比例之和必须等于 1。")
    if sum(ratios) <= 0:
        raise ValueError("至少需要一个非零比例。")
    if args.min_defects < 1 or args.max_defects < args.min_defects:
        raise ValueError("需要满足 1 <= min-defects <= max-defects。")
    if not 0 < args.min_size <= args.max_size <= 1:
        raise ValueError("需要满足 0 < min-size <= max-size <= 1。")
    if not 0 < args.occlusion_opacity <= 1:
        raise ValueError("occlusion-opacity 必须大于 0 且不超过 1。")

    defect_types = tuple(item.strip().lower() for item in args.defect_types.split(",") if item.strip())
    allowed = {"occlusion", "dirt"}
    if not defect_types or any(item not in allowed for item in defect_types):
        raise ValueError("defect-types 只能包含 occlusion 和 dirt。")
    if args.dirt_dir is not None and not args.dirt_dir.is_dir():
        raise ValueError(f"dirt-dir 不存在：{args.dirt_dir}")
    return defect_types


def list_images(root: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix().lower(),
    )
    if not paths:
        raise ValueError(f"输入文件夹中没有找到支持的图像：{root}")
    return paths


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> SplitCounts:
    raw = [total * ratio / sum(ratios) for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        (index for index, ratio in enumerate(ratios) if ratio > 0),
        key=lambda index: raw[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return SplitCounts(*counts)


def split_images(paths: list[Path], ratios: tuple[float, float, float], seed: int) -> dict[str, list[Path]]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    counts = allocate_counts(len(shuffled), ratios)
    first = counts.train_good
    second = first + counts.test_good
    return {
        "train/good": shuffled[:first],
        "test/good": shuffled[first:second],
        "test/ng": shuffled[second:],
    }


def safe_output_name(source: Path, input_root: Path) -> str:
    relative = source.relative_to(input_root).as_posix()
    stem = Path(relative).with_suffix("").as_posix()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem.replace("/", "__"))[:80]
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()
    suffix = source.suffix.lower()
    return f"{stem}__{digest}{suffix}"


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    # Validate ownership before any deletion; never follow a directory link.
    if output_dir.exists() and any(output_dir.iterdir()) and overwrite:
        manifest_path = output_dir / MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("拒绝覆盖：缺少本脚本的有效 manifest.json。")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("generator") != "make_anomaly_dataset":
            raise ValueError("拒绝覆盖：该目录不是本脚本生成的数据集。")
        expected = {MANIFEST_NAME} | {row["output"] for row in old["items"]}
        actual = set()
        allowed_dirs = {"train", "train/good", "test", "test/good", "test/ng"}
        for path in output_dir.rglob("*"):
            if path.is_symlink() or path.is_junction():
                raise ValueError(f"拒绝覆盖链接：{path}")
            relative = path.relative_to(output_dir).as_posix()
            if path.is_dir() and relative not in allowed_dirs:
                raise ValueError(f"拒绝覆盖额外目录：{path}")
            if path.is_file():
                actual.add(relative)
        if actual != expected:
            raise ValueError("拒绝覆盖：目录文件与 manifest 不一致，请使用新的输出目录。")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {path.parts[0] for path in OUTPUT_DIRS} | {MANIFEST_NAME}
    existing = {path.name for path in output_dir.iterdir()}
    unexpected = existing - generated
    if unexpected and overwrite:
        raise ValueError(
            f"拒绝覆盖：输出目录包含非本脚本生成的内容：{', '.join(sorted(unexpected))}"
        )
    if existing and not overwrite:
        raise FileExistsError(
            f"输出目录非空：{output_dir}；如确认重新生成，请添加 --overwrite。"
        )
    if overwrite:
        for relative in OUTPUT_DIRS:
            target = output_dir / relative
            if target.exists():
                if not target.resolve().is_relative_to(output_dir.resolve()):
                    raise ValueError(f"输出路径越界：{target}")
                shutil.rmtree(target)
        manifest = output_dir / MANIFEST_NAME
        if manifest.exists():
            manifest.unlink()

    for relative in OUTPUT_DIRS:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"不支持多帧图像：{path}")
        if min(image.size) < 2:
            raise ValueError(f"图像宽高至少为 2：{path}")
        return ImageOps.exif_transpose(image).convert("RGB")


def unique_images(paths: list[Path]) -> tuple[list[Path], list[dict]]:
    """Deduplicate decoded RGB pixels, including renamed lossless copies."""
    seen = {}
    unique, duplicates = [], []
    for path in paths:
        image = load_rgb(path)
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest in seen:
            duplicates.append({"skipped": str(path), "kept": str(seen[digest])})
        else:
            seen[digest] = path
            unique.append(path)
    return unique, duplicates


def random_box(width: int, height: int, rng: random.Random, min_size: float, max_size: float) -> tuple[int, int, int, int]:
    short_side = max(2, min(width, height))
    min_dim = max(2, int(short_side * min_size))
    max_dim = max(min_dim, int(short_side * max_size))
    defect_width = rng.randint(min_dim, min(max_dim, width))
    defect_height = rng.randint(min_dim, min(max_dim, height))
    left = rng.randint(0, max(0, width - defect_width))
    top = rng.randint(0, max(0, height - defect_height))
    return left, top, left + defect_width, top + defect_height


def apply_occlusion(image: Image.Image, rng: random.Random, min_size: float, max_size: float, opacity: float) -> Image.Image:
    width, height = image.size
    box = random_box(width, height, rng, min_size, max_size)
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    shape = rng.choice(("rectangle", "ellipse", "polygon"))
    if shape == "rectangle":
        draw.rectangle(box, fill=255)
    elif shape == "ellipse":
        draw.ellipse(box, fill=255)
    else:
        left, top, right, bottom = box
        points = [
            (left + rng.randint(0, max(0, right - left)), top + rng.randint(0, max(0, bottom - top))),
            (left + rng.randint(0, max(0, right - left)), bottom),
            (right, top + rng.randint(0, max(0, bottom - top))),
            (right - rng.randint(0, max(0, right - left)), top),
        ]
        draw.polygon(points, fill=255)
    if min(width, height) >= 32:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(0.5, min(width, height) * 0.003)))
    mask = mask.point(lambda value: int(value * opacity))

    palette = (
        (8, 8, 8),
        (35, 35, 35),
        (90, 90, 90),
        (230, 230, 230),
        (rng.randint(30, 100), rng.randint(20, 80), rng.randint(10, 60)),
    )
    fill = Image.new("RGB", image.size, rng.choice(palette))
    return Image.composite(fill, image, mask)


def synthetic_dirt_patch(size: tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = size
    patch = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch, "RGBA")
    blob_count = rng.randint(3, 8)
    colors = ((25, 20, 15), (55, 45, 35), (95, 85, 70), (20, 35, 45), (120, 100, 75))
    for _ in range(blob_count):
        cx = rng.randint(-width // 5, max(0, width + width // 5))
        cy = rng.randint(-height // 5, max(0, height + height // 5))
        radius_x = rng.randint(max(2, width // 12), max(2, width // 3))
        radius_y = rng.randint(max(2, height // 12), max(2, height // 3))
        color = rng.choice(colors)
        alpha = rng.randint(65, 170)
        draw.ellipse((cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y), fill=(*color, alpha))
    for _ in range(max(8, blob_count * 5)):
        x = rng.randrange(max(1, width))
        y = rng.randrange(max(1, height))
        radius = rng.randint(1, max(1, min(width, height) // 18))
        color = rng.choice(colors)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, rng.randint(80, 190)))
    blur_radius = max(0.5, min(width, height) * 0.025)
    return patch.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def external_dirt_patch(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        patch = source.convert("RGBA")
    patch.thumbnail(size, Image.Resampling.LANCZOS)
    if patch.getchannel("A").getextrema() == (255, 255):
        alpha = Image.new("L", patch.size, 0)
        ImageDraw.Draw(alpha).ellipse((0, 0, patch.width - 1, patch.height - 1), fill=210)
        alpha = alpha.filter(ImageFilter.GaussianBlur(max(1, min(patch.size) // 20)))
        patch.putalpha(alpha)
    return patch


def apply_dirt(
    image: Image.Image,
    rng: random.Random,
    min_size: float,
    max_size: float,
    dirt_paths: list[Path],
) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = random_box(width, height, rng, min_size, max_size)
    patch_size = (max(2, right - left), max(2, bottom - top))
    if dirt_paths:
        patch = external_dirt_patch(rng.choice(dirt_paths), patch_size)
    else:
        patch = synthetic_dirt_patch(patch_size, rng)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer.alpha_composite(patch, (left, top))
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def make_ng_image(
    image: Image.Image,
    rng: random.Random,
    defect_types: tuple[str, ...],
    min_defects: int,
    max_defects: int,
    min_size: float,
    max_size: float,
    opacity: float,
    dirt_paths: list[Path],
) -> tuple[Image.Image, list[str]]:
    result = image.convert("RGB")
    applied: list[str] = []
    for _ in range(rng.randint(min_defects, max_defects)):
        defect_type = rng.choice(defect_types)
        if defect_type == "occlusion":
            result = apply_occlusion(result, rng, min_size, max_size, opacity)
        else:
            result = apply_dirt(result, rng, min_size, max_size, dirt_paths)
        applied.append(defect_type)
    if ImageChops.difference(result, image.convert("RGB")).getbbox() is None:
        raise ValueError("缺陷未改变图像像素，请检查脏污贴图是否全透明，或调整参数。")
    return result, applied


def save_image(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.convert("RGB").save(destination, format="JPEG", quality=95, subsampling=0)
    elif suffix == ".png":
        image.save(destination, format="PNG", optimize=True)
    elif suffix == ".webp":
        image.save(destination, format="WEBP", quality=95)
    else:
        image.save(destination)


def generate_dataset(args: argparse.Namespace, defect_types: tuple[str, ...]) -> dict:
    validate_args(args)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    sources, duplicates = unique_images(list_images(input_dir, args.recursive))
    split = split_images(
        sources,
        (args.train_ratio, args.test_good_ratio, args.test_ng_ratio),
        args.seed,
    )
    dirt_paths = []
    if args.dirt_dir is not None:
        dirt_paths = list_images(args.dirt_dir.resolve(), True)
        for path in dirt_paths:
            load_rgb(path)
    prepare_output(output_dir, args.overwrite)

    rng = random.Random(args.seed + 1)
    manifest_rows = []
    for split_name, paths in split.items():
        destination_dir = output_dir / split_name
        for source in paths:
            output_name = safe_output_name(source, input_dir)
            destination = destination_dir / output_name
            relative_source = source.relative_to(input_dir).as_posix()
            if split_name == "test/ng":
                # Lossless output preserves small defects and avoids introducing
                # whole-image JPEG recompression artifacts only in the NG class.
                destination = destination.with_suffix(".png")
                generated, applied = make_ng_image(
                    load_rgb(source),
                    rng,
                    defect_types,
                    args.min_defects,
                    args.max_defects,
                    args.min_size,
                    args.max_size,
                    args.occlusion_opacity,
                    dirt_paths,
                )
                save_image(generated, destination)
            else:
                shutil.copy2(source, destination)
                applied = []
            manifest_rows.append(
                {
                    "split": split_name,
                    "source": relative_source,
                    "output": destination.relative_to(output_dir).as_posix(),
                    "defects": applied,
                }
            )

    manifest = {
        "version": 1,
        "generator": "make_anomaly_dataset",
        "duplicates_skipped": duplicates,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "ratios": {
            "train/good": args.train_ratio,
            "test/good": args.test_good_ratio,
            "test/ng": args.test_ng_ratio,
        },
        "counts": {name: len(items) for name, items in split.items()},
        "ng_sources_are_disjoint": True,
        "defect_types": list(defect_types),
        "defect_parameters": {key: getattr(args, key) for key in (
            "min_defects", "max_defects", "min_size", "max_size", "occlusion_opacity")},
        "items": manifest_rows,
    }
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    try:
        defect_types = validate_args(args)
        manifest = generate_dataset(args, defect_types)
    except (FileExistsError, OSError, ValueError) as error:
        print(f"错误：{error}")
        return 2

    print("数据集生成完成：")
    print(f"  跳过重复图像: {len(manifest['duplicates_skipped'])}")
    for split_name, count in manifest["counts"].items():
        print(f"  {split_name}: {count}")
    print(f"  manifest: {args.output_dir.resolve() / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
