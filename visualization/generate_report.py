"""Independent, offline result viewer. Never imports or executes algorithms."""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"2D image directory does not exist: {directory}")
    def natural_key(path: Path):
        return [(1, int(v)) if v.isdigit() else (0, v.casefold()) for v in re.split(r"(\d+)", path.name)]
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)


def unpack(document: Any) -> tuple[dict, list[dict]]:
    """Accept raw Bridge JSON, our envelope, or a chronological JSON array."""
    if isinstance(document, list):
        records = [r for r in document if isinstance(r, dict)]
        if not records:
            raise ValueError("Result array is empty")
        response = next((r for r in reversed(records) if "results_board" in r or "results_3d" in r), records[-1])
        return response, [r for r in records if isinstance(r.get("results_2d"), dict)]
    if not isinstance(document, dict):
        raise ValueError("Expected a Bridge response object or response array")
    response = document.get("response", document)
    if not isinstance(response, dict):
        raise ValueError("response must be an object")
    if not any(k in response for k in ("ok", "results_board", "results_2d", "results_3d", "error")):
        raise ValueError("This JSON is not a Bridge detection response")
    records = document.get("responses_2d", [])
    if not isinstance(records, list):
        raise ValueError("responses_2d must be an array")
    return response, [r for r in records if isinstance(r, dict)]


def _data_url(image: Image.Image, kind: str = "JPEG") -> dict:
    width, height = image.size
    preview = image.convert("RGB")
    preview.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    preview.save(stream, format=kind, **({"quality": 86} if kind == "JPEG" else {}))
    mime = "image/jpeg" if kind == "JPEG" else "image/png"
    return {"width": width, "height": height, "url": f"data:{mime};base64," + base64.b64encode(stream.getvalue()).decode("ascii")}


def _depth_rgb(depth: np.ndarray) -> tuple[Image.Image, list[float] | None]:
    valid = np.isfinite(depth) & (depth != 0) & (depth != 65535)
    if not np.any(valid):
        return Image.fromarray(np.zeros((*depth.shape, 3), dtype=np.uint8)), None
    low, high = np.percentile(depth[valid], [2, 98]).astype(float)
    ratio = np.clip((depth.astype(np.float32) - low) / max(high - low, 1), 0, 1)
    stops = np.array([[41, 52, 118], [33, 142, 170], [107, 196, 133], [249, 186, 85], [232, 97, 92]], dtype=np.float32)
    location = ratio * (len(stops) - 1)
    index = np.minimum(location.astype(int), len(stops) - 2)
    fraction = (location - index)[..., None]
    rgb = stops[index] * (1 - fraction) + stops[index + 1] * fraction
    rgb[~valid] = [17, 24, 36]
    return Image.fromarray(rgb.astype(np.uint8)), [float(low), float(high)]


def build_payload(document: Any, *, base_dir: Path, images_2d: Path | None = None, depth_path: Path | None = None) -> dict:
    response, replies = unpack(document)
    envelope = document if isinstance(document, dict) else {}
    board = response.get("results_board") or {}
    source_pieces = board.get("pieces") or (response.get("results_3d") or {}).get("pieces") or []
    pieces = {int(p.get("piece_id", i + 1)): dict(p) for i, p in enumerate(source_pieces) if isinstance(p, dict)}
    if isinstance(response.get("results_2d"), dict):
        replies = [*replies, response]
    for i, reply in enumerate(replies):
        pid = int(reply.get("piece_id") or i + 1)
        if isinstance(reply.get("results_2d"), dict):
            piece = pieces.setdefault(pid, {"piece_id": pid})
            piece["results_2d"] = reply["results_2d"]
            if "is_ok" not in piece:
                piece["is_ok"] = reply["results_2d"].get("is_ok")
    if len({str(r.get("board_id") or r.get("batch_id")) for r in [response, *replies] if r.get("board_id") or r.get("batch_id")}) > 1:
        raise ValueError("Response array contains multiple board IDs; split it into one report per board")
    warnings = [str(item) for item in envelope.get("warnings", []) if isinstance(item, str)]
    mapping: dict[int, Path] = {}
    if images_2d:
        mapping = {i + 1: p for i, p in enumerate(image_files(images_2d))}
        warnings.append("2D 图像按文件名自然排序对应片号 1…N，请核对与检测顺序一致。")
    elif isinstance(envelope.get("images_2d"), list):
        for item in envelope["images_2d"]:
            if isinstance(item, dict) and item.get("path"):
                p = Path(item["path"])
                mapping[int(item["piece_id"])] = p if p.is_absolute() else base_dir / p
    raw_depth = depth_path or envelope.get("image_3d")
    depth = None
    if raw_depth:
        p = Path(raw_depth)
        p = p if p.is_absolute() else base_dir / p
        try:
            with Image.open(p) as image:
                image.load()
                depth = np.asarray(image).copy()
            if depth.ndim != 2 or depth.dtype.kind not in "ui":
                raise ValueError("Depth must be a single-channel integer image")
        except (OSError, ValueError) as exc:
            warnings.append(f"3D 预览不可用：{exc}")
            depth = None
    for pid, piece in sorted(pieces.items()):
        if pid in mapping:
            try:
                with Image.open(mapping[pid]) as image:
                    piece["preview_2d"] = _data_url(image)
                expected = piece.get("image_2d_size")
                asset = piece["preview_2d"]
                if expected and list(expected) != [asset["width"], asset["height"]]:
                    piece["overlay_2d_valid"] = False
                    warnings.append(f"片 {pid} 的 2D 图片尺寸不匹配，已停用该片 2D 叠加。")
            except (OSError, ValueError) as exc:
                warnings.append(f"片 {pid} 2D 预览不可用：{exc}")
        if depth is not None:
            start, end = piece.get("depth_start_row"), piece.get("depth_end_row")
            if isinstance(start, int) and isinstance(end, int):
                if not 0 <= start < end <= depth.shape[0]:
                    warnings.append(f"片 {pid} 的深度行范围无效，未生成该片 3D 预览。")
                    continue
                local = depth[start:end, :]  # Keep full X: masks use full-width piece-local coordinates.
                piece["depth_origin_row"] = start
            else:
                local = depth
                piece["depth_origin_row"] = 0
            rgb, value_range = _depth_rgb(local)
            piece["preview_3d"] = {**_data_url(rgb, "PNG"), "gray_range": value_range}
    if not pieces:
        warnings.append("结果没有可显示的片数据。请查看服务错误或原始 JSON。")
    return {
        "viewer_version": 1,
        "title": str(envelope.get("title") or "检测结果复核"),
        "mode": envelope.get("mode") or response.get("stage") or "import",
        "board_id": response.get("board_id") or response.get("batch_id") or envelope.get("board_id") or "未提供板号",
        "response": response,
        "pieces": [pieces[k] for k in sorted(pieces)],
        "warnings": warnings,
    }


def safe_script_json(value: Any) -> str:
    # JSON data must not be able to terminate the script element.
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def render_report(document: Any, output: Path, *, base_dir: Path, images_2d: Path | None = None, depth_path: Path | None = None, force: bool = False) -> Path:
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite: {output}. Use --force if intended.")
    payload = build_payload(document, base_dir=base_dir, images_2d=images_2d, depth_path=depth_path)
    template = (HERE / "viewer.html").read_text(encoding="utf-8")
    template = template.replace("/*__VIEWER_CSS__*/", (HERE / "viewer.css").read_text(encoding="utf-8"))
    template = template.replace("/*__VIEWER_JS__*/", (HERE / "viewer.js").read_text(encoding="utf-8"))
    template = template.replace("__REPORT_JSON__", safe_script_json(payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template, encoding="utf-8")
    return output.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a standalone HTML viewer from saved Bridge JSON; no inference.")
    parser.add_argument("--result", type=Path, required=True, help="Raw Bridge response JSON or detect_and_view result.json")
    parser.add_argument("--images-2d", type=Path, help="Optional image folder, naturally sorted to piece IDs")
    parser.add_argument("--depth", type=Path, help="Optional original integer depth PNG")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        source = args.result.resolve()
        document = json.loads(source.read_text(encoding="utf-8-sig"))
        output = render_report(document, args.output or source.with_suffix(".report.html"), base_dir=source.parent,
                               images_2d=args.images_2d, depth_path=args.depth, force=args.force)
        print(f"Report: {output}")
        if args.open:
            webbrowser.open(output.as_uri())
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Cannot generate report: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
