"""CPU-only affine experiments; no model loading and no production config writes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time

# Set before importing NumPy/SciPy. These limits belong to this process only.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import cv2
import numpy as np
import scipy
import PIL
from algorithm.config import load_runtime_config
from algorithm.registration import (
    RegistrationConfig, affine_coordinate_context, platform_mask_2d,
    platform_mask_3d, refine_affine_registration, register_platform_masks,
)
from algorithm.tilt_correction import correct_smartray_depth, production_pitch_parameters

segment = importlib.import_module("algorithm.3D.segment")
VARIANTS = ("current", "baseline", "cache", "calibration", "multiscale", "budget")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(arguments):
    try:
        result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True,
                                timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment():
    return dict(host=platform.node(), platform=platform.platform(), processor=platform.processor(),
                logical_cpus=os.cpu_count(), python=sys.version, executable=sys.executable,
                versions=dict(numpy=np.__version__, scipy=scipy.__version__,
                              opencv=cv2.__version__, pillow=PIL.__version__),
                native_thread_env={key: os.environ[key] for key in
                                   ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                opencv_cpu_features=cv2.getCPUFeaturesLine(),
                windows_hardware=command(["powershell", "-NoProfile", "-Command",
                    "@{cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors);"
                    "ram_bytes=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory} | ConvertTo-Json -Compress"])
                    if os.name == "nt" else None,
                git_commit=command(["git", "rev-parse", "HEAD"]),
                git_status=command(["git", "status", "--short"]),
                torch_loaded="torch" in sys.modules)


def parameters(original, variant, profile=None):
    values = copy.deepcopy(original)
    if values.get("score_mode", "full") != "full":
        raise ValueError("This experiment requires score_mode=full; ROI sampling is excluded")
    if variant == "current":
        return values
    values.update(score_mode="full", cache_enabled=variant != "baseline",
                  calibration=dict(enabled=False, profile=None),
                  multiscale=dict(enabled=False), maxfev=None, total_maxfev=None)
    if variant == "calibration":
        if profile is None:
            raise ValueError("calibration requires --profile from a manually verified registration")
        values["calibration"] = dict(enabled=True, profile=copy.deepcopy(profile))
    elif variant == "multiscale":
        values["multiscale"] = dict(enabled=True, search_scale=0.5, search_maxiter=15,
                                    refine_maxiter=10, search_maxfev=None, refine_maxfev=None,
                                    refine_window_ratio=0.25)
    elif variant == "budget":
        values.update(maxfev=150, total_maxfev=150)
    return values


def select_inputs(directory, depth, pieces, expected):
    """One board per directory; accept 001.png or board_piece.png, never guess board IDs."""
    images, boards = {}, set()
    for path in sorted(directory.glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        match = re.fullmatch(r"(?:(\d+)_)?(\d+)", path.stem)
        if not match:
            continue
        board, number = match.groups()
        pid = int(number)
        if pid in images:
            raise ValueError(f"Duplicate piece {pid}: {images[pid]} / {path}")
        boards.add(board)
        images[pid] = path.resolve()
    if len(boards) > 1:
        raise ValueError("Use one board and one naming convention per --images-dir")
    selected = pieces if pieces is not None else list(range(1, expected + 1))
    if len(set(selected)) != len(selected) or any(pid < 1 or pid > expected for pid in selected):
        raise ValueError(f"--pieces must be unique IDs in 1..{expected}")
    missing = set(selected) - images.keys()
    if missing:
        raise ValueError(f"Missing 2D pieces: {sorted(missing)} in {directory}")
    if not depth.is_file():
        raise ValueError(f"Depth image does not exist: {depth}")
    board = next(iter(boards), None)
    if board is not None and depth.stem.isdigit() and int(board) != int(depth.stem):
        raise ValueError("2D board number differs from 3D filename")
    return [(pid, images[pid]) for pid in selected]


def prepare(config, inputs, depth_path):
    started = time.perf_counter()
    height = config.get("height_measurement_3d", {}).get("parameters", {})
    pitch = production_pitch_parameters(config, height.get("gray_offset", 32768.),
                                        height.get("vertical_resolution_mm", .00105))
    raw = segment.decode_depth_png(depth_path.read_bytes(), "png")
    depth, correction = correct_smartray_depth(raw, pitch)
    segmentation = segment.SegmentationConfig.from_dict(config["segmentation_3d"]["parameters"])
    boxes = {box["piece_id"]: box for box in segment.segment_depth_image(depth, segmentation)}
    registration = RegistrationConfig.from_dict(config["registration_3d"]["parameters"])
    prepared, info = [], []
    depth.setflags(write=False)
    for pid, path in inputs:
        image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode 2D image: {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        box = boxes[pid]
        # Match formal detection: preserve the full board width, crop rows only.
        local = depth[box["y1"]:box["y2"], :]
        coarse = register_platform_masks(platform_mask_2d(image, registration),
                                          platform_mask_3d(local, registration), registration)
        context = affine_coordinate_context(gray.shape, local.shape, correction)
        gray.setflags(write=False)
        prepared.append(dict(piece_id=pid, gray=gray, depth=local, coarse=coarse, context=context))
        info.append(dict(piece_id=pid, path=str(path), sha256=sha256(path),
                         image_size=context["image_size"], depth_size=context["depth_size"],
                         segmentation_box=box, coarse=coarse, context=context))
    return prepared, dict(preparation_ms=(time.perf_counter()-started)*1000,
                          pitch=correction, depth_shape=list(depth.shape), pieces=info,
                          note="Decode, Pitch, segmentation and coarse matching run once; excluded from affine batch timings")


def measure_piece(item, values, submitted):
    started, cpu_started = time.perf_counter(), time.thread_time()
    row = dict(piece_id=item["piece_id"], queue_ms=(started-submitted)*1000)
    try:
        row["registration"] = refine_affine_registration(
            item["gray"], item["depth"], copy.deepcopy(item["coarse"]),
            copy.deepcopy(values), context=item["context"])
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    row.update(affine_ms=(time.perf_counter()-started)*1000,
               thread_cpu_ms=(time.thread_time()-cpu_started)*1000)
    return row


def measure_round(pool, prepared, values):
    started, cpu_started = time.perf_counter(), time.process_time()
    futures = [pool.submit(measure_piece, item, values, time.perf_counter()) for item in prepared]
    rows = [future.result() for future in futures]
    return dict(batch_wall_ms=(time.perf_counter()-started)*1000,
                process_cpu_ms=(time.process_time()-cpu_started)*1000, pieces=rows)


def stats(values):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not values:
        return dict(mean=None, median=None, p95=None, maximum=None)
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                p95=float(np.percentile(values, 95)), maximum=max(values))


def displacement(first, second, image_size):
    try:
        a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        if a.shape != (2, 3) or b.shape != (2, 3) or not np.isfinite([a, b]).all():
            return None
        w, h = image_size
        points = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1],
                           [w-1, h-1, 1], [(w-1)/2, (h-1)/2, 1]])
        return float(np.linalg.norm(points @ (a-b).T, axis=1).max())
    except (TypeError, ValueError):
        return None


def summarize(groups, input_info):
    sizes = {p["piece_id"]: p["image_size"] for p in input_info["pieces"]}
    references = {}
    for group in sorted(groups, key=lambda g: g["workers"]):
        for row in group["rounds"][0]["pieces"]:
            references.setdefault((group["variant"], group["opencv_threads_actual"], row["piece_id"]), row)
    available = {group["variant"] for group in groups}
    reference_variant = next(name for name in ("baseline", "current", "cache", "calibration", "multiscale", "budget") if name in available)
    output = []
    for group in groups:
        rows = [row for repeat in group["rounds"] for row in repeat["pieces"]]
        results = [row.get("registration", {}) for row in rows]
        differences, loss_differences, eval_differences = [], [], []
        variant_differences, variant_loss_differences = [], []
        for row in rows:
            key = (group["variant"], group["opencv_threads_actual"], row["piece_id"])
            ref = references[key].get("registration", {})
            result = row.get("registration", {})
            differences.append(displacement(ref.get("affine_matrix_2d_to_3d"), result.get("affine_matrix_2d_to_3d"), sizes[row["piece_id"]]))
            for target, field in ((loss_differences, "affine_score_after"), (eval_differences, "affine_function_evaluations")):
                left, right = ref.get(field), result.get(field)
                if isinstance(left, (float, int)) and isinstance(right, (float, int)):
                    target.append(abs(left-right))
            cross = references.get((reference_variant, group["opencv_threads_actual"], row["piece_id"]), {}).get("registration", {})
            variant_differences.append(displacement(cross.get("affine_matrix_2d_to_3d"), result.get("affine_matrix_2d_to_3d"), sizes[row["piece_id"]]))
            left, right = cross.get("affine_score_after"), result.get("affine_score_after")
            if isinstance(left, (float, int)) and isinstance(right, (float, int)):
                variant_loss_differences.append(right-left)
        walls = [r["batch_wall_ms"] for r in group["rounds"]]
        output.append(dict(variant=group["variant"], workers=group["workers"],
                           opencv_threads=group["opencv_threads_actual"],
                           piece_ms=stats([row["affine_ms"] for row in rows]), batch_ms=stats(walls),
                           queue_ms=stats([row["queue_ms"] for row in rows]),
                           throughput_pieces_per_s=len(rows)*1000/sum(walls),
                           nfev=stats([r.get("affine_function_evaluations") for r in results]),
                           warps=stats([r.get("affine_warp_evaluations") for r in results]),
                           compute_mean_ms=stats([r.get("affine_score_compute_mean_ms") for r in results]),
                           cache_hit_rate=stats([r.get("affine_cache_hit_rate") for r in results]),
                           budget_exhausted_count=sum(bool((r.get("affine_budget") or {}).get("exhausted")) for r in results),
                           fallback_count=sum(bool((r.get("affine_multiscale") or {}).get("fallback_used")) for r in results),
                           valid_count=sum(bool(r.get("affine_valid")) for r in results), count=len(rows),
                           error_count=sum(bool(row.get("error") or row.get("registration", {}).get("affine_error")) for row in rows),
                           max_reference_displacement_px=stats(differences)["maximum"],
                           reference_matrix_comparison_count=sum(v is not None for v in differences),
                           max_reference_loss_difference=stats(loss_differences)["maximum"],
                           max_reference_nfev_difference=stats(eval_differences)["maximum"],
                           comparison_variant=reference_variant,
                           max_comparison_displacement_px=stats(variant_differences)["maximum"],
                           comparison_matrix_count=sum(v is not None for v in variant_differences),
                           max_comparison_loss_increase=stats(variant_loss_differences)["maximum"],
                           reference_workers=min(g["workers"] for g in groups if g["variant"] == group["variant"]
                                                 and g["opencv_threads_actual"] == group["opencv_threads_actual"])))
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_report(folder, report):
    (folder / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    rows = []
    for group in report["groups"]:
        for index, repeat in enumerate(group["rounds"], 1):
            for row in repeat["pieces"]:
                result = row.get("registration", {})
                rows.append(dict(variant=group["variant"], workers=group["workers"],
                                 opencv_threads=group["opencv_threads_actual"], repeat=index,
                                 piece_id=row["piece_id"], affine_ms=row["affine_ms"], queue_ms=row["queue_ms"],
                                 thread_cpu_ms=row["thread_cpu_ms"], batch_wall_ms=repeat["batch_wall_ms"],
                                 nfev=result.get("affine_function_evaluations"), warps=result.get("affine_warp_evaluations"),
                                 cache_hit_rate=result.get("affine_cache_hit_rate"),
                                 compute_mean_ms=result.get("affine_score_compute_mean_ms"),
                                 loss=result.get("affine_score_after"), valid=result.get("affine_valid"),
                                 stop_reason=result.get("affine_stop_reason"),
                                 error=row.get("error") or result.get("affine_error")))
    write_csv(folder / "pieces.csv", rows)
    flat = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
             for key, value in summary.items()} for summary in report["summary"]]
    write_csv(folder / "summary.csv", flat)
    lines = ["# 仿射独立实验", "", f"代码：{report['environment']['git_commit']}",
             "", "仅计仿射求解；不含解码、Pitch、粗配准、YOLO、掩码映射与测高。",
             "线程组按固定随机顺序执行；预热不参与统计。小样本P95只供描述。", "",
             "|方案|并发|OpenCV线程|单片中位ms|单片P95ms|整批中位ms|片/秒|平均nfev|有效/总数|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in report["summary"]:
        lines.append(f"|{s['variant']}|{s['workers']}|{s['opencv_threads']}|{s['piece_ms']['median']:.1f}|"
                     f"{s['piece_ms']['p95']:.1f}|{s['batch_ms']['median']:.1f}|"
                     f"{s['throughput_pieces_per_s']:.2f}|{s['nfev']['mean'] or 0:.1f}|{s['valid_count']}/{s['count']}|")
    for variant in report["variants"]:
        measured = [s for s in report["summary"] if s["variant"] == variant]
        if not measured:
            continue
        eligible = [s for s in measured if not s["error_count"] and s["valid_count"] == s["count"]]
        lines += ["", f"### {variant} 实测记录", ""]
        if eligible:
            best = min(eligible, key=lambda s: s["batch_ms"]["median"])
            lines.append(f"在所有片均通过内置有效性判断的已测组中，整批中位耗时最小：workers={best['workers']}、"
                         f"OpenCV={best['opencv_threads']}，{best['batch_ms']['median']:.1f}ms。仍需叠加及测高验收。")
        else:
            lines.append("没有所有片均有效且无执行错误的组；本次不能推荐生产参数。")
        for s in measured:
            offset = s["max_reference_displacement_px"]
            lines.append(f"- workers={s['workers']}、OpenCV={s['opencv_threads']}："
                         f"错误{s['error_count']}，预算耗尽{s['budget_exhausted_count']}/{s['count']}，"
                         f"执行回退{s['fallback_count']}/{s['count']}；相对同方案最小并发首轮，"
                         f"最大nfev差={s['max_reference_nfev_difference']}，最大点位差={offset}px"
                         f"（成功比较{s['reference_matrix_comparison_count']}/{s['count']}次）。")
    lines += ["", "## 如何判读", "",
              "- 同方案、同片的nfev/warp/矩阵/损失不变，并发后单片变慢：优先检查执行竞争和调度，不能直接判定都是GIL。",
              "- 单片变慢但整批变快：吞吐与单片延迟存在取舍，按整板需求选并发。",
              "- nfev明显增加：先核对输入、参数、初值和停止原因。compute_mean_ms增加表示单次评分也变慢。",
              "- summary的reference差异以同方案最小并发第一轮为参照；不同方案的质量须按piece_id在results.json比较。",
              "- 预算触顶、回退、无效、区域损失恶化均需逐片检查。affine_valid不能代替彩色叠加和最终测高验收。",
              "- thread_cpu_ms仅为当前线程CPU时间；差值包含多种等待，不直接等于GIL等待。",
              "- 此工具没有正式流程的其它并行负载；实际整板验证仍需在原检测环境进行。",
              "", "完整参数、各级停止原因和矩阵在results.json；逐片计时在pieces.csv。"]
    (folder / "analysis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=ROOT / "demodatasets/2D")
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.json")
    parser.add_argument("--pieces", type=int, nargs="+")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 15])
    parser.add_argument("--opencv-threads", type=int, nargs="+", default=[1])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=["current"])
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "visualization/output/affine-benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and prepare inputs; do not optimize")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0 or min(args.workers) < 1 or min(args.opencv_threads) < 0:
        parser.error("repeats/workers must be positive; warmup/opencv-threads must be nonnegative")
    if len(set(args.workers)) != len(args.workers) or len(set(args.opencv_threads)) != len(args.opencv_threads) or len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate experiment groups are not allowed")
    config_path = args.config.resolve()
    config = load_runtime_config(config_path.read_text(encoding="utf-8-sig"), str(config_path))
    profile = json.loads(args.profile.read_text(encoding="utf-8-sig")) if args.profile else None
    variants = {name: parameters(config["dynamic_affine_3d"]["parameters"], name, profile) for name in args.variants}
    if args.depth is None:
        candidates = list((ROOT / "demodatasets/3D").glob("*.png"))
        if len(candidates) != 1:
            parser.error("Specify --depth; the default 3D directory must contain exactly one PNG")
        args.depth = candidates[0]
    inputs = select_inputs(args.images_dir, args.depth, args.pieces,
                           config["segmentation_3d"]["parameters"].get("expected_piece_count", 10))
    cv2.setNumThreads(1)
    prepared, input_info = prepare(config, inputs, args.depth)
    folder = args.output_dir / ("run-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(format="affine_benchmark_v1", environment=environment(),
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  config=config, config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  depth=dict(path=str(args.depth.resolve()), sha256=sha256(args.depth)),
                  code_sha256={str(path.relative_to(ROOT)): sha256(path) for path in
                               (Path(__file__).resolve(), ROOT / "python/algorithm/registration.py",
                                ROOT / "python/algorithm/tilt_correction.py", ROOT / "python/algorithm/3D/segment.py")},
                  inputs=input_info, variants=variants, groups=[], summary=[], complete=False)
    if args.dry_run:
        report.update(complete=True, dry_run=True)
        save_report(folder, report)
        print(f"Inputs OK; no affine optimization. Report: {folder}", flush=True)
        return 0
    plan = [(variant, workers, native) for variant in variants for workers in args.workers for native in args.opencv_threads]
    random.Random(args.seed).shuffle(plan)
    report["group_order"] = plan
    save_report(folder, report)
    for index, (variant, workers, native) in enumerate(plan, 1):
        cv2.setNumThreads(native)  # Previous executor has drained before this global change.
        group = dict(variant=variant, workers=workers, opencv_threads_requested=native,
                     opencv_threads_actual=cv2.getNumThreads(), rounds=[])
        print(f"[{index}/{len(plan)}] {variant}, workers={workers}, OpenCV={cv2.getNumThreads()}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(args.warmup):
                measure_round(pool, prepared, variants[variant])
            for repeat in range(args.repeats):
                measured = measure_round(pool, prepared, variants[variant])
                group["rounds"].append(measured)
                print(f"  round {repeat+1}: batch={measured['batch_wall_ms']:.1f} ms", flush=True)
        report["groups"].append(group)
        report["summary"] = summarize(report["groups"], input_info)
        save_report(folder, report)
    report["complete"] = True
    report["environment"]["torch_loaded"] = "torch" in sys.modules
    save_report(folder, report)
    print(f"Report: {folder}", flush=True)
    return 1 if any(s["error_count"] for s in report["summary"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
