"""Optional standalone detection client; leaves the Bridge and algorithms untouched."""
from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from generate_report import image_files, render_report

ROOT = Path(__file__).resolve().parents[1]


class ProtocolClient:
    def __init__(self, python: Path, server: Path, log_path: Path):
        self.log = log_path.open("w", encoding="utf-8")
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            self.process = subprocess.Popen(
                [str(python), "-u", str(server)], cwd=str(server.parent),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, encoding="utf-8", bufsize=1, env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException:
            self.log.close()
            raise
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.pending: dict[str, queue.Queue[dict]] = {}
        self.pending_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.timeline: list[dict] = []
        self.started_at = time.perf_counter()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    self.lines.put(line)
                    continue
                request_id = response.get("request_id") if isinstance(response, dict) else None
                if request_id:
                    with self.pending_lock:
                        waiter = self.pending.get(str(request_id))
                    if waiter is not None:
                        waiter.put(response)
                        continue
                self.lines.put(response)
        finally:
            self.lines.put(None)

    def request(self, value: dict, timeout: float) -> dict:
        request_id = uuid.uuid4().hex
        request = {**value, "request_id": request_id}
        waiter: queue.Queue[dict] = queue.Queue(maxsize=1)
        with self.pending_lock:
            self.pending[request_id] = waiter
        try:
            with self.write_lock:
                sent = time.perf_counter()
                self.process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
                self.process.stdin.flush()
        except BaseException:
            with self.pending_lock:
                self.pending.pop(request_id, None)
            raise
        try:
            response = waiter.get(timeout=timeout)
        except queue.Empty as exc:
            self.process.kill()  # A late response must not become the next request's reply.
            raise TimeoutError(f"Request timed out after {timeout:g}s: {value.get('cmd')}") from exc
        finally:
            with self.pending_lock:
                self.pending.pop(request_id, None)
        received = time.perf_counter()
        with self.pending_lock:
            self.timeline.append({
                "cmd": value.get("cmd"), "stage": value.get("stage"),
                "piece_id": value.get("piece_id"), "request_id": request_id,
                "sent_ms": round((sent - self.started_at) * 1000, 2),
                "received_ms": round((received - self.started_at) * 1000, 2),
                "duration_ms": round((received - sent) * 1000, 2),
                "ok": bool(response.get("ok")),
            })
        return response

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self.request({"cmd": "release"}, 10)
                    self.process.wait(timeout=3)
                except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                    if self.process.poll() is None:
                        self.process.kill()
            self.process.wait(timeout=5)
        finally:
            if self.process.stdin:
                self.process.stdin.close()
            self.reader.join(timeout=2)
            if self.process.stdout:
                self.process.stdout.close()
            self.log.close()


def parse_threshold(value: str) -> float | dict[str, float]:
    """--threshold chip_short=0.4,chip_long=2.0; scalar retained for explicit global overrides."""
    def numeric(text: str) -> float:
        try:
            number = float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("threshold must contain numeric values") from exc
        if not math.isfinite(number) or number < 0:
            raise argparse.ArgumentTypeError("threshold must be finite and non-negative")
        return number
    if "=" not in value:
        return numeric(value)
    result = {}
    for item in value.split(","):
        if item.count("=") != 1:
            raise argparse.ArgumentTypeError("Use chip_short=0.4,chip_long=2.0")
        name, number = (part.strip() for part in item.split("=", 1))
        if not name or name in result:
            raise argparse.ArgumentTypeError("threshold model names must be non-empty and unique")
        result[name] = numeric(number)
    return result


def algorithm_request(score_mode: str | None, score_pool_kernel: int | None,
                      silver_calibration: Path | None = None) -> dict:
    manifest = json.loads((ROOT / "config/config.json").read_text(encoding="utf-8-sig"))
    first_path = (ROOT / "config" / manifest["config_2d"]).resolve()
    second_path = (ROOT / "config" / manifest["config_3d"]).resolve()
    first = json.loads(first_path.read_text(encoding="utf-8-sig"))
    second = json.loads(second_path.read_text(encoding="utf-8-sig"))
    combined = {**first, **{k: v for k, v in second.items() if k != "model_path"}}
    chip_config = combined.get("chip_anomaly_2d", {}).get("parameters", {})
    if silver_calibration is not None:
        calibration_path = silver_calibration.resolve()
        if not calibration_path.is_file():
            raise FileNotFoundError(calibration_path)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
        if calibration.get("mode") != "polygon" or len(calibration.get("template_points", [])) < 3:
            raise ValueError("银浆标定文件没有有效的多边形顶点；请先完成并应用可视化标定")
        combined["silver_overflow_2d"]["parameters"]["boundary"] = {
            "mode": "polygon", "calibration_path": str(calibration_path)
        }
    if score_mode is not None:
        chip_config["score_mode"] = score_mode
    if score_pool_kernel is not None:
        chip_config["score_pool_kernel"] = score_pool_kernel
    models = {name: str((first_path.parent / path).resolve()) for name, path in first["model_path"].items()}
    for path in models.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return {"cmd": "init_algorithm", "stage": "board", "config_path": str(first_path),
            "config_text": json.dumps(combined, ensure_ascii=True), "model_path": models}


def threshold_request(threshold: float | dict[str, float] | None) -> dict:
    return {"threshold": threshold} if threshold is not None else {}


def mapped_file(stack: ExitStack, path: Path, kind: str, piece_id: int | None = None) -> dict:
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"Empty input image: {path}")
    name = "result_viewer_" + uuid.uuid4().hex
    mapping = mmap.mmap(-1, len(raw), tagname=name, access=mmap.ACCESS_WRITE)
    stack.callback(mapping.close)
    mapping.write(raw)
    if kind == "2d":
        return {"piece_id": piece_id, "shm_name": name, "image_size": len(raw), "image_format": "encoded"}
    return {"shm_name": name, "data_size": len(raw), "data_format": "png"}


def save_timing_csv(envelope: dict, path: Path) -> None:
    """Flatten existing measurements only; do not infer or add overlapping stages."""
    rows = []
    def collect(phase, piece_id, timings):
        for stage, value in (timings or {}).items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                rows.append(dict(board_id=envelope["board_id"], phase=phase,
                                 piece_id=piece_id, stage=stage, elapsed_ms=round(value, 3)))
    for reply in envelope.get("responses_2d", []):
        collect("2d", reply.get("piece_id"), reply.get("timings"))
    response = envelope.get("response", {})
    collect("board", "", response.get("timings"))
    for piece in response.get("results_board", {}).get("pieces", []):
        collect("joint_3d", piece.get("piece_id"), piece.get("timings"))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["board_id", "phase", "piece_id", "stage", "elapsed_ms"])
        writer.writeheader()
        writer.writerows(rows)


def detect_batch(client: ProtocolClient, args, paths: list[Path], depth: Path,
                 envelope: dict) -> None:
    """Run one board on an already initialized service; never reload models."""
    board_id = envelope["board_id"]
    started = time.perf_counter()
    timeline_index = len(client.timeline)
    input_prepare_ms = 0.0

    def send(request):
        response = client.request(request, args.timeout)
        label = (f"2D #{request['piece_id']}" if request.get("piece_id")
                 else "3D board" if request["cmd"] == "detect_3d_board" else "Batch board")
        print(f"[{board_id}] {label} returned: request_ok={response.get('ok')}, "
              f"algorithm_ms={response.get('elapsed_ms', 'not provided')}", flush=True)
        timings = response.get("timings", {})
        keys = ("gpu_queue_ms", "cuda_before_yolo_ms", "yolo_predict_wall_ms", "cuda_after_yolo_ms",
                "mask_extract_to_cpu_ms", "efficientad_total_ms", "silver_total_ms",
                "cpu_postprocess_queue_ms")
        details = "; ".join(f"{key}={timings[key]:.1f}ms" for key in keys if key in timings)
        if details:
            print(f"[{board_id}] {label} timings: {details}", flush=True)
        return response

    def map_image(stack, path, kind, piece_id=None):
        nonlocal input_prepare_ms
        start = time.perf_counter()
        value = mapped_file(stack, path, kind, piece_id)
        input_prepare_ms += (time.perf_counter() - start) * 1000
        return value

    try:
        if args.mode == "stream" and args.parallel:
            print("Parallel stream: submit 10 2D pieces and 1 3D board scan...", flush=True)
            with ExitStack() as stack, ThreadPoolExecutor(max_workers=11) as pool:
                descriptors = [map_image(stack, path, "2d", index)
                               for index, path in enumerate(paths, 1)]
                depth_descriptor = map_image(stack, depth, "3d")
                depth_future = pool.submit(send, {"cmd": "detect_3d_board", "stage": "3d",
                                                 "board_id": board_id, **depth_descriptor})
                piece_futures = [pool.submit(send, {
                    "cmd": "detect_2d_piece", "stage": "2d", "board_id": board_id,
                    **threshold_request(args.threshold), **descriptor}) for descriptor in descriptors]
                for future in as_completed(piece_futures):
                    envelope["responses_2d"].append(future.result())
                failures = [item for item in envelope["responses_2d"] if not item.get("ok")]
                if failures:
                    client.request({"cmd": "release_board", "board_id": board_id}, args.timeout)
                response = depth_future.result()
                if failures:
                    response = {"ok": False, "cmd": "detect_3d_board", "board_id": board_id,
                                "error": "2d_piece_failed", "message": "; ".join(
                                    f"piece {item.get('piece_id')}: {item.get('message') or item.get('error')}"
                                    for item in failures)}
        elif args.mode == "stream":
            for index, path in enumerate(paths, 1):
                print(f"2D piece {index}/10: {path.name}", flush=True)
                with ExitStack() as stack:
                    piece_response = send({"cmd": "detect_2d_piece", "stage": "2d",
                        "board_id": board_id, **threshold_request(args.threshold),
                        **map_image(stack, path, "2d", index)})
                envelope["responses_2d"].append(piece_response)
                if not piece_response.get("ok"):
                    raise RuntimeError(piece_response.get("error") or f"2D piece {index} failed")
            with ExitStack() as stack:
                response = send({"cmd": "detect_3d_board", "stage": "3d", "board_id": board_id,
                                 **map_image(stack, depth, "3d")})
        else:
            with ExitStack() as stack:
                response = send({"cmd": "detect_board", "stage": "board", "batch_id": board_id,
                    "images_2d": [map_image(stack, path, "2d", index + 1)
                                  for index, path in enumerate(paths)],
                    "image_3d": map_image(stack, depth, "3d")})
        envelope["response"] = response
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        envelope["response"] = {"ok": False, "board_id": board_id, "error": str(exc)}
        print(f"Detection did not complete: {exc}", file=sys.stderr)
    finally:
        envelope["responses_2d"].sort(key=lambda item: int(item.get("piece_id") or 0))
        envelope["timeline"] = [dict(item) for item in client.timeline[timeline_index:]
                                if item["cmd"] in {"detect_2d_piece", "detect_3d_board", "detect_board"}]
        entries = envelope["timeline"]
        envelope["benchmark"] = {
            "batch_wall_ms": (time.perf_counter() - started) * 1000,
            "input_prepare_ms": input_prepare_ms,
            "request_window_ms": (max(item["received_ms"] for item in entries)
                                  - min(item["sent_ms"] for item in entries)) if entries else None,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the existing Bridge Python service independently, save JSON, and generate an offline report.")
    parser.add_argument("--images-2d", type=Path, default=ROOT / "demodatasets/2D")
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--board-id", default="")
    parser.add_argument("--mode", choices=("stream", "board"), default="stream",
                        help="stream runs all configured single-piece 2D checks; board matches the current batch Demo")
    parser.add_argument("--parallel", action="store_true",
                        help="submit ten 2D pieces and the 3D board scan concurrently; stream mode only")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Run N batches using the same images and ONE initialized service; N=2 compares first/reused inference")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=600, help="Per-request timeout in seconds")
    parser.add_argument("--score-mode", choices=("pool+top",), default=None)
    parser.add_argument("--score-pool-kernel", type=int, default=None)
    parser.add_argument("--silver-calibration", type=Path,
                        help="Use this calibrated silver boundary JSON for this run without modifying production config")
    parser.add_argument("--threshold", type=parse_threshold, default=None,
                        help="Per-model overrides, e.g. chip_short=0.4,chip_long=2.0; a single number explicitly overrides both")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The existing algorithm transport uses Windows named shared memory.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.parallel and args.mode != "stream":
        parser.error("--parallel only works with --mode stream")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    try:
        paths = image_files(args.images_2d.resolve())
        if len(paths) != 10:
            raise ValueError(f"Expected 10 2D images, found {len(paths)}")
        if args.depth:
            depth = args.depth.resolve()
        else:
            candidates = image_files(ROOT / "demodatasets/3D")
            if len(candidates) != 1:
                raise ValueError("Use --depth to select exactly one integer depth PNG")
            depth = candidates[0]
        if not depth.is_file():
            raise FileNotFoundError(depth)
        if args.score_pool_kernel is not None and (args.score_pool_kernel < 1 or args.score_pool_kernel % 2 == 0):
            raise ValueError("--score-pool-kernel must be a positive odd integer")
        if args.silver_calibration is not None and not args.silver_calibration.is_file():
            raise FileNotFoundError(args.silver_calibration)
        if args.threshold is not None and args.mode != "stream":
            raise ValueError("--threshold is an inference request option and requires --mode stream")
        if isinstance(args.threshold, dict):
            allowed = {"chip_short", "chip_long"}
            unknown = set(args.threshold) - allowed
            if unknown:
                raise ValueError(f"Unknown threshold model names: {', '.join(sorted(unknown))}")
        init_req = algorithm_request(args.score_mode, args.score_pool_kernel,
                                     args.silver_calibration)
        board_id = args.board_id or "board-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        folder = args.output_dir or Path(__file__).parent / "output" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
        folder = folder.resolve()
        folder.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Input validation failed: {exc}", file=sys.stderr)
        return 1

    envelope = {"viewer_version": 1, "mode": args.mode, "board_id": board_id,
                "images_2d": [{"piece_id": i + 1, "path": str(p)} for i, p in enumerate(paths)],
                "image_3d": str(depth), "responses_2d": [], "response": {}}
    client = None
    startup = {"process_pid": None, "initialization_count": 0,
               "profiling": {"log_enabled": os.environ.get("BRIDGE_TIMING_LOG", "1").lower() in {"1", "true", "yes", "on"},
                             "cuda_sync_debug": os.environ.get("BRIDGE_TIMING_CUDA_SYNC", "0").lower() in {"1", "true", "yes", "on"}}}
    batches = []
    summary = {"requested_batches": args.repeat, "same_images": True,
               "startup": startup, "batches": []}
    startup_started = time.perf_counter()
    try:
        client = ProtocolClient(args.python.resolve(), ROOT / "python/transport/bridge_server.py", folder / "python.log")
        startup["process_pid"] = client.process.pid
        phase_start = time.perf_counter()
        response = client.request({"cmd": "init"}, args.timeout)
        startup["protocol_response"] = response
        startup["protocol_init_wall_ms"] = (time.perf_counter() - phase_start) * 1000
        if not response.get("ok"):
            envelope["response"] = response
            raise RuntimeError(response.get("error") or "Protocol init failed")
        print("Loading existing models (CUDA required)...", flush=True)
        phase_start = time.perf_counter()
        startup["initialization_count"] += 1
        response = client.request(init_req, args.timeout)
        startup["algorithm_response"] = response
        startup["algorithm_init_wall_ms"] = (time.perf_counter() - phase_start) * 1000
        startup["startup_wall_ms"] = (time.perf_counter() - startup_started) * 1000
        envelope["response"] = response
        if args.silver_calibration is not None:
            envelope["calibration"] = {
                "silver_boundary": str(args.silver_calibration.resolve()),
                "method": "sift_ransac",
            }
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "Algorithm initialization failed")
        print(f"Service PID={client.process.pid}; initialized once in "
              f"{startup['startup_wall_ms'] / 1000:.3f}s. Running {args.repeat} batch(es).", flush=True)
        (folder / "startup.json").write_text(json.dumps(startup, ensure_ascii=False, indent=2), encoding="utf-8")
        for index in range(1, args.repeat + 1):
            current_id = board_id if args.repeat == 1 else f"{board_id}-run-{index:02d}"
            batch_folder = folder if args.repeat == 1 else folder / f"batch-{index:02d}"
            if args.repeat > 1:
                batch_folder.mkdir(exist_ok=False)
            envelope = {"viewer_version": 1, "mode": args.mode, "board_id": current_id,
                        "images_2d": [{"piece_id": i + 1, "path": str(p)} for i, p in enumerate(paths)],
                        "image_3d": str(depth), "responses_2d": [], "response": {}}
            if args.silver_calibration is not None:
                envelope["calibration"] = {"silver_boundary": str(args.silver_calibration.resolve()),
                                           "method": "sift_ransac"}
            print(f"\nBatch {index}/{args.repeat}: {current_id} | "
                  f"{'first inference' if index == 1 else 'reuse loaded models, NO initialization'}", flush=True)
            detect_batch(client, args, paths, depth, envelope)
            benchmark = envelope["benchmark"]
            benchmark.update(batch_index=index, process_pid=client.process.pid,
                             model_reused=index > 1,
                             startup_charged_ms=startup["startup_wall_ms"] if index == 1 else 0.0)
            benchmark["total_with_startup_ms"] = benchmark["batch_wall_ms"] + benchmark["startup_charged_ms"]
            if args.repeat > 1:
                envelope["title"] = f"连续检测第 {index}/{args.repeat} 批 · {'首次推理' if index == 1 else '复用模型推理'}"
                envelope["warnings"] = [
                    f"同一服务 PID={client.process.pid}；模型初始化仅一次。本批检测（含输入准备）"
                    f" {benchmark['batch_wall_ms'] / 1000:.3f} 秒；"
                    f"本批计入初始化 {benchmark['startup_charged_ms'] / 1000:.3f} 秒；"
                    f"合计 {benchmark['total_with_startup_ms'] / 1000:.3f} 秒。"
                    "顶部算法整板耗时不含服务初始化或报告生成。"]
            summary["batches"].append({**benchmark, "board_id": current_id,
                "request_ok": bool(envelope["response"].get("ok")),
                "product_is_ok": envelope["response"].get("results_board", {}).get("is_ok"),
                "result_path": str(batch_folder / "result.json"),
                "report_path": str(batch_folder / "report.html")})
            batches.append((batch_folder, envelope))
            (batch_folder / "result.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
            save_timing_csv(envelope, batch_folder / "timings.csv")
            print(f"Batch {index}: detection incl. input preparation={benchmark['batch_wall_ms'] / 1000:.3f}s; "
                  f"incl. charged startup={benchmark['total_with_startup_ms'] / 1000:.3f}s", flush=True)
            if not envelope["response"].get("ok"):
                print("Stopping remaining batches after a request failure; saved partial results.", file=sys.stderr)
                break
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"Detection did not complete: {exc}", file=sys.stderr)
        summary["error"] = str(exc)
        if envelope["response"].get("ok") or not envelope["response"]:
            envelope["response"] = {"ok": False, "board_id": board_id, "error": str(exc)}
    finally:
        if client is not None:
            try:
                client.close()  # Release models only AFTER all requested batches.
            except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
                summary["cleanup_error"] = str(exc)
                print(f"Service cleanup failed: {exc}", file=sys.stderr)
    if not batches:
        batches.append((folder, envelope))
        (folder / "result.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        save_timing_csv(envelope, folder / "timings.csv")
    (folder / "startup.json").write_text(json.dumps(startup, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["completed_batches"] = len(summary["batches"])
    summary["success"] = (len(summary["batches"]) == args.repeat
                          and all(item["request_ok"] for item in summary["batches"])
                          and "error" not in summary
                          and "cleanup_error" not in summary)
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTiming summary: {folder / 'summary.json'}\nService log: {folder / 'python.log'}", flush=True)
    reports_ok = True
    # Generate previews after inference, so report CPU/IO never pollutes the next batch.
    for batch_folder, envelope in batches:
        try:
            report = render_report(envelope, batch_folder / "report.html", base_dir=batch_folder)
            print(f"Report: {report}", flush=True)
            if args.open:
                webbrowser.open(report.as_uri())
        except (OSError, ValueError, TypeError, KeyError) as exc:
            reports_ok = False
            print(f"Saved JSON, but report generation failed: {exc}", file=sys.stderr)
    return 0 if summary["success"] and reports_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
