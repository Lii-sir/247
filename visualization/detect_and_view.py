"""Optional standalone detection client; leaves the Bridge and algorithms untouched."""
from __future__ import annotations

import argparse
import json
import mmap
import os
import queue
import subprocess
import sys
import threading
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
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


def algorithm_request(score_mode: str | None, score_pool_kernel: int | None) -> dict:
    manifest = json.loads((ROOT / "config/config.json").read_text(encoding="utf-8-sig"))
    first_path = (ROOT / "config" / manifest["config_2d"]).resolve()
    second_path = (ROOT / "config" / manifest["config_3d"]).resolve()
    first = json.loads(first_path.read_text(encoding="utf-8-sig"))
    second = json.loads(second_path.read_text(encoding="utf-8-sig"))
    combined = {**first, **{k: v for k, v in second.items() if k != "model_path"}}
    chip_config = combined.get("chip_anomaly_2d", {}).get("parameters", {})
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=600, help="Per-request timeout in seconds")
    parser.add_argument("--score-mode", choices=("pool+top",), default=None)
    parser.add_argument("--score-pool-kernel", type=int, default=None)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The existing algorithm transport uses Windows named shared memory.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
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
        init_req = algorithm_request(args.score_mode, args.score_pool_kernel)
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
    try:
        client = ProtocolClient(args.python.resolve(), ROOT / "python/transport/bridge_server.py", folder / "python.log")
        response = client.request({"cmd": "init"}, args.timeout)
        if not response.get("ok"):
            envelope["response"] = response
            raise RuntimeError(response.get("error") or "Protocol init failed")
        print("Loading existing models (CUDA required)...", flush=True)
        response = client.request(init_req, args.timeout)
        envelope["response"] = response
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "Algorithm initialization failed")
        if args.mode == "stream":
            if args.parallel:
                print("Parallel stream: submit 10 2D pieces and 1 3D board scan...", flush=True)
                with ExitStack() as stack, ThreadPoolExecutor(max_workers=11) as pool:
                    descriptors = [mapped_file(stack, path, "2d", index)
                                   for index, path in enumerate(paths, 1)]
                    depth_descriptor = mapped_file(stack, depth, "3d")
                    depth_future = pool.submit(
                        client.request,
                        {"cmd": "detect_3d_board", "stage": "3d",
                         "board_id": board_id, **depth_descriptor},
                        args.timeout)
                    piece_futures = [pool.submit(
                        client.request,
                        {"cmd": "detect_2d_piece", "stage": "2d",
                         "board_id": board_id, **descriptor},
                        args.timeout)
                        for descriptor in descriptors]
                    piece_responses = [future.result() for future in piece_futures]
                    response = depth_future.result()
                envelope["responses_2d"] = sorted(
                    piece_responses, key=lambda item: int(item.get("piece_id") or 0))
            else:
                for index, path in enumerate(paths, 1):
                    print(f"2D piece {index}/10: {path.name}", flush=True)
                    with ExitStack() as stack:
                        descriptor = mapped_file(stack, path, "2d", index)
                        piece_response = client.request({"cmd": "detect_2d_piece", "stage": "2d", "board_id": board_id, **descriptor}, args.timeout)
                    envelope["responses_2d"].append(piece_response)
                    if not piece_response.get("ok"):
                        raise RuntimeError(piece_response.get("error") or f"2D piece {index} failed")
                print("3D board: segmentation, registration and height measurement...", flush=True)
                with ExitStack() as stack:
                    response = client.request({"cmd": "detect_3d_board", "stage": "3d", "board_id": board_id,
                                               **mapped_file(stack, depth, "3d")}, args.timeout)
        else:
            print("Batch board detection (same algorithm path as Bridge.Demo)...", flush=True)
            with ExitStack() as stack:
                response = client.request({"cmd": "detect_board", "stage": "board", "batch_id": board_id,
                                           "images_2d": [mapped_file(stack, p, "2d", i + 1) for i, p in enumerate(paths)],
                                           "image_3d": mapped_file(stack, depth, "3d")}, args.timeout)
        envelope["response"] = response
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"Detection did not complete: {exc}", file=sys.stderr)
        if envelope["response"].get("ok") or not envelope["response"]:
            envelope["response"] = {"ok": False, "board_id": board_id, "error": str(exc)}
    finally:
        if client is not None:
            client.close()
    (folder / "result.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        report = render_report(envelope, folder / "report.html", base_dir=folder)
        print(f"JSON: {folder / 'result.json'}\nReport: {report}\nLog: {folder / 'python.log'}")
        if args.open:
            webbrowser.open(report.as_uri())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Saved JSON, but report generation failed: {exc}", file=sys.stderr)
        return 1
    return 0 if envelope["response"].get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
