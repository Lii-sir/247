"""Local full-detection validation UI with explicit board/piece pairing."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from urllib.parse import urlsplit, parse_qs
from urllib.request import Request, urlopen
import uuid
import webbrowser
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SERVICE_NAME = "detection-validation-lab"
MAX_REQUEST_INTERVAL_MS = 3600000
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "python"))
from detect_and_view import ProtocolClient, algorithm_request, mapped_file, save_timing_csv
from generate_report import render_report
from algorithm.tilt_correction import production_pitch_parameters


def board_number(value):
    value = str("" if value is None else value).strip()
    if not value or len(value) > 128 or any(ord(c) < 32 for c in value):
        raise ValueError("板号不能为空，且不得超过128个字符")
    return str(int(value)) if value.isascii() and value.isdigit() else value


def guess_numbers(name, kind):
    stem = Path(name).stem
    if kind == "3d":
        return dict(board_id=stem, piece_id=None)
    match = re.fullmatch(r"(.+)_(\d+)", stem)
    return dict(board_id=match[1], piece_id=int(match[2])) if match else dict(board_id="", piece_id=None)


def submission_options(submission_mode="concurrent", request_interval_ms=0):
    if not isinstance(submission_mode, str) or submission_mode not in {"concurrent", "paced"}:
        raise ValueError("提交模式须为concurrent或paced")
    if (isinstance(request_interval_ms, bool) or not isinstance(request_interval_ms, (int, float))
            or not math.isfinite(request_interval_ms) or not 0 <= request_interval_ms <= MAX_REQUEST_INTERVAL_MS):
        raise ValueError(f"请求间隔须为0～{MAX_REQUEST_INTERVAL_MS}毫秒的有限数值")
    return dict(submission_mode=submission_mode,
                request_interval_ms=float(request_interval_ms) if submission_mode == "paced" else 0.)


def board_inputs(rows, board_id, mode, count):
    if mode not in {"joint", "2d"}:
        raise ValueError("验证方式须为joint或2d")
    board_id = board_number(board_id)
    if not isinstance(rows, list) or len(rows) > 1000:
        raise ValueError("文件清单无效或超过1000行")
    selected = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("文件清单中存在无效记录")
        try:
            same = board_number(row.get("board_id")) == board_id
        except ValueError:
            same = False
        if same:
            selected.append(row)
    images, depths = {}, []
    for row in selected:
        kind = row.get("kind")
        if kind not in {"2d", "3d"}:
            raise ValueError("未知图像类型")
        if mode == "2d" and kind == "3d":
            continue
        path = Path(str(row.get("path", ""))).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"文件不存在或为空：{row.get('name', path.name)}")
        if kind == "3d":
            if path.suffix.lower() != ".png":
                raise ValueError("正式3D接口需要原始单通道深度PNG")
            depths.append({**row, "path": str(path)})
        else:
            piece = row.get("piece_id")
            if isinstance(piece, bool) or not isinstance(piece, int) or not 1 <= piece <= count:
                raise ValueError(f"片号须为1～{count}的整数：{row.get('name', path.name)}")
            if piece in images:
                raise ValueError(f"板{board_id}的第{piece}片重复，请移除或修改编号")
            images[piece] = {**row, "path": str(path), "piece_id": piece}
    if not images:
        raise ValueError(f"板{board_id}没有2D片图")
    if mode == "joint":
        missing = sorted(set(range(1, count+1)) - images.keys())
        if missing:
            raise ValueError("联合验证缺少2D片号：" + ", ".join(map(str, missing)))
        if len(depths) != 1:
            raise ValueError(f"联合验证每板需要恰好一张3D整板图，目前{len(depths)}张")
    return board_id, [images[p] for p in sorted(images)], depths[0] if depths and mode == "joint" else None


class Runner:
    def __init__(self, folder, client_factory=ProtocolClient, init_builder=algorithm_request,
                 mapper=mapped_file, reporter=render_report):
        self.folder = Path(folder)
        self.client_factory, self.init_builder = client_factory, init_builder
        self.mapper, self.reporter = mapper, reporter
        self.client = None
        self.fingerprint = None
        self._state_lock = threading.RLock()
        self._stopping = threading.Event()

    def close(self, *, force=False):
        with self._state_lock:
            if self.client is not None:
                try:
                    if force:
                        self.client.close(force=True)
                    else:
                        self.client.close()
                finally:
                    self.client = None
                    self.fingerprint = None

    def stop(self):
        with self._state_lock:
            self._stopping.set()
            self.close(force=True)

    def _ensure(self, request, progress, *, board_timeout_seconds=None):
        stamps = {p: [Path(p).stat().st_mtime_ns, Path(p).stat().st_size]
                  for p in request.get("model_path", {}).values()}
        overrides = {"BRIDGE_BOARD_TIMEOUT_SECONDS": str(board_timeout_seconds)} if board_timeout_seconds is not None else {}
        fingerprint = hashlib.sha256(json.dumps([request, stamps, overrides], sort_keys=True).encode()).hexdigest()
        with self._state_lock:
            if self._stopping.is_set():
                raise RuntimeError("验证台正在停止")
            if self.client is not None and self.client.process.poll() is None and fingerprint == self.fingerprint:
                return dict(model_reused=True, process_pid=self.client.process.pid)
            self.close()
            progress("初始化模型（首次需要加载CUDA模型）")
            args = (Path(sys.executable), ROOT / "python/transport/bridge_server.py", self.folder / "python.log")
            self.client = self.client_factory(*args, **({"env_overrides": overrides} if overrides else {}))
            client = self.client
        for req in ({"cmd": "init"}, request):
            reply = client.request(req, 900)
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error") or reply.get("message") or "模型初始化失败")
        with self._state_lock:
            if self._stopping.is_set():
                raise RuntimeError("验证台正在停止")
            self.fingerprint = fingerprint
        return dict(model_reused=False, process_pid=client.process.pid)

    def run(self, rows, board_id, mode, count, folder, progress, *,
            submission_mode="concurrent", request_interval_ms=0):
        options = submission_options(submission_mode, request_interval_ms)
        board_id, images, depth = board_inputs(rows, board_id, mode, count)
        folder = Path(folder)
        envelope = dict(viewer_version=1, mode="stream", board_id=board_id,
                        title=f"板 {board_id} · {'联合检测' if mode == 'joint' else '2D验证'}",
                        images_2d=[dict(piece_id=r["piece_id"], path=r["path"], name=r.get("name")) for r in images],
                        responses_2d=[], response={}, warnings=[],
                        submission={**options, "requests": [], "planned_count": len(images)+bool(depth),
                                    "interval_basis": "minimum_start_to_start_no_response_wait"})
        if depth:
            envelope["image_3d"] = depth["path"]
        envelope["title"] += (f" · 按间隔提交 {options['request_interval_ms']:g}ms"
                              if options["submission_mode"] == "paced" else " · 并发提交")
        started, startup, client, timeline_start, release_sent = time.perf_counter(), {}, None, 0, False
        first_error = None
        try:
            if self._stopping.is_set():
                raise RuntimeError("验证台正在停止")
            progress("检查图像格式与深度位数")
            for row in images + ([depth] if depth else []):
                with Image.open(row["path"]) as image:
                    if row["kind"] == "3d" and (image.format != "PNG" or image.mode not in {"I", "I;16", "I;16B", "I;16L"}):
                        raise ValueError("3D须为原始16位单通道深度PNG，不能使用彩色图或8位截图")
                    image.verify()
            init = self.init_builder(None, None)
            config = json.loads(init["config_text"])
            actual_count = int(config["common_3d"]["parameters"]["input"]["piece_count"])
            if actual_count != count:
                raise ValueError("配置片数已变，请重新加载配置并检查文件清单")
            if mode == "2d":
                # Local service copy only; 2D validation must not depend on depth calibration.
                pitch = config.get("smartray_pitch_3d", {}).get("parameters")
                if pitch is not None:
                    pitch["enabled_3d"] = False
                threading_3d = config.get("common_3d", {}).get("parameters", {}).get("threading", {})
                affinity = threading_3d.get("registration_affinity")
                if isinstance(affinity, dict):
                    # No 3D request will use registration in this mode. Keep
                    # its inherited default for the active 2D pool, but do not
                    # reserve physical cores for idle registration workers.
                    threading_2d = config.setdefault("common_2d", {}).setdefault("parameters", {}).setdefault("threading", {})
                    threading_2d.setdefault("cpu_process_affinity", dict(enabled=affinity.get("enabled", False),
                        physical_cores=threading_2d.get("cpu_process_workers", 2)))
                    affinity["enabled"] = False
                joint_affinity = threading_3d.get("joint_affinity")
                if isinstance(joint_affinity, dict):
                    joint_affinity["enabled"] = False
                init["config_text"] = json.dumps(config, ensure_ascii=True)
            height = config["height_measurement_3d"]["parameters"]
            production_pitch_parameters(config, float(height.get("gray_offset", 32768)), float(height.get("vertical_resolution_mm", .00105)))
            cache_timeout = float(os.environ.get("BRIDGE_BOARD_TIMEOUT_SECONDS", "90"))
            if not math.isfinite(cache_timeout) or cache_timeout <= 0:
                raise ValueError("BRIDGE_BOARD_TIMEOUT_SECONDS须为正的有限数值")
            schedule_span = (len(images)+bool(depth)-1)*options["request_interval_ms"]/1000
            needed_timeout = max(cache_timeout, schedule_span+30.) if schedule_span > 0 else cache_timeout
            startup = self._ensure(init, progress,
                board_timeout_seconds=needed_timeout if needed_timeout > cache_timeout else None)
            envelope["submission"]["board_cache_timeout_seconds"] = needed_timeout
            with self._state_lock:
                if self._stopping.is_set():
                    raise RuntimeError("验证台正在停止")
                client = self.client
            timeline_start = len(client.timeline)
            progress("准备共享内存")
            with ExitStack() as stack, ThreadPoolExecutor(max_workers=min(len(images)+1, 16)) as pool:
                inputs = [(r, self.mapper(stack, Path(r["path"]), "2d", r["piece_id"])) for r in images]
                depth_request = (dict(cmd="detect_3d_board", stage="3d", board_id=board_id,
                    **self.mapper(stack, Path(depth["path"]), "3d")) if depth else None)
                requests = [dict(cmd="detect_2d_piece", stage="2d", board_id=board_id, **desc) for row, desc in inputs]
                if options["submission_mode"] == "paced":
                    if depth_request:
                        requests.append(depth_request)
                elif depth_request:
                    requests.insert(0, depth_request)
                futures, consumed = {}, set()
                finished = 0
                submission_started = time.perf_counter()
                submission_lock, last_sent = threading.Lock(), [None]

                def release():
                    nonlocal release_sent
                    if release_sent or self._stopping.is_set():
                        return
                    release_sent = True
                    try:
                        client.request(dict(cmd="release_board", board_id=board_id), 15)
                    except Exception:
                        self.close(force=True)

                def consume(future):
                    nonlocal finished, first_error
                    if future in consumed:
                        return
                    consumed.add(future)
                    reply = future.result()
                    piece = futures[future]
                    if piece is None:
                        envelope["response"] = reply
                    else:
                        envelope["responses_2d"].append(reply)
                        finished += 1
                    progress(f"已完成2D {finished}/{len(images)}" + (" · 等待/处理整板3D" if depth else ""))
                    if not reply.get("ok") and first_error is None:
                        details = [str(reply[k]) for k in ("error", "message") if reply.get(k)]
                        first_error = (f"第{piece}片：" if piece is not None else "整板3D：") + "；".join(dict.fromkeys(details) or ["请求失败"])
                        release()

                def collect_ready():
                    for future in list(futures):
                        if future.done() and future not in consumed:
                            consume(future)

                def wait_until(deadline, sent_event=None, sender=None):
                    while True:
                        if self._stopping.is_set():
                            raise RuntimeError("验证台正在停止")
                        collect_ready()
                        if first_error is not None:
                            return False
                        remaining = deadline-time.perf_counter() if deadline is not None else .05
                        if sent_event is not None:
                            if sent_event.is_set():
                                return True
                            if sender.done():
                                sender.result()
                                raise RuntimeError("请求未确认发送")
                        elif remaining <= 0:
                            return True
                        if sent_event is not None:
                            sent_event.wait(.01)
                        else:
                            self._stopping.wait(min(.05, max(.001, remaining)))

                previous_sent = None
                try:
                    for request in requests:
                        if options["submission_mode"] == "paced" and previous_sent is not None:
                            if not wait_until(previous_sent+options["request_interval_ms"]/1000):
                                break
                        collect_ready()
                        if first_error is not None:
                            break
                        if self._stopping.is_set():
                            raise RuntimeError("验证台正在停止")
                        event, sent_at = threading.Event(), []
                        def on_sent(request_id, sent, req=request, event=event, sent_at=sent_at):
                            with submission_lock:
                                records = envelope["submission"]["requests"]
                                records.append(dict(order=len(records)+1, cmd=req["cmd"], piece_id=req.get("piece_id"),
                                    request_id=request_id, sent_ms=round((sent-submission_started)*1000, 3),
                                    gap_ms=round((sent-last_sent[0])*1000, 3) if last_sent[0] is not None else None))
                                last_sent[0] = sent
                                sent_at.append(sent)
                            event.set()
                        future = pool.submit(client.request, request, 900, on_sent=on_sent)
                        futures[future] = request.get("piece_id")
                        if options["submission_mode"] == "paced":
                            if not wait_until(None, event, future):
                                break
                            previous_sent = sent_at[0]
                            progress(f"已发送 {'整板3D' if request['cmd']=='detect_3d_board' else '2D第'+str(request['piece_id'])+'片'} · 间隔{options['request_interval_ms']:g}ms")
                    for future in as_completed(futures):
                        consume(future)
                except Exception:
                    release()
                    raise
                envelope["responses_2d"].sort(key=lambda r: r.get("piece_id", 0))
            if mode == "2d":
                envelope["response"] = dict(ok=all(r.get("ok") for r in envelope["responses_2d"]),
                                             stage="2d", board_id=board_id, cmd="validate_2d")
            if first_error:
                envelope["response"]["ok"] = False
                envelope["response"]["error"] = first_error
                envelope["response"]["message"] = "验证因上述错误停止；缓存释放属于后续清理。"
        except Exception as exc:
            envelope["response"] = dict(ok=False, board_id=board_id, error=first_error or str(exc))
        finally:
            if client is not None:
                envelope["timeline"] = client.timeline[timeline_start:]
            envelope["submission"]["aborted"] = len(envelope["submission"]["requests"]) < envelope["submission"]["planned_count"]
            if client is not None and not release_sent and not self._stopping.is_set():
                try:
                    client.request(dict(cmd="release_board", board_id=board_id), 15)
                except (OSError, RuntimeError, TimeoutError):
                    self.close()
        envelope["benchmark"] = dict(total_wall_ms=round((time.perf_counter()-started)*1000, 2), **startup)
        (folder / "result.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        save_timing_csv(envelope, folder / "timings.csv")
        if self._stopping.is_set():
            return dict(ok=False, product_is_ok=None, board_id=board_id,
                        error="验证台已停止，本次验证已中断", output_dir=str(folder))
        progress("生成交互报告")
        self.reporter(envelope, folder / "report.html", base_dir=folder, force=True)
        if self.client is not None:
            self.client.log.flush()
        if (self.folder / "python.log").exists():
            shutil.copyfile(self.folder / "python.log", folder / "python.log")
        ok = bool(envelope["response"].get("ok"))
        board = envelope["response"].get("results_board", {})
        product = board.get("is_ok") if mode == "joint" else all(r.get("results_2d", {}).get("is_ok") is True for r in envelope["responses_2d"])
        return dict(ok=ok, product_is_ok=product if ok else None, board_id=board_id,
                    **options,
                    pieces=len(envelope["responses_2d"]), model_reused=startup.get("model_reused", False),
                    error=envelope["response"].get("error"), output_dir=str(folder),
                    total_wall_ms=envelope["benchmark"]["total_wall_ms"])


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, port, output):
        super().__init__(("127.0.0.1", port), Handler)
        self.folder = Path(output).resolve() / ("session-" + uuid.uuid4().hex[:10])
        (self.folder / "uploads").mkdir(parents=True)
        self.assets = {f: (HERE / f).read_bytes() for f in ("index.html", "app.js", "pairing.js", "style.css")}
        self.runner, self.jobs = Runner(self.folder), {}
        self.pool, self.lock = ThreadPoolExecutor(1), threading.Lock()

    def server_close(self):
        super().server_close()
        # Interrupt pending protocol waits BEFORE joining the job executor.
        # Joining first can otherwise wait for the full 900-second timeout.
        try:
            self.runner.stop()
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)


def defaults():
    request = algorithm_request(None, None)
    cfg = json.loads(request["config_text"])
    count = int(cfg["common_3d"]["parameters"]["input"]["piece_count"])
    samples = [dict(kind="2d", name=p.name, path=str(p), board_id="demo", piece_id=i)
               for i, p in enumerate(sorted((ROOT / "demodatasets/2D").glob("*.png")), 1)]
    samples += [dict(kind="3d", name=p.name, path=str(p), board_id="demo", piece_id=None)
                for p in sorted((ROOT / "demodatasets/3D").glob("*.png"))]
    return dict(piece_count=count, samples=samples, pitch=cfg.get("smartray_pitch_3d", {}).get("parameters", {}),
                max_request_interval_ms=MAX_REQUEST_INTERVAL_MS)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def allowed(self):
        host = f"127.0.0.1:{self.server.server_port}"
        return self.headers.get("Host") == host and self.headers.get("Origin", "http://"+host) == "http://"+host

    def send(self, code, value, mime="application/json; charset=utf-8"):
        data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        for k, v in (("Content-Type", mime), ("Content-Length", str(len(data))), ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff")):
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if not self.allowed():
            return self.send(403, {"error": "仅允许本机页面访问"})
        path = urlsplit(self.path).path
        try:
            if path == "/api/defaults":
                return self.send(200, defaults())
            if path == "/api/health":
                return self.send(200, dict(service=SERVICE_NAME))
            if path == "/api/history":
                with self.server.lock:
                    rows = [dict(id=key, **job) for key, job in reversed(list(self.server.jobs.items()))]
                return self.send(200, rows)
            if path.startswith("/api/jobs/"):
                with self.server.lock:
                    job = dict(self.server.jobs.get(path.rsplit("/", 1)[-1], {}))
                return self.send(200 if job else 404, job or {"error": "记录不存在"})
            if path.startswith("/runs/"):
                parts = path.split("/")
                if len(parts) == 4 and re.fullmatch(r"[a-f0-9]{32}", parts[2]) and parts[3] in {"report.html", "result.json", "timings.csv", "python.log"}:
                    file = self.server.folder / parts[2] / parts[3]
                    mime = {".html": "text/html", ".json": "application/json", ".csv": "text/csv", ".log": "text/plain"}[file.suffix]
                    return self.send(200, file.read_bytes(), mime+"; charset=utf-8")
            asset = {"/": "index.html", "/app.js": "app.js", "/pairing.js": "pairing.js", "/style.css": "style.css"}.get(path)
            if asset:
                return self.send(200, self.server.assets[asset], {".html":"text/html", ".js":"text/javascript", ".css":"text/css"}[Path(asset).suffix]+"; charset=utf-8")
            self.send(404, {"error": "文件不存在"})
        except (OSError, ValueError) as exc:
            self.send(400, {"error": str(exc)})

    def do_POST(self):
        if not self.allowed():
            return self.send(403, {"error": "仅允许本机页面访问"})
        try:
            parsed = urlsplit(self.path)
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 160*1024*1024:
                raise ValueError("请求为空或超过160MB")
            if parsed.path == "/api/upload":
                query = parse_qs(parsed.query);kind = query.get("kind", [""])[0];name = Path(query.get("name", [""])[0]).name
                if kind not in {"2d", "3d"} or Path(name).suffix.lower() not in ({".png"} if kind == "3d" else {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}):
                    raise ValueError("请选择支持的2D图像或3D深度PNG")
                token = uuid.uuid4().hex
                file = self.server.folder / "uploads" / (token + Path(name).suffix.lower())
                file.write_bytes(self.rfile.read(size))
                return self.send(200, dict(id=token, kind=kind, name=name, path=str(file), **guess_numbers(name, kind)))
            if parsed.path == "/api/shutdown":
                self.rfile.read(size)
                self.send(200, dict(stopping=True))
                # shutdown() must run outside the serve_forever thread.
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if parsed.path != "/api/run":
                return self.send(404, {"error": "接口不存在"})
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("验证请求须为JSON对象")
            options = submission_options(request.get("submission_mode", "concurrent"), request.get("request_interval_ms", 0))
            count = defaults()["piece_count"]
            board, images, depth = board_inputs(request.get("files"), request.get("board_id"), request.get("mode"), count)
            rows = images + ([depth] if depth else [])
            with self.server.lock:
                if any(j["status"] == "running" for j in self.server.jobs.values()):
                    return self.send(409, {"error": "已有验证运行中，请等待完成"})
                if len(self.server.jobs) >= 32:
                    self.server.jobs.pop(next(iter(self.server.jobs)))
                key = uuid.uuid4().hex
                self.server.jobs[key] = dict(status="running", phase="准备验证", board_id=board)
            folder = self.server.folder / key;folder.mkdir()
            def update(**values):
                with self.server.lock:
                    self.server.jobs[key].update(values)
            def execute():
                try:
                    result = self.server.runner.run(rows, board, request["mode"], count, folder,
                        lambda phase:update(phase=phase), **options)
                    update(status="done" if result["ok"] else "failed", result=result, error=result.get("error"))
                except Exception as exc:
                    update(status="failed", error=str(exc))
            self.server.pool.submit(execute)
            self.send(202, {"id":key})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.send(400, {"error":str(exc)})


def service_running(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
            value = json.load(response)
        return isinstance(value, dict) and value.get("service") == SERVICE_NAME
    except (OSError, ValueError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--output-dir", type=Path, default=ROOT/"visualization/output/validation-lab")
    p.add_argument("--open", action="store_true")
    p.add_argument("--stop", action="store_true", help="停止指定端口的验证台及其检测进程")
    args = p.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    def reopen_existing():
        print(f"验证台已在运行，复用：{url}", flush=True)
        if args.open:
            webbrowser.open(url)
    running = service_running(args.port)
    if args.stop:
        if not running:
            print(f"未找到运行中的验证台：{url}", flush=True)
            return
        with urlopen(Request(url + "/api/shutdown", data=b"{}",
                             headers={"Content-Type": "application/json"}), timeout=5) as response:
            json.load(response)
        print("已发送停止命令，验证台及其检测进程将退出。", flush=True)
        return
    if running:
        reopen_existing()
        return
    try:
        server = Server(args.port, args.output_dir)
    except OSError as exc:
        # Two launch commands can both pass the first check before one binds.
        if service_running(args.port):
            reopen_existing()
            return
        raise SystemExit(f"无法启动验证台：{exc}。请检查端口或用 --port 指定其他端口。") from exc
    with server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Detection validation: {url}", flush=True)
        print("按 Ctrl+C 停止；也可在另一终端运行同一命令并加 --stop。", flush=True)
        if args.open:webbrowser.open(url)
        try:server.serve_forever()
        except KeyboardInterrupt:pass


if __name__ == "__main__":main()
