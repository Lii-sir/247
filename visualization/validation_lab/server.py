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
PITCH_3D = "pitch_piece_3d"
PITCH_START = "pitch_start"
MANUAL_MODES = {"manual_2d", "manual_piece_3d", PITCH_3D}
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "python"))
from detect_and_view import ProtocolClient, algorithm_request, mapped_depth_piece, mapped_file, save_timing_csv, summarize_2d_latency
from generate_report import render_report
from algorithm.runtime.config import feature_parameters
from algorithm.vision3d.depth.depth_preprocessing import preprocessing_parameters
from algorithm.vision3d.depth.segment import SegmentationConfig
from validation_lab.pitch_stream import PitchPreparations, PitchTrigger


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


def board_inputs(rows, board_id, mode, count, *, piece_id=None):
    if not isinstance(mode, str) or mode not in {"2d", PITCH_START, *MANUAL_MODES}:
        raise ValueError("未知验证方式")
    if mode in MANUAL_MODES and (type(piece_id) is not int or not 1 <= piece_id <= count):
        raise ValueError(f"片号须为1～{count}的整数")
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
        if mode in {"2d", "manual_2d"} and kind == "3d":
            continue
        if mode == PITCH_START and kind == "2d":
            continue
        if mode in MANUAL_MODES and kind == "2d" and row.get("piece_id") != piece_id:
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
    needs_depth = mode in {"manual_piece_3d", PITCH_3D, PITCH_START}
    if not images and not needs_depth:
        raise ValueError(f"板{board_id}没有2D片图")
    if needs_depth and len(depths) != 1:
        raise ValueError(f"每板需要恰好一张3D整板图，目前{len(depths)}张")
    return board_id, [images[p] for p in sorted(images)], depths[0] if needs_depth else None


def fit_manual_affinity(config):
    """Keep model settings while allowing manual tests on a smaller Windows host."""
    if os.name != "nt":
        return None
    two_d = config.get("common_2d", {}).get("parameters", {}).get("threading", {})
    three_d = config["common_3d"]["parameters"].get("threading", {})
    affinity = [settings[key] for settings, key in (
        (two_d, "cpu_process_affinity"), (three_d, "registration_affinity"), (three_d, "joint_affinity"))
        if isinstance(settings.get(key), dict) and settings[key].get("enabled") is True]
    if not affinity:
        return None
    from algorithm.execution.cpu_partition import current_cpus, physical_cores

    allowed = set(current_cpus())
    available = sum(bool(allowed.intersection(core["logical_cpus"])) for core in physical_cores())
    required = max(1, three_d.get("main_reserved_physical_cores", 0))
    required += sum(group.get("physical_cores", 0) for group in affinity)
    if required <= available:
        return None
    for group in affinity:
        group["enabled"] = False
    three_d["main_reserved_physical_cores"] = 0
    return f"本机可用物理核 {available} 个，正式配置需要 {required} 个。本次手动验证使用不绑核的进程池。"


class Runner:
    def __init__(self, folder, client_factory=ProtocolClient, init_builder=algorithm_request,
                 mapper=mapped_file, reporter=render_report):
        self.folder = Path(folder)
        self.client_factory, self.init_builder = client_factory, init_builder
        self.mapper, self.reporter = mapper, reporter
        self.client = None
        self.fingerprint = None
        self._startup_threading = None
        self._pitch = PitchPreparations()
        self._pitch_triggers = {}
        self._state_lock = threading.RLock()
        self._submission_lock = threading.Lock()
        self._stopping = threading.Event()
        self._manual_boards = {}
        self._manual_active = 0

    def close(self, *, force=False):
        with self._state_lock:
            try:
                if self.client is not None:
                    if force:
                        self.client.close(force=True)
                    else:
                        self.client.close()
            finally:
                self.client = None
                self.fingerprint = None
                self._startup_threading = None
                self._manual_boards.clear()

    def stop(self):
        with self._state_lock:
            self._stopping.set()
            self.close(force=True)
            keys = list(self._pitch_triggers)
        for key in keys:
            self._fail_pitch_trigger(key, "验证台正在停止")
        self._pitch.close()

    def _ensure(self, request, progress, *, board_timeout_seconds=None):
        stamps = {p: [Path(p).stat().st_mtime_ns, Path(p).stat().st_size]
                  for p in request.get("model_path", {}).values()}
        overrides = {"BRIDGE_BOARD_TIMEOUT_SECONDS": str(board_timeout_seconds)} if board_timeout_seconds is not None else {}
        fingerprint = hashlib.sha256(json.dumps([request, stamps, overrides], sort_keys=True).encode()).hexdigest()
        with self._state_lock:
            if self._stopping.is_set():
                raise RuntimeError("验证台正在停止")
            if self.client is not None and self.client.process.poll() is None and fingerprint == self.fingerprint:
                return dict(model_reused=True, process_pid=self.client.process.pid,
                            threading=self._startup_threading)
            if self._manual_active:
                raise RuntimeError("检测配置已变更或进程已退出，请等已提交的请求结束后重试")
            self.close()
            progress("初始化模型（首次需要加载CUDA模型）")
            args = (Path(sys.executable), ROOT / "python/transport/bridge_server.py", self.folder / "python.log")
            self.client = self.client_factory(*args, **({"env_overrides": overrides} if overrides else {}))
            client = self.client
        for req in ({"cmd": "init"}, request):
            reply = client.request(req, 900)
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error") or reply.get("message") or "模型初始化失败")
            if req['cmd'] == 'init_algorithm':
                self._startup_threading = reply.get('threading')
        with self._state_lock:
            if self._stopping.is_set():
                raise RuntimeError("验证台正在停止")
            self.fingerprint = fingerprint
        return dict(model_reused=False, process_pid=client.process.pid,
                    threading=self._startup_threading)

    def _pitch_input(self, path, config, settings):
        segmentation = SegmentationConfig.from_dict(feature_parameters(config, "segmentation_3d"))
        return segmentation, self._pitch.key(path, segmentation, settings)

    def start_pitch(self, board_id, depth):
        config = json.loads(self.init_builder(None, None)["config_text"])
        settings = preprocessing_parameters(config)
        segmentation, depth_key = self._pitch_input(depth["path"], config, settings)
        with self._state_lock:
            if self._stopping.is_set():
                raise RuntimeError("验证台正在停止")
            started_at = time.time()
            pieces, reused = self._pitch.start(board_id, depth_key, segmentation, settings)
            watching = []
            for key, trigger in self._pitch_triggers.items():
                if trigger.board_id == board_id and trigger.depth_key == depth_key and trigger.depth is None:
                    trigger.depth = pieces[trigger.piece_id]
                    if trigger.timer.ident is None:
                        trigger.timer.start()
                    watching.append((key, trigger.depth))
        for key, future in watching:
            self._watch_pitch_piece(key, future)
        return dict(board_id=board_id, started_at=started_at, reused=reused)

    def register_pitch_trigger(self, key, board_id, piece_id, images, depth, ready, dispatch, fail, *,
                               wait_for_pitch_start=False):
        if not images:
            raise ValueError("Pitch 就绪送检需要对应片的 2D 文件")
        config = json.loads(self.init_builder(None, None)["config_text"])
        _, depth_key = self._pitch_input(depth["path"], config, preprocessing_parameters(config))
        timeout = float(os.environ.get("BRIDGE_BOARD_TIMEOUT_SECONDS", "90"))
        timer = threading.Timer(timeout, self._fail_pitch_trigger, args=(key, "等待单片深度或 2D 提交超时"))
        timer.daemon = True
        image_path = images[0]["path"]
        with self._state_lock:
            if self._stopping.is_set():
                raise RuntimeError("验证台正在停止")
            state = self._manual_boards.get(board_id)
            available = state and (state["active"] or time.monotonic() - state["touched"] < timeout)
            record = state["images"].get(piece_id) if available else None
            already_sent = bool(record and record["sent"] and record["path"] == image_path)
            future = None if wait_for_pitch_start else self._pitch.piece(board_id, depth_key, piece_id)
            self._pitch_triggers[key] = PitchTrigger(
                board_id, piece_id, image_path, depth_key, future, ready, dispatch, fail, timer, already_sent)
            if not wait_for_pitch_start:
                timer.start()
        if future is not None:
            self._watch_pitch_piece(key, future)

    def _watch_pitch_piece(self, key, future):
        future.add_done_callback(lambda _: self._advance_pitch_trigger(key))

    def _piece_2d_sent(self, board_id, piece_id, record):
        with self._state_lock:
            record["sent"] = True
            keys = [key for key, trigger in self._pitch_triggers.items()
                    if (trigger.board_id, trigger.piece_id, trigger.image_path) ==
                    (board_id, piece_id, record["path"])]
            for key in keys:
                self._pitch_triggers[key].two_d_sent = True
        for key in keys:
            self._advance_pitch_trigger(key)

    def _advance_pitch_trigger(self, key):
        with self._state_lock:
            trigger = self._pitch_triggers.get(key)
            if trigger is None or trigger.depth is None or not trigger.depth.done():
                return
            error = trigger.depth.exception()
            can_dispatch = trigger.two_d_sent and error is None
            if error is not None or can_dispatch:
                self._pitch_triggers.pop(key)
                trigger.timer.cancel()
        if error is not None:
            trigger.fail(str(error))
            return
        piece = trigger.depth.result()
        trigger.ready(piece.ready_at)
        if can_dispatch:
            trigger.dispatch(piece)

    def _fail_pitch_trigger(self, key, reason):
        with self._state_lock:
            trigger = self._pitch_triggers.pop(key, None)
        if trigger is not None:
            trigger.timer.cancel()
            trigger.fail(reason)

    def finish_pitch_submissions(self, keys, stopped):
        with self._state_lock:
            pending = [key for key in keys if key in self._pitch_triggers and
                       (stopped or self._pitch_triggers[key].depth is None or not self._pitch_triggers[key].two_d_sent)]
            triggers = [self._pitch_triggers.pop(key) for key in pending]
        for trigger in triggers:
            trigger.timer.cancel()
            if stopped:
                trigger.fail("后续送检已停止")
            elif trigger.depth is None:
                trigger.fail("该板尚未启动 Pitch")
            else:
                trigger.fail("该片 2D 请求尚未提交")

    def _manual_piece_depth(self, path, piece_id, config, settings, folder, board_id):
        _, depth_key = self._pitch_input(path, config, settings)
        future = self._pitch.piece(board_id, depth_key, piece_id)
        if future is None:
            raise ValueError("请先执行该板的「启动 Pitch」节点")
        piece = future.result()
        asset = folder / "piece-depth.png"
        Image.fromarray(piece.pixels).save(asset)
        return asset, piece.correction

    def run_manual(self, rows, board_id, mode, piece_id, count, folder, progress, *,
                   on_sent=None, on_finished=None, prepared_piece=None):
        board_id, images, depth = board_inputs(rows, board_id, mode, count, piece_id=piece_id)
        if mode == "manual_2d":
            with Image.open(images[0]["path"]) as image:
                image.verify()
        init = self.init_builder(None, None)
        config = json.loads(init["config_text"])
        if int(config["common_3d"]["parameters"]["input"]["piece_count"]) != count:
            raise ValueError("配置片数已变，请刷新页面")
        settings = preprocessing_parameters(config) if depth else None
        if prepared_piece is not None:
            segmentation = SegmentationConfig.from_dict(feature_parameters(config, "segmentation_3d"))
            if prepared_piece.profile != self._pitch.profile(segmentation, settings):
                raise ValueError("Pitch 配置或分片范围已变更，请重新启动流程")
        affinity_note = fit_manual_affinity(config)
        config.pop("smartray_pitch_3d", None)
        init["config_text"] = json.dumps(config, ensure_ascii=True)
        idle_timeout = float(os.environ.get("BRIDGE_BOARD_TIMEOUT_SECONDS", "90"))
        chosen = {r["piece_id"]: r["path"] for r in images}
        command = "detect_2d_piece" if mode == "manual_2d" else "detect_3d_piece"
        label = "2D" if mode == "manual_2d" else "单片3D"
        title = f"板 {board_id} · 第{piece_id}片 · {label}"
        envelope = dict(viewer_version=1, mode="stream", board_id=board_id, title=title,
                        images_2d=images, responses_2d=[], warnings=[affinity_note] if affinity_note else [])
        reply, request_id = {}, None
        with ExitStack() as submission, ExitStack() as stack:
            # Serialize input preparation and pipe writes only. Response waits
            # and report generation belong to the individual jobs.
            submission.enter_context(self._submission_lock)
            startup = self._ensure(init, progress)
            with self._state_lock:
                if self._stopping.is_set():
                    raise RuntimeError("验证台正在停止")
                client = self.client
                now = time.monotonic()
                for key, cached in list(self._manual_boards.items()):
                    if not cached["active"] and now - cached["touched"] >= idle_timeout:
                        self._manual_boards.pop(key, None)
                state = self._manual_boards.setdefault(board_id, dict(images={}, active=0, touched=now))
                known = state["images"]
                record = known.get(piece_id)
                if mode != "manual_2d" and record is not None and record["path"] != chosen.get(piece_id):
                    raise ValueError("该片2D文件已变更，请重新提交对应2D")
                if mode == "manual_2d":
                    record = None
                state["active"] += 1
                self._manual_active += 1

            def finish_request():
                with self._state_lock:
                    self._manual_active -= 1
                    state["active"] -= 1
                    if self._manual_boards.get(board_id) is not state:
                        return
                    state["touched"] = time.monotonic()
                    remove = not reply.get("ok") if mode == "manual_2d" else reply.get("ok")
                    if record is not None and known.get(piece_id) is record:
                        if remove:
                            known.pop(piece_id, None)
                        elif mode == "manual_2d":
                            record["ready"] = True
                    if not known and not state["active"]:
                        self._manual_boards.pop(board_id, None)

            stack.callback(finish_request)
            progress(f"准备{label}输入")
            if mode == "manual_2d":
                descriptor = self.mapper(stack, Path(images[0]["path"]), "2d", piece_id)
            elif prepared_piece is not None:
                descriptor = mapped_depth_piece(stack, prepared_piece.pixels, piece_id, prepared_piece.correction)
            else:
                asset, metadata = self._manual_piece_depth(Path(depth["path"]), piece_id, config, settings, folder, board_id)
                descriptor = self.mapper(stack, asset, "3d", piece_id)
                descriptor.update(piece_id=piece_id, pitch_correction=metadata)
                envelope.update(image_3d=str(asset), image_3d_preprocessed=True)
            if self._stopping.is_set():
                raise RuntimeError("验证台正在停止")
            if mode == "manual_2d":
                with self._state_lock:
                    record = known[piece_id] = {"path": chosen[piece_id], "ready": False, "sent": False}

            def sent(identifier, sent_at):
                nonlocal request_id
                request_id = identifier
                if mode == "manual_2d":
                    self._piece_2d_sent(board_id, piece_id, record)
                submission.close()
                if on_sent is not None:
                    on_sent(sent_at)

            started = time.perf_counter()
            progress(f"等待{label}响应")
            received = False
            try:
                reply = client.request(dict(cmd=command, board_id=board_id, **descriptor), 900, on_sent=sent)
                received = True
            finally:
                if request_id is not None and on_finished is not None:
                    on_finished(time.perf_counter(), received)
        elapsed = (time.perf_counter() - started) * 1000
        artifact_errors = {}
        if prepared_piece is not None:
            asset = folder / "piece-depth.png"
            envelope["pitch_ready_at"] = prepared_piece.ready_at
            try:
                Image.fromarray(prepared_piece.pixels).save(asset)
                envelope.update(image_3d=str(asset), image_3d_preprocessed=True)
            except (OSError, ValueError) as exc:
                artifact_errors["piece-depth.png"] = f"深度预览保存失败：{exc}"
        with self._state_lock:
            ready_pieces = sorted(pid for pid, source in known.items() if source["ready"])
        timeline = [row for row in client.timeline if row.get("request_id") == request_id]
        envelope.update(response=reply, timeline=timeline,
                        submission=dict(submission_mode="manual", requests=timeline))
        (folder / "result.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            save_timing_csv(envelope, folder / "timings.csv")
        except Exception as exc:
            artifact_errors["timings.csv"] = f"耗时 CSV 导出失败：{exc}"
        try:
            self.reporter(envelope, folder / "report.html", base_dir=folder, force=True)
        except Exception as exc:
            artifact_errors["report.html"] = f"报告生成失败：{exc}"
        if (self.folder / "python.log").exists():
            try:
                shutil.copyfile(self.folder / "python.log", folder / "python.log")
            except OSError as exc:
                artifact_errors["python.log"] = f"检测日志复制失败：{exc}"
        detail = reply.get("results_2d") or reply.get("results_piece", {})
        latency = summarize_2d_latency(timeline, expected_count=1) if mode == "manual_2d" else None
        return dict(ok=bool(reply.get("ok")), product_is_ok=detail.get("is_ok"),
                    board_id=board_id, piece_id=piece_id, submission_mode="manual", action=label,
                     ready_piece_ids=ready_pieces, pieces=1,
                    total_wall_ms=elapsed, error=(reply.get("message") or reply.get("error")) if not reply.get("ok") else None,
                    two_d_latency=latency, skip_3d_reason=detail.get("skip_3d_reason"), note=affinity_note,
                    output_dir=str(folder), model_reused=startup.get("model_reused", False),
                    artifact_errors=artifact_errors)

    def release_manual(self, board_id):
        self._pitch.release(board_id)
        with self._state_lock:
            self._manual_boards.pop(board_id, None)
            keys = [key for key, trigger in self._pitch_triggers.items() if trigger.board_id == board_id]
        for key in keys:
            self._fail_pitch_trigger(key, "板缓存已释放")
        if self.client is not None and self.client.process.poll() is None:
            reply = self.client.request(dict(cmd="release_board", board_id=board_id), 15)
            if not reply.get("ok"):
                raise RuntimeError(reply.get("message") or "释放板缓存失败")

    def run(self, rows, board_id, mode, count, folder, progress, *,
            submission_mode="concurrent", request_interval_ms=0):
        options = submission_options(submission_mode, request_interval_ms)
        board_id, images, _ = board_inputs(rows, board_id, mode, count)
        folder = Path(folder)
        envelope = dict(viewer_version=1, mode="stream", board_id=board_id,
                         title=f"板 {board_id} · 2D验证",
                        images_2d=[dict(piece_id=r["piece_id"], path=r["path"], name=r.get("name")) for r in images],
                        responses_2d=[], response={}, warnings=[],
                         submission={**options, "requests": [], "planned_count": len(images),
                                    "interval_basis": "minimum_start_to_start_no_response_wait"})
        envelope["title"] += (f" · 按间隔提交 {options['request_interval_ms']:g}ms"
                              if options["submission_mode"] == "paced" else " · 并发提交")
        started, startup, client, timeline_start, release_sent = None, {}, None, 0, False
        first_error = None
        try:
            if self._stopping.is_set():
                raise RuntimeError("验证台正在停止")
            progress("检查图像格式")
            for row in images:
                with Image.open(row["path"]) as image:
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
            config.pop('smartray_pitch_3d', None)
            init['config_text'] = json.dumps(config, ensure_ascii=True)
            cache_timeout = float(os.environ.get("BRIDGE_BOARD_TIMEOUT_SECONDS", "90"))
            if not math.isfinite(cache_timeout) or cache_timeout <= 0:
                raise ValueError("BRIDGE_BOARD_TIMEOUT_SECONDS须为正的有限数值")
            schedule_span = (len(images)-1)*options["request_interval_ms"]/1000
            needed_timeout = max(cache_timeout, schedule_span+30.) if schedule_span > 0 else cache_timeout
            startup = self._ensure(init, progress,
                board_timeout_seconds=needed_timeout if needed_timeout > cache_timeout else None)
            envelope["submission"]["board_cache_timeout_seconds"] = needed_timeout
            with self._state_lock:
                if self._stopping.is_set():
                    raise RuntimeError("验证台正在停止")
            client = self.client
            if board_id in self._manual_boards:
                self.release_manual(board_id)
            timeline_start = len(client.timeline)
            progress("准备共享内存")
            with ExitStack() as stack, ThreadPoolExecutor(max_workers=min(len(images), 16)) as pool:
                inputs = [(r, self.mapper(stack, Path(r["path"]), "2d", r["piece_id"])) for r in images]
                started = time.perf_counter()
                requests = [dict(cmd="detect_2d_piece", stage="2d", board_id=board_id, **desc) for row, desc in inputs]
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
                    envelope["responses_2d"].append(reply)
                    finished += 1
                    progress(f"已完成2D {finished}/{len(images)}")
                    if not reply.get("ok") and first_error is None:
                        details = [str(reply[k]) for k in ("error", "message") if reply.get(k)]
                        first_error = f"第{piece}片：" + "；".join(dict.fromkeys(details) or ["请求失败"])
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
                            progress(f"已发送2D第{request['piece_id']}片 · 间隔{options['request_interval_ms']:g}ms")
                    for future in as_completed(futures):
                        consume(future)
                except Exception:
                    release()
                    raise
                envelope["responses_2d"].sort(key=lambda r: r.get("piece_id", 0))
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
        envelope["benchmark"] = dict(total_wall_ms=round((time.perf_counter()-started)*1000, 2)
            if started is not None else 0., excludes_depth_preprocessing=True, **startup)
        latency = summarize_2d_latency(envelope.get('timeline', []), expected_count=len(images))
        envelope['benchmark']['two_d_latency'] = latency
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
        product = all(r.get("results_2d", {}).get("is_ok") is True for r in envelope["responses_2d"])
        return dict(ok=ok, product_is_ok=product if ok else None, board_id=board_id,
                    **options,
                    pieces=len(envelope["responses_2d"]), model_reused=startup.get("model_reused", False),
                    error=envelope["response"].get("error"), output_dir=str(folder),
                    total_wall_ms=envelope["benchmark"]["total_wall_ms"], two_d_latency=latency)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, port, output):
        super().__init__(("127.0.0.1", port), Handler)
        self.folder = Path(output).resolve() / ("session-" + uuid.uuid4().hex[:10])
        (self.folder / "uploads").mkdir(parents=True)
        self.assets = {f: (HERE / f).read_bytes() for f in ("index.html", "app.js", "pairing.js", "style.css")}
        self.runner, self.jobs = Runner(self.folder), {}
        self.workflows, self.grouped_runs = {}, set()
        self.workflow_number = 0
        # Match the bridge's bound of 26 protocol requests in flight.
        self.pool, self.lock = ThreadPoolExecutor(26), threading.Lock()

    def finish_workflow(self, request):
        ids, outcome = request.get("run_ids"), request.get("outcome")
        if not isinstance(ids, list) or any(
            not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{32}", key) for key in ids
        ) or len(set(ids)) != len(ids):
            raise ValueError("流程的检测记录无效或重复")
        if outcome not in ("done", "stopped", "failed"):
            raise ValueError("未知流程结束状态")
        with self.lock:
            if self.grouped_runs.intersection(ids):
                raise ValueError("检测记录已归入流程")
            jobs = [dict(self.jobs.get(key, {})) for key in ids]
            if any(job.get("status") == "running" for job in jobs):
                raise ValueError("流程仍有请求正在处理")
            self.workflow_number += 1
            title = f"流程 {self.workflow_number}"
            self.grouped_runs.update(ids)
        try:
            return self._write_workflow(ids, jobs, title, outcome, str(request.get("error") or ""))
        except Exception:
            with self.lock:
                self.grouped_runs.difference_update(ids)
            raise

    def _write_workflow(self, ids, jobs, title, outcome, error):
        runs, timing_rows = [], []
        for key, job in zip(ids, jobs):
            child = self.folder / key
            result_file = child / "result.json"
            if result_file.is_file():
                result = json.loads(result_file.read_text(encoding="utf-8"))
                if result.get("mode") == "workflow":
                    raise ValueError("流程报告不能作为单次检测归入")
            elif job.get("status") == "failed":
                result = dict(board_id=job["board_id"], response=dict(ok=False, error=job.get("error")))
            else:
                raise ValueError("找不到流程中的检测记录")
            artifact_errors = (job.get("result") or {}).get("artifact_errors")
            if artifact_errors:
                result["artifact_errors"] = artifact_errors
            runs.append(dict(id=key, board_id=job.get("board_id"), piece_id=job.get("piece_id"),
                             mode=job.get("mode"), document=result))
            timings = child / "timings.csv"
            if timings.is_file():
                timing_rows.extend(timings.read_text(encoding="utf-8-sig").splitlines()[1:])
        timed_jobs = [job for job in jobs if "request_sent_at" in job and "request_finished_at" in job]
        total_wall_ms = None
        if timed_jobs:
            begin = min(job["request_sent_at"] for job in timed_jobs)
            end = max(job["request_finished_at"] for job in timed_jobs)
            total_wall_ms = round((end - begin) * 1000, 2)
            timing_rows.insert(0, f",workflow,,total_wall_ms,{total_wall_ms}")
        key = uuid.uuid4().hex
        folder = self.folder / key
        folder.mkdir()
        document = dict(mode="workflow", title=title, outcome=outcome, error=error, runs=runs,
                        total_wall_ms=total_wall_ms)
        render_report(document, folder / "report.html", base_dir=folder)
        (folder / "result.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "timings.csv").write_text("\n".join(
            ["board_id,phase,piece_id,stage,elapsed_ms", *timing_rows]) + "\n", encoding="utf-8-sig")
        entry = dict(kind="workflow", title=title, status=outcome, created=time.time(), runs=len(runs),
                     boards=len({run["document"]["board_id"] for run in runs}), error=error,
                     total_wall_ms=total_wall_ms)
        with self.lock:
            self.workflows[key] = entry
        return dict(id=key, **entry)

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
                    rows = [dict(id=key, **job) for key, job in self.server.jobs.items()
                            if key not in self.server.grouped_runs]
                    rows.extend(dict(id=key, **group) for key, group in self.server.workflows.items())
                return self.send(200, sorted(rows, key=lambda row: row.get("created", 0), reverse=True)[:32])
            if path.startswith("/api/jobs/"):
                with self.server.lock:
                    key = path.rsplit("/", 1)[-1]
                    job = dict(self.server.jobs.get(key) or self.server.workflows.get(key) or {})
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
            if parsed.path not in {"/api/run", "/api/release-board", "/api/workflows", "/api/pitch/start", "/api/pitch/finish"}:
                return self.send(404, {"error": "接口不存在"})
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("验证请求须为JSON对象")
            if parsed.path == "/api/workflows":
                return self.send(201, self.server.finish_workflow(request))
            if parsed.path == "/api/pitch/start":
                board, _, depth = board_inputs(request.get("files"), request.get("board_id"), PITCH_START, 0)
                return self.send(202, self.server.runner.start_pitch(board, depth))
            if parsed.path == "/api/pitch/finish":
                ids, stopped = request.get("run_ids"), request.get("stopped")
                if not isinstance(ids, list) or type(stopped) is not bool or any(
                    not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{32}", key) for key in ids
                ):
                    raise ValueError("无效的 Pitch 送检记录")
                self.server.runner.finish_pitch_submissions(ids, stopped)
                return self.send(200, {})
            if parsed.path == "/api/release-board":
                board = board_number(request.get("board_id"))
                with self.server.lock:
                    if any(j["status"] == "running" and not j.get("pitch_waiting")
                           for j in self.server.jobs.values()):
                        return self.send(409, {"error": "已有验证运行中，请等待完成"})
                self.server.runner.release_manual(board)
                return self.send(200, {"board_id": board})
            mode = request.get("mode")
            manual = isinstance(mode, str) and mode in MANUAL_MODES
            wait_for_pitch_start = request.get("wait_for_pitch_start", False) if mode == PITCH_3D else False
            if type(wait_for_pitch_start) is not bool:
                raise ValueError("无效的 Pitch 启动等待设置")
            piece_id = request.get("piece_id") if manual else None
            options = {} if manual else submission_options(request.get("submission_mode", "concurrent"), request.get("request_interval_ms", 0))
            count = defaults()["piece_count"]
            board, images, depth = board_inputs(request.get("files"), request.get("board_id"), mode, count, piece_id=piece_id)
            rows = images + ([depth] if depth else [])
            with self.server.lock:
                running = [j for j in self.server.jobs.values() if j["status"] == "running"]
                if running and (not manual or any(j.get("mode") not in MANUAL_MODES for j in running)):
                    return self.send(409, {"error": "批量验证须单独运行，请等待当前请求完成"})
                same_piece = [j for j in running if j.get("board_id") == board and j.get("piece_id") == piece_id]
                conflict = any((not j.get("pitch_waiting") if mode == "manual_2d" else j.get("mode") != "manual_2d")
                               for j in same_piece)
                if manual and conflict:
                    return self.send(409, {"error": f"板 {board} 第 {piece_id} 片已有冲突请求正在处理"})
                key = uuid.uuid4().hex
                self.server.jobs[key] = dict(status="running", phase="准备验证", board_id=board, created=time.time())
                if manual:
                    self.server.jobs[key].update(mode=mode, piece_id=piece_id)
                if mode == PITCH_3D:
                    self.server.jobs[key].update(pitch_waiting=True, phase="等待单片深度和同片 2D 提交")
            folder = self.server.folder / key;folder.mkdir()
            submitted = threading.Event()
            request_sent_at = submitted_at = None

            def on_sent(sent_at):
                nonlocal request_sent_at, submitted_at
                request_sent_at = sent_at
                submitted_at = time.time() - (time.perf_counter() - sent_at)
                update(request_sent_at=sent_at, submitted_at=submitted_at)
                submitted.set()

            def on_finished(ended, received):
                received_at = submitted_at + (ended - request_sent_at) if received else None
                update(request_finished_at=ended, received_at=received_at)

            def update(**values):
                with self.server.lock:
                    self.server.jobs[key].update(values)
            def execute(prepared_piece=None):
                try:
                    progress = lambda phase: update(phase=phase)
                    if manual:
                        prepared = {"prepared_piece": prepared_piece} if prepared_piece is not None else {}
                        result = self.server.runner.run_manual(
                            rows, board, mode, piece_id, count, folder, progress, on_sent=on_sent,
                            on_finished=on_finished, **prepared)
                    else:
                        result = self.server.runner.run(rows, board, mode, count, folder, progress, **options)
                    update(status="done" if result["ok"] else "failed", result=result, error=result.get("error"))
                except Exception as exc:
                    update(status="failed", error=str(exc))

            if mode == PITCH_3D:
                def fail(reason):
                    update(status="failed", error=reason, pitch_waiting=False)

                def dispatch(piece):
                    update(pitch_waiting=False)
                    try:
                        self.server.pool.submit(execute, piece)
                    except RuntimeError as exc:
                        fail(str(exc))

                try:
                    self.server.runner.register_pitch_trigger(
                        key, board, piece_id, images, depth,
                        lambda ready_at: update(pitch_ready_at=ready_at), dispatch, fail,
                        wait_for_pitch_start=wait_for_pitch_start)
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    fail(str(exc))
                    return self.send(202, dict(id=key, submitted=False, error=str(exc)))
                return self.send(202, dict(id=key, pitch_waiting=True))
            future = self.server.pool.submit(execute)
            if manual:
                future.add_done_callback(lambda _: submitted.set())
                submitted.wait()
                with self.server.lock:
                    sent = submitted_at is not None
                    error = self.server.jobs[key].get("error") if not sent else None
                return self.send(202, dict(id=key, submitted=sent, submitted_at=submitted_at, error=error))
            self.send(202, {"id":key})
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
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
