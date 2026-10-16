import json
import sys
import time
import traceback
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    # 允许直接以脚本方式运行 python/transport/bridge_server.py 时仍能导入 python 根包
    sys.path.insert(0, str(PYTHON_ROOT))

from algorithm.timing import timing_context, timed_call, record_duration
from algorithm.config import load_runtime_config


def configure_native_threads():
    """Set library defaults before loading NumPy/SciPy/PyTorch in the service process."""
    threads = int(os.environ.get("BRIDGE_NATIVE_THREADS", "1"))
    if threads < 1:
        raise ValueError("BRIDGE_NATIVE_THREADS must be positive")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))


class AlgorithmServer:
    def __init__(self):
        # Unified board algorithm is loaded only when it is initialized or used.
        self.board = None
        # Keep the legacy modules loadable while callers migrate to detect_board.
        self.two_d = None
        self.three_d = None
        self.cpu_pool = None
        self.postprocess_pool = None
        self.silver_pool = None
        self.joint_pool = None
        self.gpu_pool = None
        self.pipeline = None
        self.threading_info = {}
        self._configure_thread_pools(
            self._thread_settings({}), source="environment_default"
        )
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending = set()
        self._capacity = threading.BoundedSemaphore(26)
        self._stop = threading.Event()
        self._closed = False

    def log(self, msg: str):
        # stdout 只输出一行 JSON 响应；日志必须走 stderr，避免 C# 解析失败。
        print(msg, file=sys.stderr, flush=True)

    def handle_init(self, req: Dict[str, Any]) -> Dict[str, Any]:
        init_start = time.perf_counter()
        return {
            "ok": True,
            "cmd": "init",
            "message": "bridge protocol server ready",
            "threading": self.threading_info,
            "timings": {
                "total_init_ms": self.elapsed_ms(init_start),
            },
            "model_load_timings": {},
        }

    def handle_init_algorithm(self, req: Dict[str, Any]) -> Dict[str, Any]:
        stage = str(req.get("stage", "")).lower()
        try:
            config = load_runtime_config(
                str(req.get("config_text") or ""),
                str(req.get("config_path") or "") or None,
            )
            settings = self._thread_settings(config)
            if stage == "board":
                board = timed_call("init_board_import_and_construct_ms", self.get_board)
                configure_gpu_workers = getattr(board, "configure_gpu_worker_count", None)
                if callable(configure_gpu_workers):
                    configure_gpu_workers(settings["gpu_workers"])
                elif settings["gpu_workers"] > 1:
                    raise ValueError("multiple GPU workers are supported only by the unified board detector")
                response = board.handle_init(req)
                if response.get("ok"):
                    self._configure_thread_pools(settings, source="stage_config")
                    response["threading"] = self.threading_info
            elif stage == "2d":
                if settings["gpu_workers"] > 1:
                    raise ValueError("multiple GPU workers are supported only by the unified board detector")
                response = self.get_two_d().handle_init(req)
            elif stage == "3d":
                if settings["gpu_workers"] > 1:
                    raise ValueError("multiple GPU workers are supported only by the unified board detector")
                response = self.get_three_d().handle_init(req)
            else:
                return {
                    "ok": False,
                    "cmd": "init_algorithm",
                    "stage": stage,
                    "error": f"unsupported algorithm init stage: {stage or '<empty>'}",
                }
            if response.get("ok") and stage != "board":
                self._configure_thread_pools(settings, source="stage_config")
                response["threading"] = self.threading_info
            return response
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "ok": False,
                "cmd": "init_algorithm",
                "stage": stage,
                "error": str(exc),
            }

    @staticmethod
    def _threading_parameters(config: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
        stage = config.get(stage_name, {})
        if not isinstance(stage, dict):
            raise ValueError(f"{stage_name} must be an object")
        parameters = stage.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"{stage_name}.parameters must be an object")
        values = parameters.get("threading", {})
        if not isinstance(values, dict):
            raise ValueError(f"{stage_name}.parameters.threading must be an object")
        return values

    @staticmethod
    def _thread_count(name: str, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        if isinstance(value, int):
            count = value
        elif isinstance(value, str) and value.strip().isdecimal():
            count = int(value.strip())
        else:
            raise ValueError(f"{name} must be a positive integer")
        if count < 1:
            raise ValueError(f"{name} must be a positive integer")
        return count

    def _thread_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        values_2d = self._threading_parameters(config, "common_2d")
        values_3d = self._threading_parameters(config, "common_3d")
        budget = self._thread_count(
            "worker_budget",
            values_3d.get("worker_budget", os.environ.get("BRIDGE_MAX_WORKERS", 13)),
        )
        if budget < 2:
            raise ValueError("worker_budget/BRIDGE_MAX_WORKERS must be at least 2")
        gpu = self._thread_count(
            "gpu_workers",
            values_2d.get("gpu_workers", os.environ.get("BRIDGE_GPU_WORKERS", 1)),
        )
        silver_pool_shared = "silver_workers" not in values_2d and "BRIDGE_SILVER_WORKERS" not in os.environ
        joint_pool_shared = "joint_workers" not in values_3d and "BRIDGE_JOINT_WORKERS" not in os.environ
        # Reserve explicitly requested independent pools before assigning the
        # remaining budget to preparation/postprocessing. Missing keys retain
        # the old aliases and do not consume an extra worker allocation.
        silver = 0 if silver_pool_shared else self._thread_count(
            "silver_workers", values_2d.get("silver_workers", os.environ.get("BRIDGE_SILVER_WORKERS"))
        )
        joint = 0 if joint_pool_shared else self._thread_count(
            "joint_workers", values_3d.get("joint_workers", os.environ.get("BRIDGE_JOINT_WORKERS"))
        )
        reserved = silver + joint
        postprocess = self._thread_count(
            "postprocess_workers",
            values_2d.get(
                "postprocess_workers",
                os.environ.get("BRIDGE_POSTPROCESS_WORKERS", min(5, max(1, budget - gpu - reserved - 1))),
            ),
        )
        cpu = self._thread_count(
            "cpu_workers",
            values_3d.get(
                "cpu_workers",
                os.environ.get("BRIDGE_CPU_WORKERS", max(1, budget - postprocess - gpu - reserved)),
            ),
        )
        if silver_pool_shared:
            silver = postprocess
        if joint_pool_shared:
            joint = cpu
        early_joint = self._thread_count(
            "early_joint_workers", values_3d.get("early_joint_workers", joint)
        )
        if early_joint > joint:
            raise ValueError("early_joint_workers cannot exceed joint_workers")
        shared_cpu_pool = budget == 2 and cpu == 1 and postprocess == 1 and gpu == 1
        effective_total = cpu + gpu
        if not shared_cpu_pool:
            effective_total += postprocess
        if not silver_pool_shared:
            effective_total += silver
        if not joint_pool_shared:
            effective_total += joint
        if effective_total > budget:
            raise ValueError(
                "threading configuration exceeds worker_budget: "
                f"cpu_workers={cpu}, postprocess_workers={postprocess}, "
                f"gpu_workers={gpu}, silver_workers={silver}, joint_workers={joint}, "
                f"effective_total={effective_total}, worker_budget={budget}"
            )
        return {
            "worker_budget": budget,
            "cpu_workers": cpu,
            "postprocess_workers": postprocess,
            "gpu_workers": gpu,
            "postprocess_pool_shared": shared_cpu_pool,
            "silver_workers": silver,
            "silver_pool_shared": silver_pool_shared,
            "joint_workers": joint,
            "joint_pool_shared": joint_pool_shared,
            "early_joint_workers": early_joint,
        }

    def _configure_thread_pools(self, settings: Dict[str, Any], *, source: str) -> None:
        if self.pipeline is not None and self.pipeline.has_pending:
            raise RuntimeError("worker thread pools can only be changed while the detector is idle")
        same_budget = (
            self.threading_info.get("worker_budget") == settings["worker_budget"]
            and self.threading_info.get("cpu_workers") == settings["cpu_workers"]
            and self.threading_info.get("postprocess_workers") == settings["postprocess_workers"]
            and self.threading_info.get("postprocess_pool_shared")
            == settings["postprocess_pool_shared"]
            and self.threading_info.get("silver_workers") == settings["silver_workers"]
            and self.threading_info.get("silver_pool_shared") == settings["silver_pool_shared"]
            and self.threading_info.get("joint_workers") == settings["joint_workers"]
            and self.threading_info.get("joint_pool_shared") == settings["joint_pool_shared"]
            and self.threading_info.get("gpu_workers") == settings["gpu_workers"]
        )
        if not same_budget:
            old_pools = {
                id(pool): pool
                for pool in (self.cpu_pool, self.postprocess_pool,
                             self.silver_pool, self.joint_pool, self.gpu_pool)
                if pool is not None
            }
            created_pools = []

            def create_pool(stage):
                pool = ThreadPoolExecutor(
                    max_workers=settings[f"{stage}_workers"],
                    thread_name_prefix=f"bridge-{stage}",
                )
                created_pools.append(pool)
                return pool

            try:
                cpu_pool = create_pool("cpu")
                gpu_pool = create_pool("gpu")
                postprocess_pool = cpu_pool if settings["postprocess_pool_shared"] else create_pool("postprocess")
                silver_pool = postprocess_pool if settings["silver_pool_shared"] else create_pool("silver")
                joint_pool = cpu_pool if settings["joint_pool_shared"] else create_pool("joint")
            except Exception:
                for pool in created_pools:
                    pool.shutdown(wait=True)
                raise
            self.cpu_pool = cpu_pool
            self.postprocess_pool = postprocess_pool
            self.silver_pool = silver_pool
            self.joint_pool = joint_pool
            self.gpu_pool = gpu_pool
            if self.pipeline is not None:
                self.pipeline.cpu_pool = cpu_pool
                self.pipeline.postprocess_pool = postprocess_pool
                self.pipeline.silver_pool = silver_pool
                self.pipeline.joint_pool = joint_pool
                self.pipeline.gpu_pool = gpu_pool
            for pool in old_pools.values():
                pool.shutdown(wait=True)
        self.threading_info = {
            **settings,
            "native_threads": {
                key: os.environ.get(key)
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "source": source,
        }
        if self.pipeline is not None:
            self.pipeline.early_joint_workers = settings["early_joint_workers"]

    def handle_detect(self, req: Dict[str, Any]) -> Dict[str, Any]:
        stage = str(req.get("stage", "board")).lower()

        # Streaming requests are selected by explicit board/piece IDs. A
        # plain stage=2d or stage=3d request remains a standalone call.
        if stage == "2d" and req.get("board_id") and int(req.get("piece_id") or 0) > 0:
            return self.get_board().handle_detect_2d_piece(req)

        if stage == "3d" and req.get("board_id"):
            return self.get_board().handle_detect_3d_board(req)

        if stage == "board":
            return self.get_board().handle_detect(req)

        if stage == "2d":
            return self.get_two_d().handle_detect(req)

        if stage == "3d":
            return self.get_three_d().handle_detect(req)

        return {
            "ok": False,
            "cmd": "detect",
            "stage": stage,
            "error": f"unsupported detect stage: {stage}",
        }

    def handle_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        cmd = req.get("cmd")

        if cmd == "init":
            return self.handle_init(req)

        if cmd == "init_algorithm":
            return self.handle_init_algorithm(req)

        if cmd == "detect":
            return self.handle_detect(req)

        if cmd == "detect_2d_piece":
            return self.get_board().handle_detect_2d_piece(req)

        if cmd == "detect_3d_board":
            return self.get_board().handle_detect_3d_board(req)

        if cmd == "release_board":
            return self.get_board().handle_release_board(req)

        if cmd == "detect_board":
            return self.get_board().handle_detect(req)

        if cmd == "reload_defect_config":
            return self.get_two_d().handle_reload_defect_config(req)

        if cmd == "exit":
            return self.handle_shutdown("exit", "python detector exited")

        if cmd == "release":
            return self.handle_shutdown("release", "models released and python detector exited")

        return {
            "ok": False,
            "cmd": cmd,
            "error": f"unknown cmd: {cmd}",
        }

    def get_two_d(self):
        if self.two_d is None:
            from algorithm.detect_2D import Detect2D

            self.two_d = Detect2D(self.log)

        return self.two_d

    def get_board(self):
        if self.board is None:
            from algorithm.detect_board import DetectBoard

            self.board = DetectBoard(self.log)

        return self.board

    def get_three_d(self):
        if self.three_d is None:
            from algorithm.detect_3D import Detect3D

            self.three_d = Detect3D()

        return self.three_d

    def handle_shutdown(self, cmd: str, message: str) -> Dict[str, Any]:
        release_start = time.perf_counter()
        if self.board is not None:
            self.board.release()
        if self.two_d is not None:
            self.two_d.release()
        if self.three_d is not None:
            self.three_d.release()
        return {
            "ok": True,
            "cmd": cmd,
            "message": message,
            "timings": {
                "release_ms": self.elapsed_ms(release_start),
            },
        }

    def get_pipeline(self):
        if self.pipeline is None:
            from algorithm.board_pipeline import BoardPipeline
            self.pipeline = BoardPipeline(
                self.get_board(), self.cpu_pool, self.gpu_pool,
                postprocess_pool=self.postprocess_pool,
                silver_pool=self.silver_pool,
                joint_pool=self.joint_pool,
                early_joint_workers=self.threading_info["early_joint_workers"],
                timeout_seconds=float(os.environ.get("BRIDGE_BOARD_TIMEOUT_SECONDS", "90")),
                max_pending_boards=int(os.environ.get("BRIDGE_MAX_PENDING_BOARDS", "4")))
        return self.pipeline

    def dispatch(self, req):
        cmd, stage = req.get("cmd"), str(req.get("stage", "board")).lower()
        board_id = req.get("board_id")
        if cmd == "detect_2d_piece" or (cmd == "detect" and stage == "2d" and board_id):
            return self.get_pipeline().submit_2d(req)
        if cmd == "detect_3d_board" or (cmd == "detect" and stage == "3d" and board_id):
            return self.get_pipeline().submit_3d(req)
        if cmd == "release_board":
            board_id = self.get_board()._stream_board_id(req)
            return self.get_pipeline().release_board(board_id)
        if cmd in {"detect", "detect_board"}:
            # The shared model has exactly one owner. No CPU worker blocks on GPU work.
            pool = self.cpu_pool if cmd == "detect" and stage == "3d" else self.gpu_pool
            return pool.submit(self.handle_request, req)
        return self.handle_request(req)

    def _write_response(self, stream, req, response):
        response = dict(response)
        if req.get("request_id") is not None:
            response["request_id"] = req["request_id"]
        timings = {}
        with timing_context(timings, phase="transport", board_id=req.get("board_id") or req.get("batch_id"),
                            piece_id=req.get("piece_id"), request_id=req.get("request_id"), log=self.log):
            payload = timed_call("response_json_encode_ms", json.dumps, response, ensure_ascii=True)
            queued = time.perf_counter()
            with self._write_lock:
                record_duration("response_write_lock_wait_ms", (time.perf_counter() - queued) * 1000)
                timed_call("response_pipe_write_ms", print, payload, file=stream, flush=True)

    def _error(self, req, exc):
        self.log(traceback.format_exc())
        return {"ok": False, "cmd": req.get("cmd", "unknown"),
                "error": str(exc), "board_id": req.get("board_id", "")}

    def _accept(self, req, stream):
        if not self._capacity.acquire(blocking=False):
            self._write_response(stream, req, {"ok": False, "cmd": req.get("cmd"),
                                             "error": "server_busy", "message": "maximum pending requests reached"})
            return
        try:
            result = self.dispatch(req)
        except Exception as exc:
            self._capacity.release()
            self._write_response(stream, req, self._error(req, exc))
            return
        if not isinstance(result, Future):
            self._capacity.release()
            self._write_response(stream, req, result)
            return
        with self._pending_lock:
            self._pending.add(result)

        def completed(future):
            try:
                try:
                    response = future.result()
                except Exception as exc:
                    response = self._error(req, exc)
                self._write_response(stream, req, response)
            finally:
                with self._pending_lock:
                    self._pending.discard(future)
                self._capacity.release()
        result.add_done_callback(completed)

    def _expire(self):
        while not self._stop.wait(0.25):
            if self.pipeline is not None:
                self.pipeline.expire_pending()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self.pipeline is not None:
            self.pipeline.close()
        # CPU completions may still enqueue GPU jobs, which then enqueue CPU postprocessing.
        # Drain request futures before closing either executor.
        with self._pending_lock:
            pending = list(self._pending)
        if pending:
            wait(pending)
        for pool in {id(pool): pool for pool in (
                self.gpu_pool, self.silver_pool, self.postprocess_pool,
                self.joint_pool, self.cpu_pool) if pool is not None}.values():
            pool.shutdown(wait=True)

    def run(self):
        self.log("python bridge concurrent protocol server started")
        protocol_stdout = sys.stdout
        # One process-wide redirection for the lifetime of the worker pools, never per thread.
        sys.stdout = sys.stderr
        reaper = threading.Thread(target=self._expire, name="bridge-cache-expiry", daemon=True)
        reaper.start()
        try:
            while True:
                raw_line = sys.stdin.buffer.readline()
                if not raw_line:
                    break
                req = {}
                try:
                    line = raw_line.decode("utf-8-sig").strip()
                    opening = line.find("{")
                    if opening < 0:
                        continue
                    req = json.loads(line[opening:])
                    if not isinstance(req, dict):
                        raise ValueError("protocol request must be an object")
                    cmd = req.get("cmd")
                    if cmd in {"exit", "release"}:
                        self.close()
                        self._write_response(protocol_stdout, req, self.handle_shutdown(cmd, "detector exited"))
                        break
                    if cmd in {"init", "init_algorithm", "reload_defect_config"}:
                        with self._pending_lock:
                            busy = bool(self._pending) or bool(self.pipeline and self.pipeline.has_pending)
                        if busy:
                            self._write_response(protocol_stdout, req, {"ok": False, "cmd": cmd,
                                                                      "error": "initialization/configuration requires an idle detector"})
                        else:
                            timings = {}
                            with timing_context(timings, phase="init", request_id=req.get("request_id"), log=self.log):
                                response = timed_call("init_dispatch_ms", self.handle_request, req)
                            response.setdefault("timings", {}).update(timings)
                            self._write_response(protocol_stdout, req, response)
                    elif req.get("request_id") is None:
                        # Old bridges depend on ordered replies; preserve that protocol.
                        result = self.dispatch(req)
                        self._write_response(protocol_stdout, req,
                                             result.result() if isinstance(result, Future) else result)
                    else:
                        self._accept(req, protocol_stdout)
                except Exception as exc:
                    self._write_response(protocol_stdout, req, self._error(req, exc))
        finally:
            self.close()
            self.handle_shutdown("release", "stdin closed")
            reaper.join(timeout=1)
            sys.stdout = protocol_stdout

    @staticmethod
    def elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000.0


if __name__ == "__main__":
    configure_native_threads()
    AlgorithmServer().run()
