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


class AlgorithmServer:
    def __init__(self):
        # Unified board algorithm is loaded only when it is initialized or used.
        self.board = None
        # Keep the legacy modules loadable while callers migrate to detect_board.
        self.two_d = None
        self.three_d = None
        workers = int(os.environ.get("BRIDGE_MAX_WORKERS", "13"))
        if not 2 <= workers <= 13:
            raise ValueError("BRIDGE_MAX_WORKERS must be between 2 and 13")
        self.cpu_pool = ThreadPoolExecutor(max_workers=workers - 1, thread_name_prefix="bridge-cpu")
        self.gpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bridge-gpu")
        self.pipeline = None
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
            "timings": {
                "total_init_ms": self.elapsed_ms(init_start),
            },
            "model_load_timings": {},
        }

    def handle_init_algorithm(self, req: Dict[str, Any]) -> Dict[str, Any]:
        stage = str(req.get("stage", "")).lower()

        if stage == "board":
            board = timed_call("init_board_import_and_construct_ms", self.get_board)
            return board.handle_init(req)

        if stage == "2d":
            return self.get_two_d().handle_init(req)

        if stage == "3d":
            return self.get_three_d().handle_init(req)

        return {
            "ok": False,
            "cmd": "init_algorithm",
            "stage": stage,
            "error": f"unsupported algorithm init stage: {stage or '<empty>'}",
        }

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
        self.gpu_pool.shutdown(wait=True)
        self.cpu_pool.shutdown(wait=True)

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
    AlgorithmServer().run()
