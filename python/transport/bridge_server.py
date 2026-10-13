import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    # 允许直接以脚本方式运行 python/transport/bridge_server.py 时仍能导入 python 根包
    sys.path.insert(0, str(PYTHON_ROOT))


class AlgorithmServer:
    def __init__(self):
        # Unified board algorithm is loaded only when it is initialized or used.
        self.board = None
        # Keep the legacy modules loadable while callers migrate to detect_board.
        self.two_d = None
        self.three_d = None

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
            return self.get_board().handle_init(req)

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
        if stage == "2d" and req.get("board_id") is not None and req.get("piece_id") is not None:
            return self.get_board().handle_detect_2d_piece(req)

        if stage == "3d" and req.get("board_id") is not None:
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

    def run(self):
        self.log("python bridge protocol server started")

        while True:
            raw_line = sys.stdin.buffer.readline()

            if not raw_line:
                self.log("stdin closed")
                break

            line = raw_line.decode("utf-8-sig").strip()
            # .NET Framework can decode a UTF-8 BOM as two visible characters
            # on the first line. Protocol requests are JSON objects, so trim
            # any transport prefix before the first opening brace.
            # Strip any UTF-8 BOM/prefix by starting at the JSON object.
            object_start = line.find("{")
            if object_start >= 0:
                line = line[object_start:]
            else:
                continue

            if not line:
                continue

            try:
                req = json.loads(line)
                cmd = req.get("cmd")
                # stdout is a strict one-line JSON protocol. Third-party
                # algorithms (notably Ultralytics) may print banners or
                # progress text; route all such output to stderr while the
                # request is handled, then write only the JSON response to
                # the saved protocol stream.
                protocol_stdout = sys.stdout
                with redirect_stdout(sys.stderr):
                    response = self.handle_request(req)

                # The Windows Python process may expose stdout as GBK while
                # C# reads the pipe as UTF-8. ASCII-escape non-ASCII characters
                # in the protocol so JSON remains encoding-independent; the
                # C# JSON deserializer restores the original Chinese text.
                print(json.dumps(response, ensure_ascii=True), file=protocol_stdout, flush=True)

                if cmd == "exit" or cmd == "release":
                    break

            except Exception as e:
                error_response = {
                    "ok": False,
                    "cmd": "unknown",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

                print(json.dumps(error_response, ensure_ascii=True), file=sys.stdout, flush=True)

                self.log(traceback.format_exc())

    @staticmethod
    def elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000.0


if __name__ == "__main__":
    AlgorithmServer().run()
