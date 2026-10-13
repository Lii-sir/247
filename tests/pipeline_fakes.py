"""CPU-only deterministic detector for scheduling/protocol tests (never loads models)."""
import threading
import time
import numpy as np
from importlib import import_module


class FakeDetector:
    def __init__(self, count=10, delay=0.005):
        self.expected_piece_count = count
        self.segmentation_config = import_module("algorithm.3D.segment").SegmentationConfig(
            expected_piece_count=count, max_internal_gap_height=0)
        self.delay = delay
        self.lock = threading.Lock()
        self.gpu_active = self.max_gpu_active = 0
        self.cpu_active = self.max_cpu_active = 0
        self.prepared_ids = []
        self.measured_ids = []
        self.closed_inputs = 0
        self.gpu_gate = threading.Event()
        self.gpu_gate.set()
        self.gpu_started = threading.Event()
        self.depth_started = threading.Event()
        self.measured = threading.Event()

    def _stream_board_id(self, req):
        value = str(req.get("board_id") or "").strip()
        if not value:
            raise ValueError("board_id is required")
        return value

    def _stream_piece_id(self, req):
        value = int(req.get("piece_id") or 0)
        if not 1 <= value <= self.expected_piece_count:
            raise ValueError("invalid piece_id")
        return value

    def prepare_2d_piece(self, req):
        with self.lock:
            self.cpu_active += 1
            self.max_cpu_active = max(self.max_cpu_active, self.cpu_active)
        try:
            time.sleep(float(req.get("delay", self.delay)))
            with self.lock:
                self.prepared_ids.append((req["board_id"], req["piece_id"]))
            if req.get("fail_2d"):
                raise ValueError("synthetic 2D failure")
            return dict(req)
        finally:
            with self.lock:
                self.cpu_active -= 1

    def infer_2d_piece(self, prepared):
        with self.lock:
            self.gpu_active += 1
            self.max_gpu_active = max(self.max_gpu_active, self.gpu_active)
        try:
            self.gpu_started.set()
            if not self.gpu_gate.wait(3):
                raise TimeoutError("test GPU gate not opened")
            time.sleep(self.delay)
            if prepared.get("fail_gpu"):
                raise ValueError("synthetic GPU failure")
        finally:
            with self.lock:
                self.gpu_active -= 1

    def finish_2d_piece(self, prepared, *, publish_cache):
        assert not publish_cache
        response = dict(ok=True, cmd="detect_2d_piece", stage="2d",
                        board_id=prepared["board_id"], piece_id=prepared["piece_id"],
                        results_2d=dict(is_ok=True), marker=prepared.get("marker"))
        return response, dict(prepared)

    def release_prepared_2d(self, prepared):
        with self.lock:
            self.closed_inputs += 1

    def _read_3d(self, req):
        self.depth_started.set()
        if req.get("fail_3d"):
            raise ValueError("synthetic depth decode failure")
        depth = np.zeros((self.expected_piece_count * 10, 20), dtype=np.uint16)
        for index in range(self.expected_piece_count):
            depth[index * 10:index * 10 + 5] = 33000
        return depth

    def measure_prepared_piece(self, piece_id, entry, depth, piece):
        time.sleep(self.delay)
        if entry.get("fail_joint"):
            raise ValueError("synthetic registration failure")
        with self.lock:
            self.measured_ids.append((entry["board_id"], piece_id))
        self.measured.set()
        return dict(piece_id=piece_id, board_id=entry["board_id"], marker=entry.get("marker"),
                    registration=dict(is_valid=True), is_ok=True)

    def release(self):
        pass
