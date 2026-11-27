"""Caller-side preparation of one depth board, with independent futures for its pieces."""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np
from PIL import Image

from algorithm.vision3d.depth.depth_preprocessing import DepthPreprocessor
from algorithm.vision3d.depth.segment import SegmentationConfig, segment_depth_image


@dataclass(frozen=True)
class PreparedPiece:
    pixels: np.ndarray
    correction: dict
    ready_at: float
    profile: tuple


@dataclass
class PitchTrigger:
    board_id: str
    piece_id: int
    image_path: str
    depth_key: tuple
    depth: Future | None
    ready: Callable[[float], None]
    dispatch: Callable[[PreparedPiece], None]
    fail: Callable[[str], None]
    timer: threading.Timer
    two_d_sent: bool = False


@dataclass
class _Board:
    key: tuple
    pieces: dict
    cancelled: threading.Event
    touched: float


class PitchPreparations:
    def __init__(self):
        self._boards = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(2, thread_name_prefix="pitch-board")
        self._closed = False

    @staticmethod
    def profile(segmentation, settings):
        return segmentation, json.dumps(settings, sort_keys=True)

    def key(self, path, segmentation, settings):
        path = Path(path)
        stat = path.stat()
        return path, stat.st_size, stat.st_mtime_ns, *self.profile(segmentation, settings)

    def start(self, board_id, key, segmentation, settings):
        now = time.monotonic()
        with self._lock:
            if self._closed:
                raise RuntimeError("验证台正在停止")
            expired = [board for board, scan in self._boards.items()
                       if now - scan.touched > 90 and all(future.done() for future in scan.pieces.values())]
            for board in expired:
                self._boards.pop(board)
            scan = self._boards.get(board_id)
            if scan is not None and scan.key == key and any(not future.done() for future in scan.pieces.values()):
                scan.touched = now
                return scan.pieces, True
            if scan is None and len(self._boards) >= 4:
                idle = [(item.touched, board) for board, item in self._boards.items()
                        if all(future.done() for future in item.pieces.values())]
                if not idle:
                    raise RuntimeError("已有四块板正在准备深度")
                self._boards.pop(min(idle)[1])
            if scan is not None:
                scan.cancelled.set()
            scan = _Board(key, {pid: Future() for pid in range(1, segmentation.expected_piece_count + 1)},
                          threading.Event(), now)
            self._boards[board_id] = scan
            self._pool.submit(self._prepare, scan, key[0], segmentation, settings)
            return scan.pieces, False

    def piece(self, board_id, key, piece_id):
        with self._lock:
            scan = self._boards.get(board_id)
            if scan is None or scan.key != key:
                return None
            now = time.monotonic()
            if now - scan.touched > 90 and all(future.done() for future in scan.pieces.values()):
                self._boards.pop(board_id)
                return None
            scan.touched = now
            return scan.pieces[piece_id]

    @staticmethod
    def _prepare(scan, path, segmentation, settings):
        try:
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode not in {"I", "I;16", "I;16B", "I;16L"}:
                    raise ValueError("3D须为原始16位单通道深度PNG")
                raw = np.asarray(image, dtype=np.uint16)
            pieces = segment_depth_image(raw, segmentation)
            with DepthPreprocessor(settings) as processor:
                for piece in pieces:
                    if scan.cancelled.is_set():
                        raise RuntimeError("整板深度准备已停止")
                    # Match the DLL's full-width, piece-local projection.
                    source = raw[piece["y1"]:piece["y2"], :]
                    pixels, correction = processor.prepare(source)
                    if pixels is source:
                        pixels = pixels.copy()
                    scan.pieces[piece["piece_id"]].set_result(
                        PreparedPiece(pixels, correction, time.time(), scan.key[-2:]))
        except Exception as error:
            for future in scan.pieces.values():
                if not future.done():
                    future.set_exception(error)

    def release(self, board_id):
        with self._lock:
            scan = self._boards.pop(board_id, None)
            if scan is not None:
                scan.cancelled.set()

    def reset(self):
        with self._lock:
            self._cancel_boards()

    def _cancel_boards(self):
        for scan in self._boards.values():
            scan.cancelled.set()
        self._boards.clear()

    def close(self):
        with self._lock:
            self._closed = True
            self._cancel_boards()
        self._pool.shutdown(wait=True)
