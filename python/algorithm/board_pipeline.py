"""Event-driven board assembly. Waiting for input never occupies a worker."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Board:
    board_id: str
    touched: float = field(default_factory=time.monotonic)
    entries: dict = field(default_factory=dict)
    inflight: set = field(default_factory=set)
    claimed: set = field(default_factory=set)
    results: dict = field(default_factory=dict)
    depth: Any = None
    depth_inflight: bool = False
    pieces: list = field(default_factory=list)
    completion: Future | None = None
    started: float = 0.0
    timings: dict = field(default_factory=dict)


class BoardPipeline:
    def __init__(self, detector, cpu_pool, gpu_pool, *, timeout_seconds=90.0,
                 max_pending_boards=16):
        self.detector, self.cpu_pool, self.gpu_pool = detector, cpu_pool, gpu_pool
        self.timeout_seconds = timeout_seconds
        self.max_pending_boards = max_pending_boards
        self._lock = threading.RLock()
        self._boards: dict[str, _Board] = {}

    @staticmethod
    def _completed(value):
        future = Future()
        future.set_result(value)
        return future

    def _failure(self, board_id, error, message, piece_id=None):
        result = dict(ok=False, cmd="detect_3d_board" if piece_id is None else "detect_2d_piece",
                      stage="3d" if piece_id is None else "2d", board_id=board_id,
                      error=error, message=message)
        if piece_id is not None:
            result["piece_id"] = piece_id
        return result

    def _board(self, board_id):
        state = self._boards.get(board_id)
        if state is None:
            if len(self._boards) >= self.max_pending_boards:
                raise ValueError("pending board limit reached; release incomplete boards")
            state = self._boards[board_id] = _Board(board_id)
        state.touched = time.monotonic()
        return state

    def _current(self, state):
        return self._boards.get(state.board_id) is state

    def _cache_state(self, state):
        ids = sorted(state.entries)
        return dict(board_id=state.board_id, expected_piece_count=self.detector.expected_piece_count,
                    received_piece_count=len(ids), piece_ids=ids,
                    ready_for_3d=len(ids) == self.detector.expected_piece_count)

    def submit_2d(self, req):
        board_id = str(req.get("board_id") or "")
        piece_id = req.get("piece_id")
        try:
            board_id = self.detector._stream_board_id(req)
            piece_id = self.detector._stream_piece_id(req)
            with self._lock:
                state = self._board(board_id)
                if piece_id in state.inflight or piece_id in state.claimed:
                    raise ValueError("piece is already being processed; use a new board_id for a new scan")
                state.inflight.add(piece_id)
        except (ValueError, TypeError) as exc:
            return self._completed(self._failure(board_id, "invalid_piece", str(exc), piece_id))
        output = Future()

        def failed(exc):
            with self._lock:
                if self._current(state):
                    state.inflight.discard(piece_id)
            output.set_result(self._failure(board_id, "2d_piece_failed", str(exc), piece_id))

        def postprocessed(future):
            try:
                response, entry = future.result()
                with self._lock:
                    if not self._current(state):
                        raise RuntimeError("board was released or expired during 2D processing")
                    state.inflight.discard(piece_id)
                    state.entries[piece_id] = entry
                    state.touched = time.monotonic()
                    response["cache_state"] = self._cache_state(state)
                    self._schedule_ready(state)
                output.set_result(response)
            except Exception as exc:
                failed(exc)

        def inferred(future, prepared):
            try:
                future.result()
                def finish():
                    try:
                        return self.detector.finish_2d_piece(prepared, publish_cache=False)
                    finally:
                        self.detector.release_prepared_2d(prepared)
                self.cpu_pool.submit(finish).add_done_callback(postprocessed)
            except Exception as exc:
                self.detector.release_prepared_2d(prepared)
                failed(exc)

        def prepared(future):
            value = None
            try:
                value = future.result()
                self.gpu_pool.submit(self.detector.infer_2d_piece, value).add_done_callback(
                    lambda result: inferred(result, value))
            except Exception as exc:
                if value is not None:
                    self.detector.release_prepared_2d(value)
                failed(exc)

        self.cpu_pool.submit(self.detector.prepare_2d_piece, req).add_done_callback(prepared)
        return output

    def submit_3d(self, req):
        board_id = str(req.get("board_id") or "")
        try:
            board_id = self.detector._stream_board_id(req)
            with self._lock:
                state = self._board(board_id)
                if state.completion is not None:
                    raise ValueError("a 3D scan is already pending for this board")
                state.completion = Future()
                state.depth_inflight = True
                state.started = time.perf_counter()
        except (ValueError, TypeError) as exc:
            return self._completed(self._failure(board_id, "invalid_board", str(exc)))

        def decode():
            # Import only when needed so unit tests can use a CPU-only fake detector.
            from importlib import import_module
            segment = import_module("algorithm.3D.segment")
            start = time.perf_counter()
            depth = self.detector._read_3d(req)
            read_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            pieces = segment.segment_depth_image(depth, self.detector.segmentation_config)
            if len(pieces) != self.detector.expected_piece_count:
                raise ValueError("3D piece count does not match board configuration")
            return depth, pieces, read_ms, (time.perf_counter() - start) * 1000

        def decoded(future):
            try:
                depth, pieces, read_ms, segment_ms = future.result()
                with self._lock:
                    if self._current(state):
                        state.depth_inflight = False
                        state.depth, state.pieces = depth, pieces
                        state.timings.update(shared_memory_read_decode_ms=read_ms,
                                             segmentation_3d_ms=segment_ms)
                        self._schedule_ready(state)
            except Exception as exc:
                with self._lock:
                    state.depth_inflight = False
                    self._terminate(state, self._failure(board_id, "3d_board_failed", str(exc)))

        self.cpu_pool.submit(decode).add_done_callback(decoded)
        return state.completion

    def _schedule_ready(self, state):
        # Called with the state lock. Only immutable references cross the worker boundary.
        if state.depth is None:
            return
        for piece in state.pieces:
            piece_id = int(piece["piece_id"])
            if piece_id not in state.entries or piece_id in state.claimed:
                continue
            state.claimed.add(piece_id)
            future = self.cpu_pool.submit(self.detector.measure_prepared_piece,
                                          piece_id, state.entries[piece_id], state.depth, piece)
            future.add_done_callback(lambda result, pid=piece_id: self._measured(state, pid, result))

    def _measured(self, state, piece_id, future):
        try:
            piece = future.result()
            with self._lock:
                if not self._current(state):
                    return
                state.results[piece_id] = piece
                state.touched = time.monotonic()
                if len(state.results) == self.detector.expected_piece_count:
                    results = [state.results[pid] for pid in sorted(state.results)]
                    elapsed = (time.perf_counter() - state.started) * 1000
                    response = dict(ok=True, cmd="detect_3d_board", stage="3d",
                                    board_id=state.board_id, batch_id=state.board_id,
                                    results_board=dict(
                                        registration_ok=all(p["registration"]["is_valid"] for p in results),
                                        is_ok=all(p["registration"]["is_valid"] and p["is_ok"] for p in results),
                                        pieces=results),
                                    released=dict(board_id=state.board_id,
                                                  released_2d_count=len(state.entries), released_3d=True),
                                    elapsed_ms=elapsed,
                                    timings={**state.timings, "total_detect_ms": elapsed},
                                    message="concurrent board assembly complete")
                    self._terminate(state, response)
        except Exception as exc:
            with self._lock:
                self._terminate(state, self._failure(state.board_id, "joint_detection_failed", str(exc)))

    def _terminate(self, state, response):
        if not self._current(state):
            return
        self._boards.pop(state.board_id)
        if state.completion is not None and not state.completion.done():
            state.completion.set_result(response)

    def release_board(self, board_id):
        with self._lock:
            state = self._boards.get(board_id)
            count = len(state.entries) if state else 0
            if state:
                self._terminate(state, self._failure(board_id, "board_released", "board cache released"))
        return dict(ok=True, cmd="release_board", stage="board", board_id=board_id,
                    released=dict(board_id=board_id, released_2d_count=count, released_3d=bool(state and state.depth is not None)))

    def expire_pending(self):
        with self._lock:
            now = time.monotonic()
            for state in list(self._boards.values()):
                active = bool(state.inflight or state.depth_inflight
                              or state.claimed.difference(state.results))
                if not active and now - state.touched >= self.timeout_seconds:
                    result = self._failure(state.board_id, "board_timeout", "waiting for matching 2D/3D input timed out")
                    result["missing_piece_ids"] = sorted(set(range(1, self.detector.expected_piece_count + 1)) - set(state.entries))
                    self._terminate(state, result)

    def close(self):
        with self._lock:
            for state in list(self._boards.values()):
                self._terminate(state, self._failure(state.board_id, "server_closed", "detector is shutting down"))

    @property
    def has_pending(self):
        with self._lock:
            return bool(self._boards)
