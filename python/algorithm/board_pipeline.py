"""Event-driven board assembly. Waiting for input never occupies a worker."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from typing import Any
from algorithm.tilt_correction import depth_coordinate_space
from algorithm.timing import timing_context, measure, record_duration, timed_call, note
from algorithm.registration_process import SharedBoardInputs, JointSharedInputs


@dataclass
class _Board:
    board_id: str
    touched: float = field(default_factory=time.monotonic)
    entries: dict = field(default_factory=dict)
    joint_entries: dict = field(default_factory=dict)
    inflight: set = field(default_factory=set)
    claimed: set = field(default_factory=set)
    results: dict = field(default_factory=dict)
    depth: Any = None
    depth_correction: dict = field(default_factory=dict)
    depth_inflight: bool = False
    pieces: list = field(default_factory=list)
    completion: Future | None = None
    started: float = 0.0
    timings: dict = field(default_factory=dict)
    depth_ready_at: float = 0.0
    request_id: str | None = None
    registration_inputs: SharedBoardInputs = field(default_factory=SharedBoardInputs)
    mask_inputs: dict = field(default_factory=dict)


class BoardPipeline:
    def __init__(self, detector, cpu_pool, gpu_pool, *, timeout_seconds=90.0,
                 max_pending_boards=16, postprocess_pool=None,
                 silver_pool=None, joint_pool=None, registration_pool=None, early_joint_workers=1):
        self.detector, self.cpu_pool, self.gpu_pool = detector, cpu_pool, gpu_pool
        self.postprocess_pool = postprocess_pool if postprocess_pool is not None else cpu_pool
        self.silver_pool = silver_pool if silver_pool is not None else self.postprocess_pool
        self.joint_pool = joint_pool if joint_pool is not None else cpu_pool
        self.registration_pool = registration_pool if registration_pool is not None else self.joint_pool
        self.timeout_seconds = timeout_seconds
        self.max_pending_boards = max_pending_boards
        if isinstance(early_joint_workers, bool) or not isinstance(early_joint_workers, int) or early_joint_workers < 1:
            raise ValueError("early_joint_workers must be a positive integer")
        self.early_joint_workers = early_joint_workers
        self._lock = threading.RLock()
        self._boards: dict[str, _Board] = {}
        self._joint_jobs = set()

    def _track_joint_job(self, future):
        with self._lock:
            self._joint_jobs.add(future)
        def completed(done):
            with self._lock:
                self._joint_jobs.discard(done)
        future.add_done_callback(completed)
        return future

    def drain_joint_jobs(self):
        """After boards are gone, finish old CPU work before model reinitialization."""
        with self._lock:
            if self._boards:
                raise RuntimeError("cannot drain joint jobs while boards are pending")
            jobs = list(self._joint_jobs)
        if jobs:
            wait(jobs)

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

    def _submit_timed(self, pool, phase, timings, board_id, piece_id, request_id, function, *args, state=None):
        queued = time.perf_counter()
        def invoke():
            if state is not None:
                with self._lock:
                    if not self._current(state):
                        raise RuntimeError("board was released before queued 3D work started")
            started = time.perf_counter()
            with timing_context(timings, phase=phase, board_id=board_id, piece_id=piece_id,
                                request_id=request_id, log=getattr(self.detector, "log", None)):
                record_duration(phase + "_queue_ms", (started - queued) * 1000)
                with measure(phase + "_phase_ms"):
                    result = function(*args)
                note("phase_summary", timings_ms=dict(timings))
                if phase in {"cpu_postprocess", "cpu_postprocess_finalize"} and isinstance(result, tuple):
                    response, entry = result
                    response.setdefault("timings", {}).update(timings)
                    entry["_2d_ready_at"] = time.perf_counter()
                elif phase in {"joint", "joint_measurement"} and isinstance(result, dict):
                    result.setdefault("timings", {}).update(timings)
                return result
        future = pool.submit(invoke)
        return self._track_joint_job(future) if state is not None else future

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
                # A completed, unclaimed piece can be replaced. Do not pair a
                # new 3D scan with its previous generation while reprocessing.
                previous_masks = state.mask_inputs.get(piece_id)
                if previous_masks is not None:
                    previous_masks.close()
                mask_inputs = state.mask_inputs[piece_id] = SharedBoardInputs()
                state.entries.pop(piece_id, None)
                state.joint_entries.pop(piece_id, None)
                state.inflight.add(piece_id)
        except (ValueError, TypeError) as exc:
            return self._completed(self._failure(board_id, "invalid_piece", str(exc), piece_id))
        output = Future()
        detail_timings = {}
        accepted_at = time.perf_counter()
        value_post_started = [0.0]

        def failed(exc):
            with self._lock:
                if self._current(state):
                    state.inflight.discard(piece_id)
                    state.joint_entries.pop(piece_id, None)
                    if piece_id in state.claimed:
                        self._terminate(state, self._failure(
                            board_id, "2d_piece_failed", f"piece {piece_id}: {exc}"))
            response = self._failure(board_id, "2d_piece_failed", str(exc), piece_id)
            response["timings"] = dict(detail_timings)
            output.set_result(response)

        def postprocessed(future):
            try:
                response, entry = future.result()
                if "cpu_2d_phase_ms" in detail_timings:
                    detail_timings["cpu_postprocess_phase_ms"] = detail_timings["cpu_2d_phase_ms"]+detail_timings["cpu_postprocess_finalize_phase_ms"]
                    detail_timings["cpu_postprocess_queue_ms"] = detail_timings["cpu_2d_queue_ms"]
                    detail_timings["cpu_postprocess_pipeline_ms"] = (time.perf_counter()-value_post_started[0])*1000
                    response.setdefault("timings", {}).update(detail_timings)
                response.setdefault("timings", {})["request_pipeline_ms"] = (time.perf_counter() - accepted_at) * 1000
                with self._lock:
                    if not self._current(state):
                        raise RuntimeError("board was released or expired during 2D processing")
                    state.inflight.discard(piece_id)
                    state.entries[piece_id] = entry
                    state.touched = time.monotonic()
                    response["cache_state"] = self._cache_state(state)
                    self._schedule_ready(state)
                    self._complete_if_ready(state)
                output.set_result(response)
            except Exception as exc:
                failed(exc)

        def prepared(future):
            value = None
            try:
                value = future.result()
            except Exception as exc:
                failed(exc)
                return

            # No executor worker waits on sibling futures. The launch guard
            # keeps an immediately completed branch from releasing the source
            # before its sibling has even been submitted.
            branch_lock = threading.Lock()
            remaining = 1
            errors = []
            branch_timings = {}

            def branch_finished():
                nonlocal remaining
                with branch_lock:
                    remaining -= 1
                    ready = remaining == 0
                if not ready:
                    return
                for timings in branch_timings.values():
                    detail_timings.update(timings)
                if errors:
                    try:
                        self.detector.release_prepared_2d(value)
                    finally:
                        failed(errors[0])
                    return

                def finish():
                    try:
                        response, entry = self.detector.finish_2d_piece(value, publish_cache=False)
                        if value.get("cpu_2d_result") is not None:
                            response["cpu_2d_worker"] = value["cpu_2d_result"]["cpu_2d_worker"]
                        return response, entry
                    finally:
                        self.detector.release_prepared_2d(value)

                def computed(result):
                    try:
                        data = result.result()
                        if data is not None:
                            value["cpu_2d_result"] = data
                        finalized = self._submit_timed(self.postprocess_pool, "cpu_postprocess_finalize", detail_timings,
                            board_id, piece_id, req.get("request_id"), finish)
                        self._track_joint_job(finalized).add_done_callback(postprocessed)
                    except Exception as exc:
                        self.detector.release_prepared_2d(value)
                        failed(exc)

                def is_current():
                    with self._lock:
                        return self._current(state)
                try:
                    value_post_started[0] = time.perf_counter()
                    job = self.detector.submit_2d_postprocess(self.postprocess_pool, mask_inputs, value, detail_timings,
                        dict(board_id=board_id, piece_id=piece_id, request_id=req.get("request_id")), is_current)
                    self._track_joint_job(job).add_done_callback(computed)
                except Exception as exc:
                    self.detector.release_prepared_2d(value)
                    failed(exc)

            def completed_branch(phase, result):
                try:
                    data = result.result()
                    if phase == "gpu":
                        launch("geometry", self.cpu_pool, self.detector.prepare_joint_piece)
                    elif phase == "silver_prepare":
                        value["silver_boundary"] = data
                    elif phase == "geometry":
                        with self._lock:
                            if not self._current(state):
                                raise RuntimeError("board was released or expired during 2D processing")
                            data["_2d_ready_at"] = time.perf_counter()
                            state.joint_entries[piece_id] = data
                            self._schedule_ready(state)
                except Exception as exc:
                    with branch_lock:
                        errors.append(exc)
                finally:
                    branch_finished()

            def launch(phase, pool, function):
                nonlocal remaining
                with branch_lock:
                    remaining += 1
                    timings = branch_timings[phase] = {}
                try:
                    job = self._submit_timed(pool, phase, timings, board_id, piece_id,
                                             req.get("request_id"), function, value)
                except Exception as exc:
                    job = Future()
                    job.set_exception(exc)
                job.add_done_callback(lambda result: completed_branch(phase, result))

            launch("silver_prepare", self.silver_pool, self.detector.prepare_silver_boundary)
            launch("gpu", self.gpu_pool, self.detector.infer_2d_piece)
            branch_finished()

        self._submit_timed(self.cpu_pool, "cpu_prepare", detail_timings, board_id, piece_id,
                           req.get("request_id"), self.detector.prepare_2d_piece, req).add_done_callback(prepared)
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
                state.request_id = req.get("request_id")
        except (ValueError, TypeError) as exc:
            return self._completed(self._failure(board_id, "invalid_board", str(exc)))

        def read_depth():
            # Import only when needed so unit tests can use a CPU-only fake detector.
            from importlib import import_module
            segment = import_module("algorithm.3D.segment")
            start = time.perf_counter()
            depth = timed_call("depth_read_ms", self.detector._read_3d, req)
            read_ms = (time.perf_counter() - start) * 1000
            depth, correction = self.detector.prepare_3d_depth(depth)
            start = time.perf_counter()
            pieces = timed_call("depth_segmentation_ms", segment.segment_depth_image,
                                depth, self.detector.segmentation_config)
            if len(pieces) != self.detector.expected_piece_count:
                raise ValueError("3D piece count does not match board configuration")
            return depth, pieces, read_ms, (time.perf_counter() - start) * 1000, correction

        def decoded(future):
            try:
                depth, pieces, read_ms, segment_ms, correction = future.result()
                with self._lock:
                    if self._current(state):
                        state.depth_inflight = False
                        state.depth, state.pieces = depth, pieces
                        state.depth_correction = correction
                        state.depth_ready_at = time.perf_counter()
                        state.timings.update(shared_memory_read_ms=read_ms,
                                             segmentation_3d_ms=segment_ms)
                        self._schedule_ready(state)
            except Exception as exc:
                with self._lock:
                    state.depth_inflight = False
                    self._terminate(state, self._failure(board_id, "3d_board_failed", str(exc)))

        self._submit_timed(self.cpu_pool, "depth", state.timings, board_id, None,
                           req.get("request_id"), read_depth).add_done_callback(decoded)
        return state.completion

    def _schedule_ready(self, state):
        # Called with the state lock. Only immutable references cross the worker boundary.
        if state.depth is None:
            return
        for piece in state.pieces:
            piece_id = int(piece["piece_id"])
            if piece_id not in state.joint_entries or piece_id in state.claimed:
                continue
            # Keep heavy 3D from overwhelming still-running 2D diagnostics.
            # This bounds admitted pieces across both 3D stages. Once every
            # 2D entry is complete, each executor supplies its own concurrency limit.
            if (len(state.entries) < self.detector.expected_piece_count
                    and len(state.claimed) - len(state.results) >= self.early_joint_workers):
                break
            state.claimed.add(piece_id)
            entry = state.joint_entries[piece_id]
            pair_wait = abs(state.depth_ready_at - entry.get("_2d_ready_at", state.depth_ready_at)) * 1000
            timings = {"matching_input_wait_ms": pair_wait}
            started = time.perf_counter()
            try:
                def is_current(board=state):
                    with self._lock:
                        return self._current(board)
                future = self.detector.submit_registration(self.registration_pool, state.registration_inputs,
                    piece_id, entry, state.depth, piece, state.depth_correction, timings,
                    dict(board_id=state.board_id, piece_id=piece_id, request_id=state.request_id), is_current)
                self._track_joint_job(future)
                future.add_done_callback(lambda result, box=piece, source=entry, ts=timings, begin=started:
                    self._registered(state, box, source, ts, begin, result))
            except Exception as exc:
                self._terminate(state, self._failure(state.board_id, "joint_detection_failed", str(exc)))
                return

    def _registered(self, state, piece, entry, timings, started, future):
        """Hand off without blocking a registration worker on the measurement pool."""
        try:
            registration = future.result()
            with self._lock:
                if not self._current(state):
                    return
                piece_id = int(piece["piece_id"])
                def is_current():
                    with self._lock:
                        return self._current(state)
                inputs = JointSharedInputs(state.registration_inputs, state.mask_inputs[piece_id])
                measured = self.detector.submit_joint_measurement(self.joint_pool, inputs,
                    piece_id, entry, state.depth, piece, registration, state.depth_correction, timings,
                    dict(board_id=state.board_id, piece_id=piece_id, request_id=state.request_id), is_current)
                self._track_joint_job(measured).add_done_callback(lambda result: self._measured(
                    state, piece_id, result, timings=timings, started=started))
        except Exception as exc:
            with self._lock:
                self._terminate(state, self._failure(state.board_id, "joint_detection_failed", str(exc)))

    def _measured(self, state, piece_id, future, *, timings, started):
        try:
            piece = future.result()
            # Preserve the old active-work aggregate. Report both queues and
            # end-to-end stage latency separately rather than hiding queueing.
            timings["joint_queue_ms"] = timings["registration_queue_ms"]
            timings["joint_phase_ms"] = timings["registration_phase_ms"] + timings["joint_measurement_phase_ms"]
            timings["joint_pipeline_ms"] = (time.perf_counter() - started) * 1000
            piece.setdefault("timings", {}).update(timings)
            with self._lock:
                if not self._current(state):
                    return
                state.results[piece_id] = piece
                state.touched = time.monotonic()
                self._schedule_ready(state)
                self._complete_if_ready(state)
        except Exception as exc:
            with self._lock:
                self._terminate(state, self._failure(state.board_id, "joint_detection_failed", str(exc)))

    def _complete_if_ready(self, state):
        # Called under the state lock. A measurement may finish before heavy
        # 2D diagnostics, but neither partial results nor partial OKs are public.
        count = self.detector.expected_piece_count
        if len(state.results) != count or len(state.entries) != count or state.inflight:
            return
        results = [self.detector.finalize_joint_piece(state.results[pid], state.entries[pid])
                   for pid in sorted(state.results)]
        elapsed = (time.perf_counter() - state.started) * 1000
        response = dict(ok=True, cmd="detect_3d_board", stage="3d",
                        board_id=state.board_id, batch_id=state.board_id,
                        results_board=dict(
                            pitch_correction=state.depth_correction,
                            depth_coordinate_space=depth_coordinate_space(state.depth_correction),
                            registration_ok=all(p["registration"]["is_valid"] for p in results),
                            is_ok=all(p["registration"]["is_valid"] and p["is_ok"] for p in results),
                            pieces=results),
                        released=dict(board_id=state.board_id,
                                      released_2d_count=len(state.entries), released_3d=True),
                        elapsed_ms=elapsed,
                        timings={**state.timings, "total_detect_ms": elapsed},
                        message="concurrent board assembly complete")
        self._terminate(state, response)

    def _terminate(self, state, response):
        if not self._current(state):
            return
        self._boards.pop(state.board_id)
        errors = []
        for inputs in (state.registration_inputs, *state.mask_inputs.values()):
            try:
                inputs.close()
            except Exception as error:
                errors.append(error)
        if errors:
            response = self._failure(state.board_id, "registration_shared_memory_cleanup_failed", str(errors[0]))
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
