"""Event-driven board assembly. Waiting for input never occupies a worker."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from algorithm.application.board.board_depth import BoardDepth, piece_depth_geometry, read_board_depth
from algorithm.application.board.board_results import board_results, piece_3d_response, skipped_3d_piece
from algorithm.runtime.timing import timing_context, measure, record_duration, timed_call, note, PipelineTimings
from algorithm.execution.processes.registration_process import SharedBoardInputs, JointSharedInputs


@dataclass
class _PieceRequest:
    request: dict
    geometry: dict
    correction: dict
    completion: Future = field(default_factory=Future)
    started: float = field(default_factory=time.perf_counter)
    timings: dict = field(default_factory=dict)
    registration_inputs: SharedBoardInputs = field(default_factory=SharedBoardInputs)


@dataclass
class _Board:
    board_id: str
    touched: float = field(default_factory=time.monotonic)
    entries: dict = field(default_factory=dict)
    joint_entries: dict = field(default_factory=dict)
    inflight: set = field(default_factory=set)
    claimed: set = field(default_factory=set)
    results: dict = field(default_factory=dict)
    depth_scan: BoardDepth | None = None
    depth_inflight: bool = False
    completion: Future | None = None
    started: float = 0.0
    timings: dict = field(default_factory=dict)
    depth_ready_at: float = 0.0
    request_id: str | None = None
    registration_inputs: SharedBoardInputs = field(default_factory=SharedBoardInputs)
    mask_inputs: dict = field(default_factory=dict)
    piece_timelines: dict = field(default_factory=dict)
    diagnostics_ready_at: dict = field(default_factory=dict)
    piece_mode: bool = False
    piece_requests: dict[int, _PieceRequest] = field(default_factory=dict)


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

    def _failure(self, board_id, error, message, piece_id=None, *, cmd=None):
        if cmd is None:
            cmd = "detect_3d_board" if piece_id is None else "detect_2d_piece"
        result = dict(ok=False, cmd=cmd,
                      stage="2d" if cmd == "detect_2d_piece" else "3d", board_id=board_id,
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

    @staticmethod
    def _take_piece_cache(state, piece_id):
        state.entries.pop(piece_id, None)
        state.joint_entries.pop(piece_id, None)
        state.diagnostics_ready_at.pop(piece_id, None)
        return state.mask_inputs.pop(piece_id, None)

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
                if piece_id in state.inflight or piece_id in state.claimed or piece_id in state.piece_requests:
                    raise ValueError("piece is already being processed; use a new board_id for a new scan")
                # A completed, unclaimed piece can be replaced. Do not pair a
                # new 3D scan with its previous generation while reprocessing.
                previous_masks = state.mask_inputs.get(piece_id)
                if previous_masks is not None:
                    previous_masks.close()
                mask_inputs = state.mask_inputs[piece_id] = SharedBoardInputs()
                state.entries.pop(piece_id, None)
                state.joint_entries.pop(piece_id, None)
                state.piece_timelines.pop(piece_id, None)
                state.diagnostics_ready_at.pop(piece_id, None)
                state.inflight.add(piece_id)
        except (ValueError, TypeError) as exc:
            return self._completed(self._failure(board_id, "invalid_piece", str(exc), piece_id))
        output = Future()
        detail_timings = {}
        accepted_at = time.perf_counter()
        value_post_started = [0.0]
        ordered_checks = bool(getattr(self.detector, "defect_order", ()))

        def failed(exc):
            with self._lock:
                if self._current(state):
                    state.inflight.discard(piece_id)
                    if piece_id in state.claimed:
                        self._terminate(state, self._failure(
                            board_id, "2d_piece_failed", f"piece {piece_id}: {exc}"))
                    else:
                        masks = self._take_piece_cache(state, piece_id)
                        if masks is not None:
                            try:
                                masks.close()
                            except Exception as cleanup_error:
                                exc = RuntimeError(f"{exc}. Shared mask cleanup failed: {cleanup_error}")
                        self._start_piece_if_ready(state, piece_id)
                        if state.completion is None and not (state.entries or state.inflight or state.piece_requests):
                            self._terminate(state, self._failure(board_id, "2d_piece_failed", str(exc)))
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
                    if ordered_checks:
                        state.joint_entries[piece_id] = entry
                    state.diagnostics_ready_at[piece_id] = time.perf_counter()
                    state.touched = time.monotonic()
                    response["cache_state"] = self._cache_state(state)
                    self._schedule_ready(state)
                    self._complete_if_ready(state)
                    self._start_piece_if_ready(state, piece_id)
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

                def finalize():
                    phase = "cpu_postprocess" if ordered_checks else "cpu_postprocess_finalize"
                    finalized = self._submit_timed(self.postprocess_pool, phase, detail_timings,
                        board_id, piece_id, req.get("request_id"), finish)
                    self._track_joint_job(finalized).add_done_callback(postprocessed)

                def computed(result):
                    try:
                        data = result.result()
                        if data is not None:
                            value["cpu_2d_result"] = data
                        finalize()
                    except Exception as exc:
                        self.detector.release_prepared_2d(value)
                        failed(exc)

                def is_current():
                    with self._lock:
                        return self._current(state)
                try:
                    if ordered_checks:
                        finalize()
                        return
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
                    if phase == "gpu" and not ordered_checks:
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

            if not ordered_checks:
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
                if state.piece_mode:
                    raise ValueError("board is using per-piece 3D submissions")
                if state.completion is not None:
                    raise ValueError("a 3D scan is already pending for this board")
                state.completion = Future()
                state.depth_inflight = True
                state.started = time.perf_counter()
                state.request_id = req.get("request_id")
        except (ValueError, TypeError) as exc:
            return self._completed(self._failure(board_id, "invalid_board", str(exc)))

        def decoded(future):
            try:
                scan = future.result()
                with self._lock:
                    if self._current(state):
                        state.depth_inflight = False
                        state.depth_scan = scan
                        state.touched = time.monotonic()
                        state.depth_ready_at = time.perf_counter()
                        state.timings.update(shared_memory_read_ms=scan.read_ms,
                                             segmentation_3d_ms=scan.segmentation_ms)
                        self._schedule_ready(state)
                        self._complete_if_ready(state)
            except Exception as exc:
                with self._lock:
                    state.depth_inflight = False
                    self._terminate(state, self._failure(board_id, "3d_board_failed", str(exc)))

        self._submit_timed(self.cpu_pool, "depth", state.timings, board_id, None,
                           req.get("request_id"), read_board_depth, req, self.detector._read_3d,
                           self.detector.segmentation_config,
                           self.detector.expected_piece_count).add_done_callback(decoded)
        return state.completion

    def submit_3d_piece(self, req):
        board_id = str(req.get("board_id") or "")
        piece_id = req.get("piece_id")
        try:
            board_id = self.detector._stream_board_id(req)
            if type(piece_id) is not int:
                raise ValueError("piece_id must be an integer")
            piece_id = self.detector._stream_piece_id(req)
            geometry, correction = piece_depth_geometry(req, piece_id)
            with self._lock:
                state = self._board(board_id)
                if state.completion is not None:
                    raise ValueError("board already has a whole-board 3D submission")
                if piece_id in state.piece_requests:
                    raise ValueError("a 3D submission is already pending for this piece")
                state.piece_mode = True
                job = state.piece_requests[piece_id] = _PieceRequest(req, geometry, correction)
                self._start_piece_if_ready(state, piece_id)
        except (ValueError, TypeError) as exc:
            return self._completed(self._failure(
                board_id, "invalid_piece", str(exc), piece_id, cmd="detect_3d_piece"))
        return job.completion

    def _start_piece_if_ready(self, state, piece_id):
        if not self._current(state) or piece_id in state.inflight:
            return
        job = state.piece_requests.get(piece_id)
        if job is None:
            return
        entry = state.entries.get(piece_id)
        reason = "missing_2d_result" if entry is None else entry.get("skip_3d_reason")

        def fail(error):
            self._finish_piece_request(state, piece_id, self._failure(
                state.board_id, "3d_piece_failed", str(error), piece_id, cmd="detect_3d_piece"))

        def complete(piece):
            if entry is not None:
                piece = self.detector.finalize_joint_piece(piece, entry)
            self._finish_piece_request(
                state, piece_id, piece_3d_response(state.board_id, piece, job.started, job.timings),
                consume=True)

        if reason:
            try:
                complete(skipped_3d_piece(job.geometry, job.correction, reason, entry))
            except Exception as exc:
                fail(exc)
            return

        def measured(future, timings, started):
            with self._lock:
                if not self._current(state):
                    return
                try:
                    complete(self._with_joint_timings(future.result(), timings, started))
                except Exception as exc:
                    fail(exc)

        def decoded(future):
            with self._lock:
                if not self._current(state):
                    return
                try:
                    depth = future.result()
                    started = time.perf_counter()
                    timings = PipelineTimings({}, origin=job.started, timeline={})
                    inputs = JointSharedInputs(job.registration_inputs, state.mask_inputs[piece_id])
                    pending = self._submit_joint(state, job.geometry, entry, depth, job.correction,
                                                 inputs, timings, job.request.get("request_id"))
                    pending.add_done_callback(lambda result: measured(result, timings, started))
                except Exception as exc:
                    fail(exc)

        try:
            self._submit_timed(self.cpu_pool, "depth", job.timings, state.board_id, piece_id,
                               job.request.get("request_id"), self.detector._read_3d,
                               job.request, state=state).add_done_callback(decoded)
        except Exception as exc:
            fail(exc)

    def _finish_piece_request(self, state, piece_id, response, *, consume=False):
        job = state.piece_requests.pop(piece_id)
        inputs = [job.registration_inputs]
        if consume:
            masks = self._take_piece_cache(state, piece_id)
            if masks is not None:
                inputs.append(masks)
        for shared_input in inputs:
            try:
                shared_input.close()
            except Exception as exc:
                response = self._failure(state.board_id, "registration_shared_memory_cleanup_failed",
                                         str(exc), piece_id, cmd="detect_3d_piece")
        state.touched = time.monotonic()
        if not (state.entries or state.inflight or state.piece_requests):
            self._terminate(state, response)
        if not job.completion.done():
            job.completion.set_result(response)

    def _schedule_ready(self, state):
        # Called with the state lock. Only immutable references cross the worker boundary.
        scan = state.depth_scan
        if scan is None or not self._current(state):
            return
        for piece in scan.pieces:
            piece_id = int(piece["piece_id"])
            if piece_id not in state.joint_entries or piece_id in state.claimed:
                continue
            entry = state.joint_entries[piece_id]
            reason = entry.get("skip_3d_reason")
            if reason:
                if piece_id in state.entries:
                    state.results[piece_id] = skipped_3d_piece(piece, scan.correction, reason, entry)
                    state.claimed.add(piece_id)
                continue
            # Keep heavy 3D from overwhelming still-running 2D diagnostics.
            # This bounds admitted pieces across both 3D stages. Once every
            # 2D entry is complete, each executor supplies its own concurrency limit.
            if (len(state.entries) < self.detector.expected_piece_count
                    and len(state.claimed) - len(state.results) >= self.early_joint_workers):
                continue
            state.claimed.add(piece_id)
            input_ready_at = entry.get("_2d_ready_at", state.depth_ready_at)
            pair_wait = abs(state.depth_ready_at - input_ready_at) * 1000
            started = time.perf_counter()
            paired_at = max(state.depth_ready_at, input_ready_at)
            timeline = state.piece_timelines.setdefault(piece_id, {})
            timeline.update(input_2d_ready_ms=(input_ready_at-state.started)*1000,
                depth_ready_ms=(state.depth_ready_at-state.started)*1000,
                inputs_ready_ms=(paired_at-state.started)*1000,
                registration_admitted_ms=(started-state.started)*1000)
            timings = PipelineTimings(dict(matching_input_wait_ms=pair_wait,
                joint_admission_wait_ms=max(0., (started-paired_at)*1000)),
                origin=state.started, timeline=timeline)
            try:
                inputs = JointSharedInputs(state.registration_inputs, state.mask_inputs[piece_id])
                future = self._submit_joint(state, piece, entry, scan.image, scan.correction,
                                            inputs, timings, state.request_id)
                future.add_done_callback(lambda result, pid=piece_id, ts=timings, begin=started:
                    self._measured(state, pid, result, timings=ts, started=begin))
                if not self._current(state):
                    return
            except Exception as exc:
                self._terminate(state, self._failure(state.board_id, "joint_detection_failed", str(exc)))
                return

    def _submit_joint(self, state, piece, entry, depth, correction, inputs, timings, request_id):
        """Chain the two worker pools for either depth input format."""
        output = Future()
        piece_id = int(piece["piece_id"])
        metadata = dict(board_id=state.board_id, piece_id=piece_id, request_id=request_id)

        def is_current():
            with self._lock:
                return self._current(state)

        def measured(future):
            try:
                output.set_result(future.result())
            except Exception as exc:
                output.set_exception(exc)

        def registered(future):
            try:
                registration = future.result()
                with self._lock:
                    if not self._current(state):
                        raise RuntimeError("board was released during 3D registration")
                    timeline = timings.timeline
                    admitted_ms = (time.perf_counter() - timings.origin) * 1000
                    timeline["measurement_admitted_ms"] = admitted_ms
                    timings["registration_to_measurement_handoff_ms"] = max(
                        0., admitted_ms - timeline.get("registration_result_received_ms", admitted_ms))
                    job = self.detector.submit_joint_measurement(self.joint_pool, inputs,
                        piece_id, entry, depth, piece, registration, correction, timings, metadata, is_current)
                    self._track_joint_job(job).add_done_callback(measured)
            except Exception as exc:
                output.set_exception(exc)

        job = self.detector.submit_registration(self.registration_pool, inputs.board_inputs,
            piece_id, entry, depth, piece, correction, timings, metadata, is_current)
        self._track_joint_job(job).add_done_callback(registered)
        return output

    @staticmethod
    def _with_joint_timings(piece, timings, started):
        timings["joint_queue_ms"] = timings["registration_queue_ms"]
        timings["joint_phase_ms"] = timings["registration_phase_ms"] + timings["joint_measurement_phase_ms"]
        timings["joint_pipeline_ms"] = (time.perf_counter() - started) * 1000
        piece.setdefault("timings", {}).update(timings)
        return piece

    def _measured(self, state, piece_id, future, *, timings, started):
        try:
            piece = self._with_joint_timings(future.result(), timings, started)
            with self._lock:
                if not self._current(state):
                    return
                state.results[piece_id] = piece
                state.piece_timelines[piece_id]["joint_completed_ms"] = (time.perf_counter()-state.started)*1000
                state.touched = time.monotonic()
                self._schedule_ready(state)
                self._complete_if_ready(state)
        except Exception as exc:
            with self._lock:
                self._terminate(state, self._failure(state.board_id, "joint_detection_failed", str(exc)))

    def _complete_if_ready(self, state, *, missing_ids=()):
        # Called under the state lock. A measurement may finish before heavy
        # 2D diagnostics, but neither partial results nor partial OKs are public.
        count = self.detector.expected_piece_count
        if (not self._current(state) or len(state.results) != count
                or len(state.entries) + len(missing_ids) != count or state.inflight):
            return
        finalize_started = time.perf_counter()
        results = []
        for pid in sorted(state.results):
            piece = state.results[pid]
            if pid in state.entries:
                timeline = state.piece_timelines.setdefault(pid, {})
                timeline["diagnostics_2d_ready_ms"] = (state.diagnostics_ready_at[pid]-state.started)*1000
                piece = self.detector.finalize_joint_piece(piece, state.entries[pid])
                piece["timeline"] = dict(timeline)
            results.append(piece)
        state.timings["board_finalization_ms"] = (time.perf_counter()-finalize_started)*1000
        diagnostics = self._pipeline_diagnostics(state)
        state.timings["board_wait_2d_after_3d_ms"] = diagnostics["wait_2d_after_3d_ms"]
        state.timings.update(
            board_max_admission_wait_ms=diagnostics["max_admission_wait_ms"],
            board_max_registration_queue_ms=diagnostics["max_registration_queue_ms"],
            board_max_measurement_queue_ms=diagnostics["max_measurement_queue_ms"],
            board_max_registration_process_wait_ms=diagnostics["max_registration_process_wait_ms"],
            board_max_measurement_process_wait_ms=diagnostics["max_measurement_process_wait_ms"],
            board_slowest_joint_pipeline_ms=diagnostics["slowest_joint_pipeline_ms"],
            board_last_3d_pipeline_ms=diagnostics["last_3d_joint_pipeline_ms"],
            board_max_registration_phase_ms=diagnostics["max_registration_phase_ms"],
            board_max_measurement_phase_ms=diagnostics["max_measurement_phase_ms"])
        elapsed = (time.perf_counter() - state.started) * 1000
        response = dict(ok=True, cmd="detect_3d_board", stage="3d",
                        board_id=state.board_id, batch_id=state.board_id,
                        results_board=board_results(results, state.depth_scan.correction),
                        released=dict(board_id=state.board_id,
                                      released_2d_count=len(state.entries), released_3d=True),
                        elapsed_ms=elapsed,
                        timings={**state.timings, "total_detect_ms": elapsed},
                        message="concurrent board assembly complete")
        response["results_board"]["pipeline_diagnostics"] = diagnostics
        if missing_ids:
            response["missing_piece_ids"] = list(missing_ids)
        self._terminate(state, response)

    def _pipeline_diagnostics(self, state):
        """Board-relative positions and real queues: do not infer a board from one piece."""
        measured = {pid: piece for pid, piece in state.results.items() if not piece.get("skip_3d_reason")}
        last_3d = max(measured, key=lambda pid: state.piece_timelines[pid]["joint_completed_ms"], default=None)
        last_2d = max(state.diagnostics_ready_at, key=state.diagnostics_ready_at.get, default=None)
        last_3d_ms = state.piece_timelines.get(last_3d, {}).get("joint_completed_ms", 0.)
        last_2d_ms = (state.diagnostics_ready_at.get(last_2d, state.started)-state.started)*1000
        slowest = max(measured, key=lambda pid: measured[pid]["timings"]["joint_pipeline_ms"], default=None)

        def capacity(service_name, pool):
            info = getattr(getattr(self.detector, service_name, None), "info", {})
            fallback = getattr(pool, "_max_workers", 0)
            return info.get("process_workers", fallback) if isinstance(info, dict) else fallback

        def maximum(key):
            return max((piece["timings"].get(key, 0.) for piece in measured.values()), default=0.)

        last_piece = measured.get(last_3d, {})
        registration_worker = last_piece.get("registration", {}).get("registration_worker", {})
        measurement_worker = last_piece.get("joint_measurement_worker", {})
        return dict(piece_count=len(state.results), skipped_3d_count=len(state.results)-len(measured),
            registration_capacity=capacity("registration_process", self.registration_pool),
            measurement_capacity=capacity("joint_measurement_process", self.joint_pool),
            early_admission_limit=self.early_joint_workers,
            depth_ready_ms=(state.depth_ready_at-state.started)*1000,
            all_inputs_ready_ms=max((state.piece_timelines[pid]["inputs_ready_ms"] for pid in measured), default=0.),
            all_2d_diagnostics_ready_ms=last_2d_ms, all_3d_results_ready_ms=last_3d_ms,
            last_3d_piece_id=last_3d, last_2d_piece_id=last_2d,
            last_3d_registration_pid=registration_worker.get("pid"),
            last_3d_registration_cpu_ids=registration_worker.get("cpu_ids"),
            last_3d_measurement_pid=measurement_worker.get("pid"),
            last_3d_measurement_cpu_ids=measurement_worker.get("cpu_ids"),
            completion_gate="missing_2d" if len(state.entries) < len(state.results) else
                ("2d_diagnostics" if not measured or last_2d_ms > last_3d_ms else "3d_results"),
            wait_2d_after_3d_ms=max(0., last_2d_ms-last_3d_ms) if measured else 0.,
            slowest_piece_id=slowest,
            slowest_joint_pipeline_ms=measured.get(slowest, {}).get("timings", {}).get("joint_pipeline_ms", 0.),
            last_3d_joint_pipeline_ms=last_piece.get("timings", {}).get("joint_pipeline_ms", 0.),
            max_admission_wait_ms=maximum("joint_admission_wait_ms"),
            max_registration_process_wait_ms=maximum("registration_process_wait_ms"),
            max_measurement_process_wait_ms=maximum("joint_measurement_process_wait_ms"),
            max_registration_queue_ms=maximum("registration_queue_ms"),
            max_measurement_queue_ms=maximum("joint_measurement_queue_ms"),
            max_registration_phase_ms=maximum("registration_phase_ms"),
            max_measurement_phase_ms=maximum("joint_measurement_phase_ms"))

    def _terminate(self, state, response):
        if not self._current(state):
            return
        self._boards.pop(state.board_id)
        errors = []
        for inputs in (state.registration_inputs, *state.mask_inputs.values(),
                       *(job.registration_inputs for job in state.piece_requests.values())):
            try:
                inputs.close()
            except Exception as error:
                errors.append(error)
        if errors:
            response = self._failure(state.board_id, "registration_shared_memory_cleanup_failed", str(errors[0]))
        if state.completion is not None and not state.completion.done():
            state.completion.set_result(response)
        for piece_id, job in state.piece_requests.items():
            if not job.completion.done():
                job.completion.set_result(self._failure(
                    state.board_id, response["error"], response["message"], piece_id, cmd="detect_3d_piece"))
        state.piece_requests.clear()

    def release_board(self, board_id):
        with self._lock:
            state = self._boards.get(board_id)
            count = len(state.entries) if state else 0
            released_3d = bool(state and (state.depth_scan is not None or state.piece_requests))
            if state:
                self._terminate(state, self._failure(board_id, "board_released", "board cache released"))
        return dict(ok=True, cmd="release_board", stage="board", board_id=board_id,
                    released=dict(board_id=board_id, released_2d_count=count, released_3d=released_3d))

    def expire_pending(self):
        with self._lock:
            now = time.monotonic()
            for state in list(self._boards.values()):
                active = bool(state.inflight or state.depth_inflight or state.piece_requests
                              or state.claimed.difference(state.results))
                if not active and now - state.touched >= self.timeout_seconds:
                    scan = state.depth_scan
                    if scan is not None:
                        missing_ids = []
                        for piece in scan.pieces:
                            piece_id = int(piece["piece_id"])
                            if piece_id not in state.entries:
                                state.results[piece_id] = skipped_3d_piece(
                                    piece, scan.correction, "missing_2d_result")
                                missing_ids.append(piece_id)
                        self._complete_if_ready(state, missing_ids=sorted(missing_ids))
                        if not self._current(state):
                            continue
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
