"""Thread-local wall-clock profiling; diagnostic logs never use protocol stdout."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
import threading
import time

_context = ContextVar("bridge_timing", default=None)
_log_lock = threading.Lock()


def _enabled(name, default="0"):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def cuda_sync_enabled():
    """Additional EfficientAD barriers are opt-in, because they affect scheduling."""
    return _enabled("BRIDGE_TIMING_CUDA_SYNC") and _context.get() is not None


@contextmanager
def timing_context(timings, *, phase, board_id=None, piece_id=None, request_id=None, log=None):
    token = _context.set(dict(timings=timings, phase=phase, board_id=board_id,
                              piece_id=piece_id, request_id=request_id, log=log))
    try:
        yield
    finally:
        _context.reset(token)


def current_timings():
    context = _context.get()
    return context["timings"] if context is not None else {}


def note(event, **values):
    context = _context.get()
    if context is None or context["log"] is None or not _enabled("BRIDGE_TIMING_LOG", "1"):
        return
    record = {key: context[key] for key in ("phase", "board_id", "piece_id", "request_id")}
    record.update(event=event, pid=os.getpid(), thread=threading.current_thread().name,
                  clock_ms=round(time.perf_counter() * 1000, 3), **values)
    try:
        with _log_lock:
            context["log"]("[TIMING] " + json.dumps(record, ensure_ascii=True, allow_nan=False))
    except Exception:
        # A diagnostic sink must not change an inspection outcome.
        pass


def record_duration(stage, elapsed_ms):
    context = _context.get()
    if context is not None:
        context["timings"][stage] = context["timings"].get(stage, 0.0) + elapsed_ms
        note("duration", stage=stage, elapsed_ms=round(elapsed_ms, 3))


@contextmanager
def measure(stage):
    if _context.get() is None:
        yield
        return
    note("begin", stage=stage)
    started = time.perf_counter()
    error = None
    try:
        yield
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        elapsed = (time.perf_counter() - started) * 1000
        context = _context.get()
        context["timings"][stage] = context["timings"].get(stage, 0.0) + elapsed
        note("end", stage=stage, elapsed_ms=round(elapsed, 3), error=error)


def timed_call(stage, function, *args, **kwargs):
    with measure(stage):
        return function(*args, **kwargs)
