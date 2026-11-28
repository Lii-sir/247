"""Model-free spawn worker. Keep module-level imports free of NumPy and Torch."""
from __future__ import annotations

import os
import sys
import time
import traceback

_startup = None
_warmup_gate = None


def initialize(assignments, assignment_index, ready, warmup_gate, role="registration", extra_imports=(), extra_info=None):
    global _startup, _warmup_gate
    try:
        # Set before importing numerical libraries; no inherited model imports are allowed.
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[key] = "1"
        sys.stdout = sys.stderr  # Never write into the parent's JSON protocol stream.
        with assignment_index.get_lock():
            assigned = assignments[assignment_index.value]
            assignment_index.value += 1
        if assigned is not None:
            from algorithm.execution.cpu_partition import set_cpus
            set_cpus(assigned)
        import cv2
        import numpy
        import scipy
        from algorithm.vision3d.alignment import registration
        from importlib import import_module
        for name in extra_imports:
            import_module(name)
        cv2.setNumThreads(1)
        if "torch" in sys.modules:
            raise RuntimeError("registration worker imported Torch; use the model-free Bridge entry point")
        actual = None
        if os.name == "nt":
            from algorithm.execution.cpu_partition import current_cpus
            actual = current_cpus()
        _startup = dict(pid=os.getpid(), role=role, cpu_ids=actual, assigned_cpu_ids=assigned,
                        executable=sys.executable, prefix=sys.prefix,
                        opencv_threads=cv2.getNumThreads(), torch_loaded=False,
                        native_threads={key: os.environ[key] for key in
                                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                        numpy=numpy.__version__, scipy=scipy.__version__, opencv=cv2.__version__)
        _startup.update(extra_info or {})
        _warmup_gate = warmup_gate
        ready.put(dict(ok=True, **_startup))
    except BaseException as error:
        ready.put(dict(ok=False, pid=os.getpid(), error=f"{type(error).__name__}: {error}"))
        raise


def warmup():
    if not _warmup_gate.wait(90):
        raise TimeoutError("registration process warmup gate timed out")
    return _startup


def run(payload):
    from multiprocessing.shared_memory import SharedMemory
    import numpy as np
    from algorithm.runtime.timing import timing_context, measure
    from algorithm.vision3d.alignment.registration import register_arrays
    handles, arrays, timings, logs = [], {}, {}, []
    started = time.perf_counter()
    local = None
    try:
        with timing_context(timings, phase="registration", **payload["metadata"], log=logs.append):
            with measure("registration_phase_ms"):
                for key, descriptor in payload["arrays"].items():
                    handle = SharedMemory(name=descriptor["name"])
                    handles.append(handle)
                    dtype, shape = np.dtype(descriptor["dtype"]), tuple(descriptor["shape"])
                    import math
                    if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape) or dtype.kind not in "ui" or math.prod(shape)*dtype.itemsize > handle.size:
                        raise ValueError("invalid registration shared-memory descriptor")
                    arrays[key] = np.ndarray(shape, dtype=dtype, buffer=handle.buf)
                    arrays[key].setflags(write=False)
                y1, y2 = payload["rows"]
                if not 0 <= y1 < y2 <= arrays["depth"].shape[0]:
                    raise ValueError("registration piece row bounds exceed shared depth")
                local = arrays["depth"][y1:y2, :]
                registration = register_arrays(arrays["gray"], arrays["mask"], local,
                    payload["registration_config"], payload["affine_config"], payload["correction"])
        return dict(ok=True, result=registration, timings=timings, logs=logs,
                    worker=_startup, started=started, finished=time.perf_counter())
    except Exception as error:
        failure = dict(ok=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                       timings=timings, logs=logs, started=started, finished=time.perf_counter())
        error.__traceback__ = None
        return failure
    finally:
        local = None
        arrays.clear()
        for handle in handles:
            handle.close()
