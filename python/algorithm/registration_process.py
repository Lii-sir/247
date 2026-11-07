"""Persistent model-free registration processes and leased per-board shared inputs."""
from __future__ import annotations

from concurrent.futures import Future, InvalidStateError, ProcessPoolExecutor, wait
from dataclasses import asdict
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import queue
import sys
import threading
import time

from algorithm import registration_worker

_spawn_lock = threading.Lock()


class SharedBoardInputs:
    """One owner plus in-flight leases. Unlink only after actual child completion."""
    def __init__(self):
        self._lock = threading.RLock()
        self._blocks = {}
        self._closed = False
        self._leases = 0

    def acquire(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("board shared inputs have been released")
            self._leases += 1

    def release(self):
        with self._lock:
            if self._leases < 1:
                raise RuntimeError("unbalanced registration input lease")
            self._leases -= 1
            self._cleanup()

    def close(self):
        with self._lock:
            self._closed = True
            self._cleanup()

    def _cleanup(self):
        if not self._closed or self._leases:
            return
        blocks, self._blocks = self._blocks, {}
        errors = []
        for handle, descriptor, identity in blocks.values():
            try:
                handle.close()
            except Exception as error:
                errors.append(error)
            finally:
                try:
                    handle.unlink()
                except FileNotFoundError:
                    pass
                except Exception as error:
                    errors.append(error)
        if errors:
            raise RuntimeError("registration shared-memory cleanup failed: "+str(errors[0]))

    def prepare(self, piece_id, gray, mask, depth):
        import numpy as np
        with self._lock:
            if self._closed:
                raise RuntimeError("board shared inputs have been released")
            result = {}
            for name, key, source in (("depth", "depth", depth), ("gray", (piece_id, "gray"), gray),
                                      ("mask", (piece_id, "mask"), mask)):
                if key not in self._blocks:
                    if not isinstance(source, np.ndarray) or source.ndim != 2 or not source.size or source.dtype.kind not in "ui":
                        raise ValueError(f"registration {name} must be a nonempty 2D integer array")
                    handle = SharedMemory(create=True, size=source.nbytes)
                    try:
                        target = np.ndarray(source.shape, dtype=source.dtype, buffer=handle.buf)
                        target[:] = source
                        del target
                        descriptor = dict(name=handle.name, dtype=source.dtype.str, shape=list(source.shape))
                        self._blocks[key] = (handle, descriptor, id(source))
                    except BaseException:
                        handle.close()
                        handle.unlink()
                        raise
                handle, descriptor, identity = self._blocks[key]
                if identity != id(source):
                    raise ValueError("registration shared inputs changed within a board generation")
                result[name] = descriptor
            return result


class RegistrationProcesses:
    """Parent preparation threads never wait for computation in child processes."""
    def __init__(self, workers, affinity, *, startup_timeout=60):
        self._lock = threading.RLock()
        self._pending = set()
        self._closed = False
        self._executor = None
        self._restore_cpus = None
        self._queues = []
        started = time.perf_counter()
        partition = None
        if affinity.get("enabled", False):
            from algorithm.cpu_partition import current_cpus, physical_cores, choose_partition
            partition = choose_partition(physical_cores(), current_cpus(), affinity["physical_cores"])
            if workers > len(partition["registration_cores"]):
                raise ValueError("registration_workers cannot exceed registration_affinity.physical_cores")
        ctx = mp.get_context("spawn")
        assignments, ready, gate = [], ctx.Queue(), ctx.Event()
        assignment_index = ctx.Value("i", 0)
        self._queues = [ready]
        for index in range(workers):
            cpus = sorted(cpu for core in partition["registration_cores"][index::workers]
                          for cpu in core["logical_cpus"]) if partition else None
            assignments.append(cpus)
        try:
            # Spawn every process while using pythonw, then restore the global
            # executable setting. Subsequent jobs reuse these already-running workers.
            from multiprocessing.spawn import get_executable
            with _spawn_lock:
                original_executable = get_executable()
                original_launcher = os.environ.get("__PYVENV_LAUNCHER__")
                try:
                    executable = Path(sys.executable)
                    if sys.platform == "win32":
                        # Bypass the venv redirector exactly as multiprocessing's
                        # Windows Popen does for python.exe. Handles duplicated
                        # into a redirector do not belong to the real worker.
                        executable = Path(sys._base_executable).with_name("pythonw.exe")
                        if not executable.is_file():
                            raise RuntimeError("pythonw.exe is required for hidden registration workers")
                        os.environ["__PYVENV_LAUNCHER__"] = str(Path(sys.executable).with_name("pythonw.exe"))
                    mp.set_executable(str(executable))
                    self._executor = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                        initializer=registration_worker.initialize, initargs=(assignments, assignment_index, ready, gate))
                    warmups = [self._executor.submit(registration_worker.warmup) for _ in range(workers)]
                    deadline, metadata = time.monotonic()+startup_timeout, []
                    for _ in range(workers):
                        try:
                            worker = ready.get(timeout=max(.01, deadline-time.monotonic()))
                        except queue.Empty as error:
                            raise RuntimeError("registration worker startup timed out") from error
                        if not worker["ok"]:
                            raise RuntimeError("registration worker startup failed: "+worker["error"])
                        metadata.append(worker)
                    if len({worker["pid"] for worker in metadata}) != workers:
                        raise RuntimeError("registration worker warmup did not start every process")
                    gate.set()
                    for future in warmups:
                        future.result(timeout=max(.01, deadline-time.monotonic()))
                finally:
                    mp.set_executable(original_executable)
                    if original_launcher is None:
                        os.environ.pop("__PYVENV_LAUNCHER__", None)
                    else:
                        os.environ["__PYVENV_LAUNCHER__"] = original_launcher
            if partition:
                from algorithm.cpu_partition import set_cpus
                self._restore_cpus = partition["original_cpus"]
                set_cpus(partition["main_cpus"])
            self.info = dict(backend="process", process_workers=workers, workers=metadata,
                             affinity=partition, startup_ms=(time.perf_counter()-started)*1000,
                             shared_inputs="one_depth_per_board_and_one_gray_mask_per_piece")
        except BaseException:
            gate.set()
            # Python 3.10 has no public terminate_workers(). On startup failure
            # terminate only this executor's children so shutdown cannot hang
            # on a worker that never completed its initializer.
            processes = (getattr(self._executor, "_processes", None) or {}).copy()
            for process in processes.values():
                if process.is_alive():
                    process.terminate()
            self.close()
            raise

    def submit(self, preparation_pool, inputs, piece_id, entry, depth, piece, correction,
               registration_config, affine_config, timings, metadata, is_current, log=None):
        submitted, output, completion = time.perf_counter(), Future(), Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("registration process service is closed")
            inputs.acquire()
            self._pending.add(completion)

        def finish(result=None, error=None):
            # A cancelled public Future must not free memory while the actual
            # process Future is still running. This runs only after it finishes.
            try:
                try:
                    inputs.release()
                except Exception as cleanup_error:
                    error = cleanup_error
                try:
                    if error is not None:
                        output.set_exception(error)
                    else:
                        output.set_result(result)
                except InvalidStateError:
                    pass  # Caller cancellation does not cancel the underlying lease.
            finally:
                completion.set_result(None)
                with self._lock:
                    self._pending.discard(completion)

        def received(future):
            try:
                packet = future.result()
                received_at = time.perf_counter()
                timings.update(packet["timings"])
                timings["registration_queue_ms"] = (packet["started"]-submitted)*1000
                timings["registration_result_transfer_ms"] = (received_at-packet["finished"])*1000
                timings["registration_process_roundtrip_ms"] = (received_at-submitted)*1000
                relay_started = time.perf_counter()
                if log is not None:
                    for line in packet["logs"]:
                        try:
                            log(line)
                        except Exception:
                            pass
                timings["registration_log_relay_ms"] = (time.perf_counter()-relay_started)*1000
                if not packet["ok"]:
                    raise RuntimeError(packet["error"]+"\n"+packet["traceback"])
                result = packet["registration"]
                result["registration_worker"] = packet["worker"]
            except Exception as error:
                finish(error=error)
            else:
                finish(result=result)

        def prepare():
            timings["registration_dispatch_queue_ms"] = (time.perf_counter()-submitted)*1000
            if not is_current():
                raise RuntimeError("board was released before process registration preparation")
            started = time.perf_counter()
            arrays = inputs.prepare(piece_id, entry["gray_image"], entry["registration_mask"], depth)
            timings["registration_shared_memory_ms"] = (time.perf_counter()-started)*1000
            return dict(arrays=arrays, rows=[int(piece["y1"]), int(piece["y2"])],
                        registration_config=asdict(registration_config), affine_config=affine_config,
                        correction=correction, metadata=metadata)

        def prepared(future):
            try:
                payload = future.result()
                if not is_current():
                    raise RuntimeError("board was released before process registration dispatch")
                with self._lock:
                    if self._closed:
                        raise RuntimeError("registration process service closed before dispatch")
                    work = self._executor.submit(registration_worker.run, payload)
                work.add_done_callback(received)
            except Exception as error:
                finish(error=error)

        try:
            preparation_pool.submit(prepare).add_done_callback(prepared)
        except Exception as error:
            finish(error=error)
        return output

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = list(self._pending)
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            if pending:
                wait(pending)
        finally:
            try:
                if self._restore_cpus is not None:
                    from algorithm.cpu_partition import set_cpus
                    set_cpus(self._restore_cpus)
                    self._restore_cpus = None
            finally:
                for channel in self._queues:
                    channel.close()
                    channel.join_thread()
