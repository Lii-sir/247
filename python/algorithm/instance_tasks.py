"""Ordered leaf tasks; joint workers wait here, never on their own executor."""
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext

from algorithm.timing import measure, record_duration, timing_context, timing_metadata


class InstanceTasks:
    def __init__(self, workers=1):
        self.workers = 1
        self._pool = None
        self.configure(workers)

    def configure(self, workers):
        """Called during idle detector initialization, not during detection."""
        if type(workers) is not int or workers < 1:
            raise ValueError("instance_workers must be a positive integer")
        if workers == self.workers:
            return
        self.close()
        if workers > 1:
            self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="instance")
        self.workers = workers

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self.workers = 1

    def map(self, function, items, *, timing_stage=None):
        """Return input order; each child owns timings and output allocations.

        Child durations are cumulative work, not the parent phase wall time.
        On failure, drain running siblings before the caller can release inputs.
        """
        items = list(items)
        if self._pool is None or len(items) < 2:
            results = []
            for item in items:
                with measure(timing_stage) if timing_stage else nullcontext():
                    results.append(function(item))
            return results

        metadata = timing_metadata()
        child_timings = [{} for _ in items]

        def invoke(item, durations):
            with timing_context(durations, **metadata) if metadata else nullcontext():
                with measure(timing_stage) if timing_stage else nullcontext():
                    return function(item)

        futures = []
        try:
            for item, durations in zip(items, child_timings):
                futures.append(self._pool.submit(invoke, item, durations))
            return [future.result() for future in futures]
        finally:
            for future in futures:
                future.cancel()
            wait(futures)
            # Only this parent thread mutates the piece's timing dictionary.
            for durations in child_timings:
                for stage, duration in durations.items():
                    record_duration(stage, duration)
