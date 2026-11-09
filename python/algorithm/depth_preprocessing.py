"""Caller-side depth preparation. Complete this before submitting detection."""
from algorithm.tilt_correction import correct_smartray_depth, production_pitch_parameters
from transport.depth_input import input_depth_correction
from algorithm.instance_tasks import InstanceTasks
import threading
import numpy as np


def preprocessing_parameters(config):
    from algorithm.config import feature_parameters
    height = feature_parameters(config, 'height_measurement_3d', required=False) or {}
    return production_pitch_parameters(config, float(height.get('gray_offset', 32768.)),
                                       float(height.get('vertical_resolution_mm', .00105)))


def prepare_depth_for_detection(depth, settings, *, tasks=None, out=None):
    corrected, info = correct_smartray_depth(depth, settings, tasks=tasks, out=out)
    return corrected, input_depth_correction(dict(pitch_correction=info))


class DepthPreprocessor:
    """Caller-owned reusable worker pool and encoding buffer, outside detection.

    prepare() returns independently owned pixels unless the caller supplies out.
    encode() reuses private storage and snapshots bytes under the same lock, so
    a later/concurrent frame cannot overwrite a pending request's byte payload.
    """
    def __init__(self, settings):
        self._settings = dict(settings or {})
        workers = self._settings.get('projection_workers', 1) if self._settings.get('enabled_3d') else 1
        self._tasks = InstanceTasks(workers, thread_name_prefix='pitch-input')
        self._lock = threading.RLock()
        self._buffer = None
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError('depth preprocessor has been closed')

    def prepare(self, depth, *, out=None):
        with self._lock:
            self._check_open()
            return prepare_depth_for_detection(depth, self._settings, tasks=self._tasks, out=out)

    def encode(self, depth):
        with self._lock:
            self._check_open()
            if self._settings.get('enabled_3d'):
                if self._buffer is None or self._buffer.shape != depth.shape:
                    self._buffer = np.empty(depth.shape, np.uint16)
                out = self._buffer
            else:
                out = None
            corrected, info = self.prepare(depth, out=out)
            return corrected.astype('<u2', copy=False).tobytes(order='C'), info

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._tasks.close()
                self._buffer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
