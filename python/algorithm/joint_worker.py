"""Model-free mapping and height worker with its own local instance executor."""
import math
import sys
import time
import traceback
from algorithm import registration_worker as bootstrap

_runtime = None
warmup = bootstrap.warmup


def initialize(assignments, assignment_index, ready, warmup_gate, role="joint_measurement", options=None):
    global _runtime
    workers = (options or {}).get("instance_workers", 1)
    bootstrap.initialize(assignments, assignment_index, ready, warmup_gate, role,
                         extra_imports=("algorithm.joint_geometry",), extra_info=dict(instance_workers=workers))
    from algorithm.joint_geometry import JointGeometry
    _runtime = JointGeometry(workers)


def run(payload):
    from multiprocessing.shared_memory import SharedMemory
    import numpy as np
    from algorithm.height_measurement import HeightConfig
    from algorithm.timing import timing_context, measure
    arrays, handles, instances, timings, logs = {}, [], [], {}, []
    started = time.perf_counter()
    try:
        with timing_context(timings, phase="joint_measurement", **payload["metadata"], log=logs.append):
            with measure("joint_measurement_phase_ms"):
                for name, descriptor in payload["arrays"].items():
                    handle = SharedMemory(name=descriptor["name"])
                    handles.append(handle)
                    dtype, shape = np.dtype(descriptor["dtype"]), tuple(descriptor["shape"])
                    if (len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape)
                            or dtype.kind not in "buif" or math.prod(shape)*dtype.itemsize > handle.size):
                        raise ValueError("invalid joint shared-array descriptor")
                    arrays[name] = np.ndarray(shape, dtype=dtype, buffer=handle.buf)
                    arrays[name].setflags(write=False)
                y1, y2 = int(payload["piece"]["y1"]), int(payload["piece"]["y2"])
                if not 0 <= y1 < y2 <= arrays["depth"].shape[0]:
                    raise ValueError("joint piece row bounds exceed shared board depth")
                for record in payload["instances"]:
                    item = dict(record["fields"])
                    if "mask_key" in record:
                        item["mask"] = arrays[record["mask_key"]]
                    elif "empty_mask" in record:
                        shape, dtype = record["empty_mask"]
                        item["mask"] = np.empty(tuple(shape), dtype=np.dtype(dtype))
                    else:
                        item["mask"] = record.get("mask_value")
                    instances.append(item)
                _runtime.height_config = HeightConfig.from_dict(payload["height_config"])
                _runtime.min_component_area = payload["min_component_area"]
                _runtime.crop_roi = tuple(payload["crop_roi"])
                _runtime.wire_rois = payload["wire_rois"]
                entry = {**payload["entry"], "instances": instances}
                result = _runtime.measure_registered_piece(payload["piece_id"], entry, arrays["depth"],
                    payload["piece"], payload["registration"], payload["correction"])
                if "torch" in sys.modules:
                    raise RuntimeError("3D measurement worker unexpectedly loaded Torch")
        return dict(ok=True, result=result, timings=timings, logs=logs, worker=bootstrap._startup,
                    started=started, finished=time.perf_counter())
    except Exception as error:
        failure = dict(ok=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                       timings=timings, logs=logs, started=started, finished=time.perf_counter())
        error.__traceback__ = None
        return failure
    finally:
        if "item" in locals():
            item = None
        if "entry" in locals():
            entry = None
        instances.clear()
        arrays.clear()
        for handle in handles:
            handle.close()
