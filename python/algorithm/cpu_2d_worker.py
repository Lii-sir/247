"""Model-free 2D silver and public-geometry worker; no GPU objects cross IPC."""
import time
import traceback
from algorithm import registration_worker as bootstrap

warmup = bootstrap.warmup


def initialize(assignments, assignment_index, ready, warmup_gate, role="cpu_2d"):
    bootstrap.initialize(assignments, assignment_index, ready, warmup_gate, role,
                         ("algorithm.2D.silver_inspection", "algorithm.2D.instance_geometry"))


def compute(instances, shape, settings, crop_box, circle, boundary, base_dir, silver_prepare_ms=0.):
    from importlib import import_module
    from algorithm.timing import current_timings, timed_call, record_duration, measure
    silver = import_module("algorithm.2D.silver_inspection")
    geometry = import_module("algorithm.2D.instance_geometry")
    started = time.perf_counter()
    diagnostics = timed_call("silver_mask_checks_ms", silver.inspect_masks, instances, shape, settings,
        crop_box=crop_box, circle=circle, image=None, base_dir=base_dir, prepared_boundary=boundary)
    record_duration("silver_total_ms", current_timings().get("silver_mask_checks_ms", 0.)+silver_prepare_ms)
    diagnostics = timed_call("silver_geometry_offset_ms", silver.offset_geometry, diagnostics, crop_box[:2])
    mask_postprocess_ms = (time.perf_counter()-started)*1000

    def public():
        result = []
        for item in instances:
            with measure("public_2d_geometry_work_ms"):
                result.append(geometry.public_instance(item, crop_box))
        return result
    public_instances = timed_call("public_2d_geometry_ms", public)
    return dict(diagnostics=diagnostics, public_instances=public_instances, mask_postprocess_ms=mask_postprocess_ms)


def run(payload):
    import math
    from multiprocessing.shared_memory import SharedMemory
    import numpy as np
    from algorithm.timing import timing_context, measure
    arrays, handles, instances, timings, logs = {}, [], [], {}, []
    started = time.perf_counter()
    try:
        with timing_context(timings, phase="cpu_2d", **payload["metadata"], log=logs.append):
            with measure("cpu_2d_phase_ms"):
                for name, descriptor in payload["arrays"].items():
                    handle = SharedMemory(name=descriptor["name"])
                    handles.append(handle)
                    shape, dtype = tuple(descriptor["shape"]), np.dtype(descriptor["dtype"])
                    if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape) or dtype.kind not in "buif" or math.prod(shape)*dtype.itemsize > handle.size:
                        raise ValueError("invalid 2D shared-mask descriptor")
                    arrays[name] = np.ndarray(shape, dtype=dtype, buffer=handle.buf)
                    arrays[name].setflags(write=False)
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
                result = compute(instances, payload["shape"], payload["settings"], payload["crop_box"],
                    payload["circle"], payload["boundary"], payload["base_dir"], payload["silver_prepare_ms"])
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
        instances.clear()
        arrays.clear()
        for handle in handles:
            handle.close()
