"""Prepare per-piece shared masks for isolated 2D CPU postprocessing."""
from algorithm.registration_process import RegistrationProcesses, SharedBoardInputs
from algorithm import cpu_2d_worker


class Cpu2DProcesses(RegistrationProcesses):
    def __init__(self, workers, affinity):
        super().__init__(workers, affinity, worker_module=cpu_2d_worker, role="cpu_2d")
        self.info["shared_inputs"] = "one_readonly_mask_per_instance_no_image_copy"

    def submit_piece(self, preparation_pool, prepared, settings, base_dir, timings, metadata, is_current, log=None):
        inputs = SharedBoardInputs()
        def payload():
            from importlib import import_module
            boundary = prepared.get("silver_boundary")
            if boundary is None:
                raise RuntimeError("2D boundary preparation must finish before CPU process dispatch")
            arrays, records = import_module("algorithm.2D.instance_geometry").share_instances(
                prepared["joint_entry"]["instances"], inputs)
            return dict(arrays=arrays, instances=records,
                shape=list(prepared["crop"].shape[:2]), settings=settings,
                crop_box=tuple(prepared["crop_box"]), circle=prepared["circle"], boundary=boundary,
                base_dir=str(base_dir) if base_dir else None,
                silver_prepare_ms=timings.get("silver_prepare_phase_ms", 0.), metadata=metadata)
        return self.submit_task(preparation_pool, inputs, payload, timings, metadata, is_current,
                                log, close_inputs=True)
