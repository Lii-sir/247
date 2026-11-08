"""Isolated 3D mapping/height submission using the board's leased shared depth."""
from dataclasses import asdict
from importlib import import_module
from algorithm.registration_process import RegistrationProcesses
from algorithm import joint_worker


class JointMeasurementProcesses(RegistrationProcesses):
    def __init__(self, workers, affinity, instance_workers=1):
        super().__init__(workers, affinity, worker_module=joint_worker, role="joint_measurement",
                         worker_options=dict(instance_workers=instance_workers))
        self.info["instance_workers_per_process"] = instance_workers
        self.info["shared_inputs"] = "reuse_board_depth_plus_readonly_source_instance_masks"

    def submit_piece(self, preparation_pool, inputs, piece_id, entry, depth, piece, registration,
                     correction, height_config, min_component_area, crop_roi, wire_rois,
                     timings, metadata, is_current, log=None):
        def payload():
            arrays, instances = import_module("algorithm.2D.instance_geometry").share_instances(
                entry["instances"], inputs, arrays=dict(depth=depth), cache_keys=dict(depth="depth"),
                cache_prefix=("instance_mask", piece_id))
            return dict(arrays=arrays, instances=instances, piece_id=piece_id, piece=dict(piece),
                registration=registration, correction=correction, height_config=asdict(height_config),
                min_component_area=min_component_area, crop_roi=tuple(crop_roi), wire_rois=list(wire_rois),
                entry={key: entry[key] for key in ("crop_box", "circle", "roi_boxes", "image_shape", "wire_roi_checks")
                       if key in entry}, metadata=metadata)
        return self.submit_task(preparation_pool, inputs, payload, timings, metadata, is_current, log)
