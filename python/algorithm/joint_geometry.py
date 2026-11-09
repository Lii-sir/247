"""Model-free 3D mapping and height pipeline shared with the board detector."""
from __future__ import annotations
import sys
from importlib import import_module
from typing import Any, Dict, Sequence
import cv2
import numpy as np
from algorithm.height_measurement import (HeightConfig, dilate_mask, erode_wire_mask_in_place,
    erode_measurement_mask_crop, fit_chip_plane, fit_local_platform_plane, fit_platform_plane,
    height_points_to_board, map_platform_rois, measure_mask_height, split_thin_mask)
from algorithm.registration import align_crop_mask_to_depth
from transport.depth_input import depth_coordinate_space
from algorithm.instance_tasks import InstanceTasks
from algorithm.timing import timed_call
_geometry = import_module("algorithm.2D.defect_geometry")
_mask_geometry = import_module("algorithm.2D.instance_geometry").mask_geometry


class JointGeometry:
    def __init__(self, instance_workers=1):
        self._joint_ops = sys.modules[__name__]
        self._instance_tasks = InstanceTasks(instance_workers, thread_name_prefix="joint-instance")
        self.height_config = HeightConfig()
        self.min_component_area = 100
        self.crop_roi = (0, 0, 1, 1)
        self.wire_rois = []

    @staticmethod
    def finalize_joint_piece(piece, entry):
        # Full 2D diagnostics are joined by the parent after both branches end.
        return piece

    def measure_registered_piece(self, piece_id: int, entry: Dict[str, Any],
                                 depth: np.ndarray, piece: Dict[str, Any],
                                 registration: Dict[str, Any],
                                 pitch_correction: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Consume the fitted matrix; own all mapped masks and measurement outputs."""
        y1, y2 = int(piece["y1"]), int(piece["y2"])
        local_depth = depth[y1:y2]
        aligned_masks = timed_call("mask_alignment_ms", self._align_instances,
            entry["instances"], local_depth.shape, registration, entry.get("crop_box", self.crop_roi))
        measurement = timed_call("height_measurement_ms", self._measure_piece,
            local_depth,
            aligned_masks,
            registration,
            crop_box=entry.get("crop_box", self.crop_roi),
            circle=entry.get("circle"),
        )
        public_masks = timed_call("public_3d_geometry_ms", lambda: [self._public_instance(item) for item in aligned_masks])
        result = timed_call("height_points_to_board_ms", self._joint_ops.height_points_to_board, {
            "piece_id": piece_id,
            "image_2d_size": entry["image_shape"],
            "crop_2d": list(entry.get("crop_box", self.crop_roi)),
            "wire_rois_2d": [list(roi) for roi in entry.get("roi_boxes", self.wire_rois)],
            "wire_roi_checks": entry.get("wire_roi_checks", []),
            "depth_start_row": y1,
            "depth_end_row": y2,
            "pitch_correction": pitch_correction or {},
            "depth_coordinate_space": self._joint_ops.depth_coordinate_space(pitch_correction or {}),
            "registration": registration,
            "is_ok": bool(
                measurement["is_ok"]
                and all(item.get("is_ok") is True for item in entry.get("wire_roi_checks", []))
            ),
            "reference_planes": measurement["reference_planes"],
            "wire_measurements": measurement["wire_measurements"],
            "solder_measurements": measurement["solder_measurements"],
            "masks": public_masks,
        })
        return self.finalize_joint_piece(result, entry)

    def _align_instances(self, instances, depth_shape, registration, crop_box=None):
        return self._instance_tasks.map(
            lambda item: self._align_instance(item, depth_shape, registration, crop_box),
            instances, timing_stage="instance_alignment_work_ms",
        )

    def _align_instance(
        self,
        instance: Dict[str, Any],
        depth_shape: tuple[int, int],
        registration: Dict[str, Any],
        crop_box: Sequence[int] | None = None,
    ) -> Dict[str, Any]:
        crop_mask = instance["mask"]
        crop_box = tuple(crop_box or self.crop_roi)
        aligned = self._joint_ops.align_crop_mask_to_depth(
            crop_mask, crop_box, depth_shape, registration,
            binary_255=instance.get("_mask_binary_255") is True,
        )
        return {
            **instance,
            "mask_2d": self._joint_ops._mask_geometry(crop_mask, offset=crop_box[:2], epsilon=2.0),
            "mask_3d_local": self._joint_ops._mask_geometry(aligned, offset=(0, 0), epsilon=1.0),
            "_mask_2d_array": crop_mask,
            "_mask_3d_array": aligned,
        }

    def _measure_piece(
        self,
        depth: np.ndarray,
        instances: list[Dict[str, Any]],
        registration: Dict[str, Any],
        crop_box: Sequence[int] | None = None,
        circle: Sequence[float] | None = None,
    ) -> Dict[str, Any]:
        """Measure every detected wire/thin/bond, without a target-selection ROI.

        ``thin`` is both a wire and two solder joints: its first and last
        principal-axis 2 percent are measured as solder and the middle 96
        percent is measured as a wire.  Every reference plane is fitted after
        removing the complete aligned semantic mask union.  The chip itself
        is kept in the input region while fitting its own plane, but all
        other semantic masks are excluded.
        """
        crop_box = tuple(crop_box or self.crop_roi)
        semantic_exclusion = timed_call("height_semantic_exclusion_ms", self._semantic_exclusion_mask, instances, depth.shape)
        platform_plane = None
        platform_diagnostics: Dict[str, Any] = {
            "status": "not_configured", "error": "platform_roi_offsets is empty",
            "boxes_2d": [], "polygons_3d_local": [],
            "sampling_mask_3d_local": self._joint_ops._mask_geometry(
                np.zeros(depth.shape, dtype=np.uint8), offset=(0, 0), epsilon=0.0
            ),
        }
        try:
            if self.height_config.platform_roi_offsets:
                if circle is None:
                    raise ValueError("detected 2D circle is unavailable")
                if registration.get("is_valid") is False:
                    raise ValueError("2D/3D registration is invalid")
                if registration.get("is_valid") is not True:
                    raise ValueError("piece affine registration is invalid")
                roi_mask, boxes, polygons = self._joint_ops.map_platform_rois(
                    circle, self.height_config.platform_roi_offsets,
                    registration, depth.shape,
                )
                sampling_mask = (roi_mask > 0) & (semantic_exclusion == 0)
                sampling_mask &= ~np.isin(depth, self.height_config.invalid_values)
                platform_diagnostics.update(
                    boxes_2d=boxes, polygons_3d_local=polygons,
                    roi_area_px=int(np.count_nonzero(roi_mask)),
                    candidate_points=int(np.count_nonzero(sampling_mask)),
                    sampling_mask_3d_local=self._joint_ops._mask_geometry(
                        sampling_mask, offset=(0, 0), epsilon=0.0
                    ),
                )
                platform_plane = timed_call("height_fit_platform_plane_ms", self._joint_ops.fit_platform_plane,
                    depth, self.height_config, roi_mask, exclude_mask=semantic_exclusion
                )
                platform_diagnostics.update(status="ok", error=None)
        except ValueError as exc:
            platform_diagnostics.update(status="invalid", error=str(exc))
        chips = [
            item for item in instances if item["class_name"].lower() == "chip"
        ]
        timed_call("height_prepare_chip_regions_ms", self._prepare_chip_regions, chips)
        chip_planes: Dict[int, Dict[str, Any] | None] = {}
        for chip in chips:
            # Build this exclusion union without the current chip instance.
            # Resetting its pixels in an already-unioned mask would also
            # restore a bond/wire that overlaps the chip mask and bias the
            # chip reference plane upward.
            chip_exclusion = self._semantic_exclusion_mask(
                instances,
                depth.shape,
                skip_instance_id=chip["instance_id"],
            )
            try:
                chip_planes[chip["instance_id"]] = timed_call("height_fit_chip_plane_ms", self._joint_ops.fit_chip_plane,
                    depth,
                    chip["_mask_3d_array"],
                    self.height_config,
                    exclude_mask=chip_exclusion,
                )
            except ValueError:
                chip_planes[chip["instance_id"]] = None

        wire_measurements = timed_call("height_wire_regions_ms", self._measure_wire_regions,
            depth,
            instances,
            registration,
            platform_plane,
            crop_box=crop_box,
        )
        wire_by_instance = {
            int(item["instance_id"]): item for item in wire_measurements
        }

        solder_jobs = []
        for instance in instances:
            class_name = instance["class_name"].lower()
            if class_name == "thin":
                solder_jobs.extend((instance, endpoint, mask) for endpoint, mask in
                    zip(("start", "end"), instance["_thin_solder_masks_2d"]))
            elif class_name == "bond":
                solder_jobs.append((instance, "bond", instance["_mask_2d_array"]))
        solder_measurements = timed_call("height_solder_wall_ms", self._instance_tasks.map,
            lambda job: self._measure_solder_task(job, depth, registration, chips, chip_planes,
                platform_plane, semantic_exclusion, crop_box),
            solder_jobs, timing_stage="height_solder_work_ms")
        bond_by_instance = {item["source_instance_id"]: item for item in solder_measurements
                            if item["endpoint"] == "bond"}
        # All child tasks have finished. Only this parent updates instances.
        for instance in instances:
            class_name = instance["class_name"].lower()
            if class_name in {"thin", "wire"}:
                wire_result = wire_by_instance.get(int(instance["instance_id"]))
                instance["height_3d"] = wire_result["height_3d"] if wire_result is not None else None
            elif class_name == "bond":
                solder_result = bond_by_instance[instance["instance_id"]]
                instance["height_3d"] = solder_result["height_3d"]
                instance["bond_location"] = solder_result["location"]
                instance["reference_instance_id"] = solder_result["reference_instance_id"]
                instance["chip_inside"] = solder_result["chip_inside"]
            elif class_name == "chip":
                instance["height_3d"] = None
                instance["plane_3d"] = chip_planes.get(instance["instance_id"])
            else:
                instance["height_3d"] = None

        wire_ok = [
            bool(item["height_3d"].get("is_ok"))
            for item in wire_measurements
        ]
        solder_ok = [bool(item["height_3d"].get("is_ok")) for item in solder_measurements]
        return {
            "is_ok": bool(wire_measurements or solder_measurements)
            and all(wire_ok + solder_ok),
            "wire_measurements": wire_measurements,
            "solder_measurements": solder_measurements,
            "reference_planes": {
                "wire_platform": platform_plane,
                "platform_roi": platform_diagnostics,
                "chips": [
                    {
                        "instance_id": chip["instance_id"],
                        "plane": chip_planes.get(chip["instance_id"]),
                    }
                    for chip in chips
                ],
            },
        }

    def _semantic_exclusion_mask(
        self,
        instances: Sequence[Dict[str, Any]],
        shape: tuple[int, int],
        skip_instance_id: int | None = None,
    ) -> np.ndarray:
        """Return a dilated union of all aligned YOLO masks."""
        union = np.zeros(shape, dtype=np.uint8)
        for item in instances:
            if skip_instance_id is not None and item.get("instance_id") == skip_instance_id:
                continue
            mask = np.asarray(item.get("_mask_3d_array"), dtype=np.uint8)
            if mask.shape == shape:
                union = np.maximum(union, mask)
        return self._joint_ops.dilate_mask(union, self.height_config.semantic_exclusion_dilation_px)

    @staticmethod
    def _prepare_chip_regions(chips, *, fill=None):
        # Refresh once per piece; ownership workers never populate caches.
        for chip in chips:
            region = (fill or _filled_external_region)(chip["_mask_2d_array"])
            chip["_filled_chip_region_2d"] = region
            chip["_filled_chip_area_2d"] = int(np.count_nonzero(region))

    def _measure_solder_task(self, job, depth, registration, chips, chip_planes,
                             platform_plane, semantic_exclusion, crop_box):
        instance, endpoint, mask_2d = job
        if endpoint == "bond":
            mask_3d = instance["_mask_3d_array"]
        else:
            # split_thin_mask creates its own uint8 0/255 endpoint masks.
            mask_3d = self._joint_ops.align_crop_mask_to_depth(
                mask_2d, crop_box, depth.shape, registration, binary_255=True)
            # Expand after mapping: this radius is measured in depth pixels.
            mask_3d = self._joint_ops.dilate_mask(mask_3d, self.height_config.thin_solder_search_dilation_px)
        return timed_call("height_solder_ms", self._measure_solder,
            depth, mask_2d, mask_3d, instance, endpoint, chips, chip_planes,
            platform_plane, semantic_exclusion, crop_box)

    def _measure_solder(
        self,
        depth: np.ndarray,
        mask_2d: np.ndarray,
        mask_3d: np.ndarray,
        instance: Dict[str, Any],
        endpoint: str,
        chips: list[Dict[str, Any]],
        chip_planes: Dict[int, Dict[str, Any] | None],
        platform_plane: Dict[str, Any] | None,
        semantic_exclusion: np.ndarray,
        crop_box: Sequence[int],
    ) -> Dict[str, Any]:
        chip, inside = self._matching_chip(
            {"_mask_2d_array": mask_2d}, chips
        )
        on_chip = chip is not None and inside
        if on_chip:
            plane = chip_planes.get(chip["instance_id"])
            reference = "chip"
            location = "chip"
            reference_instance_id = chip["instance_id"]
            max_height = self.height_config.max_solder_chip_height_mm
        else:
            try:
                plane = self._joint_ops.fit_local_platform_plane(
                    depth, mask_3d, semantic_exclusion, self.height_config
                )
            except ValueError:
                plane = None
            reference = "local_wire_platform" if plane is not None else "wire_platform"
            location = "wire_platform"
            reference_instance_id = None
            if plane is None:
                plane = platform_plane
            max_height = self.height_config.max_solder_platform_height_mm
        # Select reference/chip ownership using the original region above.
        # Solder masks may alias instance masks: erode only an independent bbox.
        erode_px = (self.height_config.bond_measurement_erode_px
                    if endpoint == "bond" else self.height_config.thin_solder_measurement_erode_px)
        measurement_mask, measurement_offset = self._joint_ops.erode_measurement_mask_crop(mask_3d, erode_px)
        height = self._joint_ops.measure_mask_height(
            depth,
            measurement_mask,
            plane,
            self.height_config,
            reference=reference,
            max_height_mm=max_height,
            mask_offset=measurement_offset,
        )
        return {
            "measurement_id": int(instance.get("instance_id", 0)) * 3 + {
                "bond": 0,
                "start": 1,
                "end": 2,
            }.get(endpoint, 0),
            "source_instance_id": int(instance.get("instance_id", 0)),
            "class_name": str(instance.get("class_name", "")),
            "endpoint": endpoint,
            "location": location,
            "reference_instance_id": reference_instance_id,
            "chip_inside": bool(inside),
            "reference_plane": plane,
            "mask_2d": self._joint_ops._mask_geometry(mask_2d, offset=crop_box[:2], epsilon=1.0),
            "mask_3d_local": self._joint_ops._mask_geometry(measurement_mask, offset=measurement_offset, epsilon=1.0),
            "height_3d": height,
        }

    def _prepare_thin_parts(self, wires):
        # Owned by this piece; children only read these arrays. Rebuild once
        # per measurement so a later run cannot reuse a previous split ratio.
        for wire in wires:
            if wire["class_name"].lower() == "thin":
                middle, endpoints = self._joint_ops.split_thin_mask(
                    wire["_mask_2d_array"], self.height_config.thin_solder_end_ratio)
                wire["_thin_wire_mask_2d"] = middle
                wire["_thin_solder_masks_2d"] = endpoints

    def _measure_wire_regions(
        self,
        depth: np.ndarray,
        instances: list[Dict[str, Any]],
        registration: Dict[str, Any],
        platform_plane: Dict[str, Any] | None,
        crop_box: Sequence[int] | None = None,
    ) -> list[Dict[str, Any]]:
        crop_box = tuple(crop_box or self.crop_roi)
        wires = [item for item in instances if item["class_name"].lower() in {"wire", "thin"}]
        if not wires:
            return []
        timed_call("height_prepare_thin_ms", self._prepare_thin_parts, wires)
        results = self._instance_tasks.map(
            lambda wire: self._measure_wire_region(depth, wire, registration,
                                                   platform_plane, crop_box),
            wires, timing_stage="height_wire_work_ms",
        )
        for region_id, item in enumerate(results, 1):
            item["region_id"] = region_id
        return results

    def _measure_wire_region(self, depth, wire, registration, platform_plane, crop_box):
        class_name = wire["class_name"].lower()
        source_mask = np.asarray(wire["_thin_wire_mask_2d"] if class_name == "thin"
                                 else wire["_mask_2d_array"], dtype=np.uint8)
        # Clean the complete instance within its bbox, preserving morphology
        # halo. The 2D bending/presence ROIs never filter or clip height pixels.
        cleaned, local_offset = self._joint_ops._geometry.clean_mask_crop(
            source_mask, min_area=self.min_component_area, kernel_size=5)
        full_offset = np.asarray(crop_box[:2], dtype=np.int32) + local_offset
        bending = wire.get("bending_2d")
        if not isinstance(bending, dict):
            bending = {"is_valid": False, "is_ok": False, "status": "not_evaluated"}
        if not np.any(cleaned):
            empty = np.zeros((1, 1), dtype=np.uint8)
            height_result = self._joint_ops.measure_mask_height(
                depth, empty, platform_plane, self.height_config, reference="wire_platform",
                max_height_mm=self.height_config.max_wire_platform_height_mm)
            height_result["is_ok"] = False
            return dict(instance_id=int(wire.get("instance_id", 0)),
                class_name=str(wire.get("class_name", "")),
                mask_2d=self._joint_ops._mask_geometry(cleaned, offset=full_offset, epsilon=1.0),
                mask_3d_local=self._joint_ops._mask_geometry(empty, offset=(0, 0), epsilon=1.0),
                bending_2d=bending,
                height_3d=height_result)
        local_crop = (int(full_offset[0]), int(full_offset[1]),
                      int(full_offset[0] + cleaned.shape[1]), int(full_offset[1] + cleaned.shape[0]))
        aligned = self._joint_ops.align_crop_mask_to_depth(cleaned.astype(np.uint8) * 255,
                                           local_crop, depth.shape, registration, binary_255=True)
        erode_px = (self.height_config.thin_measurement_erode_px if class_name == "thin"
                    else self.height_config.wire_measurement_erode_px)
        if erode_px:
            self._joint_ops.erode_wire_mask_in_place(aligned, erode_px)
        height_result = self._joint_ops.measure_mask_height(depth, aligned, platform_plane, self.height_config,
            reference="wire_platform", max_height_mm=self.height_config.max_wire_platform_height_mm)
        return dict(instance_id=int(wire.get("instance_id", 0)),
            class_name=str(wire.get("class_name", "")),
            mask_2d=self._joint_ops._mask_geometry(cleaned, offset=full_offset, epsilon=2.0),
            mask_3d_local=self._joint_ops._mask_geometry(aligned, offset=(0, 0), epsilon=1.0),
            bending_2d=bending, height_3d=height_result)

    @staticmethod
    def _matching_chip(
        bond: Dict[str, Any],
        chips: list[Dict[str, Any]],
        *, fill=None,
    ) -> tuple[Dict[str, Any] | None, bool]:
        bond_mask = bond["_mask_2d_array"] > 0
        ys, xs = np.nonzero(bond_mask)
        if len(xs) == 0 or not chips:
            return None, False
        center_x = int(round(float(xs.mean())))
        center_y = int(round(float(ys.mean())))
        best_chip = None
        best_area = None
        for chip in chips:
            # Normal piece execution precomputes these caches in the parent.
            # Direct callers without preparation get a read-only fallback.
            chip_region = chip.get("_filled_chip_region_2d")
            if chip_region is None:
                chip_region = (fill or _filled_external_region)(chip["_mask_2d_array"])
            if chip_region[center_y, center_x]:
                area = chip.get("_filled_chip_area_2d")
                if area is None:
                    area = int(np.count_nonzero(chip_region))
                if best_area is None or area < best_area:
                    best_chip, best_area = chip, area
        return best_chip, best_chip is not None

    @staticmethod
    def _public_instance(instance: Dict[str, Any]) -> Dict[str, Any]:
        return {
            ("chip_rotation_2d" if key == "chip_alignment" else key): value
            for key, value in instance.items()
            if key != "mask" and not key.startswith("_")
        }


def _filled_external_region(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    if contours:
        points = np.concatenate(contours, axis=0)
        cv2.fillConvexPoly(filled, cv2.convexHull(points), 255)
    return filled > 0
