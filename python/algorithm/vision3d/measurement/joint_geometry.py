"""Model-free 3D mapping and height pipeline shared with the board detector."""
from __future__ import annotations
import sys
from functools import cache
from typing import Any, Dict, Sequence
import cv2
import numpy as np
from algorithm.vision3d.measurement.height_measurement import (HeightConfig, dilate_mask, erode_wire_mask_in_place,
    erode_measurement_mask_crop, fit_chip_plane, fit_local_platform_plane, fit_platform_plane,
    height_points_to_board, map_platform_rois, measure_mask_height, split_thin_mask)
from algorithm.vision3d.alignment.registration import align_crop_mask_to_depth
from transport.depth_input import depth_coordinate_space
from algorithm.execution.instance_tasks import InstanceTasks
from algorithm.runtime.defect_plan import (
    CHECKS_3D, CHIP_SOLDER_HEIGHT, PLATFORM_SOLDER_HEIGHT, WIRE_HEIGHT, evaluate_in_order,
)
from algorithm.runtime.timing import timed_call
from algorithm.vision2d import defect_geometry as _geometry
from algorithm.vision2d.instance_geometry import component_geometry, mask_geometry as _mask_geometry


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
        options = {"height_order": entry["height_order"]} if "height_order" in entry else {}
        measurement = timed_call("height_measurement_ms", self._measure_piece,
            local_depth,
            aligned_masks,
            registration,
            crop_box=entry.get("crop_box", self.crop_roi),
            circle=entry.get("circle"),
            **options,
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
            "depth_coordinate_space": (piece.get("depth_coordinate_space")
                                       or self._joint_ops.depth_coordinate_space(pitch_correction or {})),
            "registration": registration,
            "is_ok": bool(
                measurement["is_ok"]
                and all(item.get("is_ok") is True for item in entry.get("wire_roi_checks", []))
            ),
            "reference_planes": measurement["reference_planes"],
            "wire_measurements": measurement["wire_measurements"],
            "solder_measurements": measurement["solder_measurements"],
            "masks": public_masks,
            **{key: measurement[key] for key in ("defect_checks_3d", "first_ng_check") if key in measurement},
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
            "mask_2d": component_geometry(instance, crop_box[:2], epsilon=2.0, geometry=self._joint_ops._mask_geometry),
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
        *, height_order: tuple[str, ...] | None = None,
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
        chips = [item for item in instances if item["class_name"].lower() == "chip"]
        chip_planes = {}

        @cache
        def prepare_thin():
            timed_call("height_prepare_thin_ms", self._prepare_thin_parts, instances)

        def measure_wires():
            prepare_thin()
            return timed_call("height_wire_regions_ms", self._measure_wire_regions,
                depth, instances, registration, platform_plane, crop_box=crop_box, prepare_thin=False)

        def measure_solders(jobs):
            return timed_call("height_solder_wall_ms", self._instance_tasks.map,
                lambda job: self._measure_solder_task(job, depth, registration, chips, chip_planes,
                    platform_plane, semantic_exclusion, crop_box),
                jobs, timing_stage="height_solder_work_ms")

        @cache
        def solder_groups():
            prepare_thin()
            timed_call("height_prepare_chip_regions_ms", self._prepare_chip_regions, chips)
            groups = {CHIP_SOLDER_HEIGHT: [], PLATFORM_SOLDER_HEIGHT: []}
            for job in self._solder_jobs(instances):
                _, inside = self._matching_chip({"_mask_2d_array": job[2]}, chips)
                groups[CHIP_SOLDER_HEIGHT if inside else PLATFORM_SOLDER_HEIGHT].append(job)
            return groups

        summary = {}
        if height_order is None:
            timed_call("height_prepare_chip_regions_ms", self._prepare_chip_regions, chips)
            chip_planes.update(self._fit_chip_planes(depth, chips, instances))
            wire_measurements = measure_wires()
            solder_measurements = measure_solders(self._solder_jobs(instances))
            measurements = wire_measurements + solder_measurements
            is_ok = bool(measurements) and all(item["height_3d"].get("is_ok") for item in measurements)
        else:
            wire_measurements, solder_measurements = [], []

            def run_check(name):
                if name == WIRE_HEIGHT:
                    measured = measure_wires()
                else:
                    jobs = solder_groups()[name]
                    if name == CHIP_SOLDER_HEIGHT and jobs:
                        chip_planes.update(self._fit_chip_planes(depth, chips, instances))
                    measured = measure_solders(jobs)
                if name == WIRE_HEIGHT:
                    wire_measurements.extend(measured)
                else:
                    solder_measurements.extend(measured)
                passed = bool(measured) and all(item["height_3d"].get("is_ok") is True for item in measured)
                return {"enabled": True, "status": "ok" if passed else "ng", "is_ok": passed}

            checks = {name: (lambda name=name: run_check(name)) for name in CHECKS_3D}
            summary["defect_checks_3d"], first_ng = evaluate_in_order(
                height_order, checks, short_circuit=False)
            is_ok = first_ng is None
            if first_ng is not None:
                summary["first_ng_check"] = first_ng

        wire_by_instance = {int(item["instance_id"]): item for item in wire_measurements}
        bond_by_instance = {item["source_instance_id"]: item for item in solder_measurements
                            if item["endpoint"] == "bond"}
        # All child tasks have finished. Only this parent updates instances.
        for instance in instances:
            class_name = instance["class_name"].lower()
            if class_name in {"thin", "wire"}:
                wire_result = wire_by_instance.get(int(instance["instance_id"]))
                instance["height_3d"] = wire_result["height_3d"] if wire_result is not None else None
            elif class_name == "bond":
                solder_result = bond_by_instance.get(instance["instance_id"])
                instance["height_3d"] = solder_result["height_3d"] if solder_result is not None else None
                if solder_result is not None:
                    instance["bond_location"] = solder_result["location"]
                    instance["reference_instance_id"] = solder_result["reference_instance_id"]
                    instance["chip_inside"] = solder_result["chip_inside"]
            elif class_name == "chip":
                instance["height_3d"] = None
                instance["plane_3d"] = chip_planes.get(instance["instance_id"])
            else:
                instance["height_3d"] = None

        return {
            "is_ok": is_ok,
            **summary,
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

    def _fit_chip_planes(self, depth, chips, instances):
        planes = {}
        for chip in chips:
            chip_id = chip["instance_id"]
            # Exclude the other masks, including any overlap with this chip.
            exclusion = self._semantic_exclusion_mask(instances, depth.shape, skip_instance_id=chip_id)
            try:
                planes[chip_id] = timed_call("height_fit_chip_plane_ms", self._joint_ops.fit_chip_plane,
                    depth, chip["_mask_3d_array"], self.height_config, exclude_mask=exclusion)
            except ValueError:
                planes[chip_id] = None
        return planes

    @staticmethod
    def _solder_jobs(instances):
        for instance in instances:
            class_name = instance["class_name"].lower()
            if class_name == "thin":
                for endpoint, mask in zip(("start", "end"), instance["_thin_solder_masks_2d"]):
                    yield instance, endpoint, mask
            elif class_name == "bond":
                yield instance, "bond", instance["_mask_2d_array"]

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
                # Reuse bounds computed from this aligned mask for the response.
                bounds = (item.get("mask_3d_local") or {}).get("bounding_box")
                if bounds is None:
                    np.maximum(union, mask, out=union)
                elif bounds:
                    left, top, right, bottom = bounds
                    region = np.s_[top:bottom, left:right]
                    np.maximum(union[region], mask[region], out=union[region])
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
        *, prepare_thin: bool = True,
    ) -> list[Dict[str, Any]]:
        crop_box = tuple(crop_box or self.crop_roi)
        wires = [item for item in instances if item["class_name"].lower() in {"wire", "thin"}]
        if not wires:
            return []
        if prepare_thin:
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
        if not chips:
            return None, False
        source = np.asarray(bond["_mask_2d_array"])
        bond_mask = source if source.dtype == np.uint8 else (source > 0).astype(np.uint8)
        x, y, width, height = cv2.boundingRect(bond_mask)
        if width == 0:
            return None, False
        moments = cv2.moments(bond_mask[y:y + height, x:x + width], binaryImage=True)
        area = moments["m00"]
        center_x = round((moments["m10"] + x * area) / area)
        center_y = round((moments["m01"] + y * area) / area)
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
    source = np.asarray(mask)
    binary = source if source.dtype == np.uint8 else (source > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    if contours:
        points = np.concatenate(contours, axis=0)
        cv2.fillConvexPoly(filled, cv2.convexHull(points), 255)
    return filled > 0
