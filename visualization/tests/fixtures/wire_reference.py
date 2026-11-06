"""Frozen full-frame wire path from 2491f41 for differential regressions."""
import numpy as np
from algorithm.detect_board import (DetectBoard, Any, Dict, Sequence, _geometry,
    _mask_geometry, split_thin_mask, align_crop_mask_to_depth,
    erode_wire_mask_in_place, measure_mask_height, timed_call, fit_platform_plane,
    map_platform_rois, fit_chip_plane, dilate_mask, _filled_external_region)


class LegacyWireReference(DetectBoard):
    def _mask_in_wire_rois(
        self, mask: np.ndarray, wire_rois: Sequence[Sequence[int]] | None = None
    ) -> np.ndarray:
        output = np.zeros_like(mask, dtype=np.uint8)
        height, width = output.shape
        for raw_roi in self.wire_rois if wire_rois is None else wire_rois:
            x1, y1, x2, y2 = (int(value) for value in raw_roi)
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 > x1 and y2 > y1:
                output[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return output

    def _measure_wire_regions(
        self,
        depth: np.ndarray,
        instances: list[Dict[str, Any]],
        registration: Dict[str, Any],
        platform_plane: Dict[str, Any] | None,
        wire_rois: Sequence[Sequence[int]] | None = None,
        crop_box: Sequence[int] | None = None,
    ) -> list[Dict[str, Any]]:
        crop_box = tuple(crop_box or self.crop_roi)
        wires = [item for item in instances if item["class_name"].lower() in {"wire", "thin"}]
        if not wires:
            return []
        wire_rois = self.wire_rois if wire_rois is None else wire_rois
        output = []
        offset = np.asarray(crop_box[:2], dtype=np.int32)
        for wire in wires:
            class_name = wire["class_name"].lower()
            source_mask = np.asarray(wire["_mask_2d_array"], dtype=np.uint8)
            if class_name == "thin":
                source_mask, _ = split_thin_mask(
                    source_mask, self.height_config.thin_solder_end_ratio
                )
            if not np.any(self._mask_in_wire_rois(source_mask, wire_rois)):
                # Wires outside the configured 2D arc-height ROI are not measured.
                continue
            cleaned = _geometry.clean_mask(
                source_mask,
                min_area=self.min_component_area,
                kernel_size=5,
            )
            cleaned = self._mask_in_wire_rois(cleaned, wire_rois)
            if not np.any(cleaned):
                height = measure_mask_height(
                    depth, np.zeros_like(depth, dtype=np.uint8), platform_plane,
                    self.height_config, reference="wire_platform",
                    max_height_mm=self.height_config.max_wire_platform_height_mm,
                )
                height["is_ok"] = False
                output.append({
                    "region_id": len(output) + 1,
                    "instance_id": int(wire.get("instance_id", 0)),
                    "class_name": str(wire.get("class_name", "")),
                    "mask_2d": _mask_geometry(cleaned, offset=crop_box[:2], epsilon=1.0),
                    "mask_3d_local": _mask_geometry(np.zeros_like(depth, dtype=np.uint8), offset=(0, 0), epsilon=1.0),
                    "bending_2d": wire.get("bending_2d", {"is_valid": False, "is_ok": False}),
                    "height_3d": height,
                })
                continue
            # Bending is calculated once on the original YOLO instance during
            # the 2D stage.  Keep that result here; only the central thin mask
            # is changed for the independent 3D wire-height measurement.
            bending = wire.get("bending_2d")
            if not isinstance(bending, dict):
                # Use the original full instance, before thin-end splitting and
                # height-ROI clipping, just like the immediate 2D measurement.
                bending_mask = _geometry.clean_mask(
                    np.asarray(wire["_mask_2d_array"], dtype=np.uint8),
                    min_area=self.min_component_area, kernel_size=5,
                )
                bending = _geometry.measure_wire_bending(
                    bending_mask, wire_rois, self.rectangle_ratio_threshold,
                    output_offset=offset,
                )
            aligned = align_crop_mask_to_depth(
                cleaned.astype(np.uint8) * 255,
                crop_box,
                depth.shape,
                registration,
            )
            erode_px = (self.height_config.thin_measurement_erode_px
                        if class_name == "thin" else self.height_config.wire_measurement_erode_px)
            if erode_px:
                # This separately warped mask is owned by this measurement;
                # complete instance/semantic masks remain intact for planes.
                erode_wire_mask_in_place(aligned, erode_px)
            height = measure_mask_height(
                depth,
                aligned,
                platform_plane,
                self.height_config,
                reference="wire_platform",
                max_height_mm=self.height_config.max_wire_platform_height_mm,
            )
            output.append(
                {
                    "region_id": len(output) + 1,
                    "instance_id": int(wire.get("instance_id", 0)),
                    "class_name": str(wire.get("class_name", "")),
                    "mask_2d": _mask_geometry(
                        cleaned, offset=crop_box[:2], epsilon=2.0
                    ),
                    "mask_3d_local": _mask_geometry(
                        aligned, offset=(0, 0), epsilon=1.0
                    ),
                    "bending_2d": bending,
                    "height_3d": height,
                }
            )
        return output


    def _measure_piece(
        self,
        depth: np.ndarray,
        instances: list[Dict[str, Any]],
        registration: Dict[str, Any],
        wire_rois: Sequence[Sequence[int]] | None = None,
        crop_box: Sequence[int] | None = None,
        circle: Sequence[float] | None = None,
    ) -> Dict[str, Any]:
        """Measure all 3D heights for one aligned piece.

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
            "sampling_mask_3d_local": _mask_geometry(
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
                roi_mask, boxes, polygons = map_platform_rois(
                    circle, self.height_config.platform_roi_offsets,
                    registration, depth.shape,
                )
                sampling_mask = (roi_mask > 0) & (semantic_exclusion == 0)
                sampling_mask &= ~np.isin(depth, self.height_config.invalid_values)
                platform_diagnostics.update(
                    boxes_2d=boxes, polygons_3d_local=polygons,
                    roi_area_px=int(np.count_nonzero(roi_mask)),
                    candidate_points=int(np.count_nonzero(sampling_mask)),
                    sampling_mask_3d_local=_mask_geometry(
                        sampling_mask, offset=(0, 0), epsilon=0.0
                    ),
                )
                platform_plane = timed_call("height_fit_platform_plane_ms", fit_platform_plane,
                    depth, self.height_config, roi_mask, exclude_mask=semantic_exclusion
                )
                platform_diagnostics.update(status="ok", error=None)
        except ValueError as exc:
            platform_diagnostics.update(status="invalid", error=str(exc))
        chips = [
            item for item in instances if item["class_name"].lower() == "chip"
        ]
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
                chip_planes[chip["instance_id"]] = timed_call("height_fit_chip_plane_ms", fit_chip_plane,
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
            wire_rois,
            crop_box=crop_box,
        )
        wire_by_instance = {
            int(item["instance_id"]): item for item in wire_measurements
        }

        solder_measurements: list[Dict[str, Any]] = []
        for instance in instances:
            class_name = instance["class_name"].lower()
            if class_name == "thin":
                middle_2d, solder_2d = split_thin_mask(
                    instance["_mask_2d_array"],
                    self.height_config.thin_solder_end_ratio,
                )
                instance["_thin_wire_mask_2d"] = middle_2d
                instance["_thin_solder_masks_2d"] = solder_2d
                for endpoint, solder_mask_2d in zip(("start", "end"), solder_2d):
                    solder_mask_3d = align_crop_mask_to_depth(
                        solder_mask_2d,
                        crop_box,
                        depth.shape,
                        registration,
                    )
                    # The registration scale is not necessarily one.  Expand
                    # after mapping so this parameter is expressed in actual
                    # depth-image pixels, which is the search range needed to
                    # tolerate a very thin 2D endpoint missing the 3D ridge.
                    solder_mask_3d = dilate_mask(
                        solder_mask_3d,
                        self.height_config.thin_solder_search_dilation_px,
                    )
                    solder_measurements.append(
                        timed_call("height_solder_ms", self._measure_solder,
                            depth,
                            solder_mask_2d,
                            solder_mask_3d,
                            instance,
                            endpoint,
                            chips,
                            chip_planes,
                            platform_plane,
                            semantic_exclusion,
                            crop_box,
                        )
                    )
                wire_result = wire_by_instance.get(int(instance["instance_id"]))
                instance["height_3d"] = (
                    wire_result["height_3d"] if wire_result is not None else None
                )
            elif class_name == "bond":
                solder_measurements.append(
                    timed_call("height_solder_ms", self._measure_solder,
                        depth,
                        instance["_mask_2d_array"],
                        instance["_mask_3d_array"],
                        instance,
                        "bond",
                        chips,
                        chip_planes,
                        platform_plane,
                        semantic_exclusion,
                        crop_box,
                    )
                )
                solder_result = solder_measurements[-1]
                instance["height_3d"] = solder_result["height_3d"]
                instance["bond_location"] = solder_result["location"]
                instance["reference_instance_id"] = solder_result["reference_instance_id"]
                instance["chip_inside"] = solder_result["chip_inside"]
            elif class_name == "wire":
                wire_result = wire_by_instance.get(int(instance["instance_id"]))
                instance["height_3d"] = (
                    wire_result["height_3d"] if wire_result is not None else None
                )
            elif class_name == "chip":
                instance["height_3d"] = None
                instance["plane_3d"] = chip_planes.get(instance["instance_id"])
            else:
                instance["height_3d"] = None

        wire_ok = [
            bool(item["bending_2d"].get("is_ok"))
            and bool(item["height_3d"].get("is_ok"))
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


    @staticmethod
    def _matching_chip(
        bond: Dict[str, Any],
        chips: list[Dict[str, Any]],
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
            # Aligned chip dicts belong to one measurement task. Multiple
            # bond/thin endpoints reuse this unchanged external region.
            chip_region = chip.get("_filled_chip_region_2d")
            if chip_region is None:
                chip_region = _filled_external_region(chip["_mask_2d_array"])
                chip["_filled_chip_region_2d"] = chip_region
            if chip_region[center_y, center_x]:
                area = chip.get("_filled_chip_area_2d")
                if area is None:
                    area = int(np.count_nonzero(chip_region))
                    chip["_filled_chip_area_2d"] = area
                if best_area is None or area < best_area:
                    best_chip, best_area = chip, area
        return best_chip, best_chip is not None
