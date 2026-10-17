"""Small, deterministic post-processing helpers for 2D segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional


def prepare_masks_gpu(masks: torch.Tensor, allowed: np.ndarray) -> torch.Tensor:
    """Resize model masks and apply the allowed ROI without leaving the tensor device."""
    if not masks.is_cuda:
        raise RuntimeError("2D mask post-processing requires a CUDA tensor")
    allowed_gpu = torch.as_tensor(allowed, device=masks.device, dtype=torch.bool)
    if masks.shape[-2:] != allowed_gpu.shape:
        masks = torch_functional.interpolate(
            masks.unsqueeze(1), size=allowed_gpu.shape, mode="nearest"
        ).squeeze(1)
    return torch.logical_and(masks > 0.5, allowed_gpu)


def merge_masks_gpu(masks: torch.Tensor, iou_threshold: float = 0.2) -> tuple[torch.Tensor, list[int]]:
    """Transitively merge overlapping masks while pixel operations stay on the tensor device."""
    merged: list[torch.Tensor] = []
    source_indices: list[int] = []
    for source_index, mask in enumerate(masks):
        current = mask
        current_source = source_index
        while merged:
            existing = torch.stack(merged)
            intersection = torch.logical_and(existing, current).flatten(1).sum(1)
            union = existing.flatten(1).sum(1) + current.sum() - intersection
            matches = (union > 0) & (intersection / union.clamp_min(1) > iou_threshold)
            match_indices = torch.nonzero(matches).flatten()
            if not match_indices.numel():
                break
            match_index = int(match_indices[0])
            current = torch.logical_or(current, merged.pop(match_index))
            current_source = min(current_source, source_indices.pop(match_index))
        merged.append(current)
        source_indices.append(current_source)
    if not merged:
        return masks, []
    return torch.stack(merged), source_indices


def transfer_mask_crops(masks: torch.Tensor, source_indices: list[int],
                        padding: int = 5) -> list[tuple[int, np.ndarray, tuple[int, int]]]:
    """Copy only each mask's nonzero bounding region to CPU memory."""
    crops = []
    height, width = masks.shape[-2:]
    for mask, source_index in zip(masks, source_indices):
        rows = torch.nonzero(torch.any(mask, dim=1)).flatten()
        columns = torch.nonzero(torch.any(mask, dim=0)).flatten()
        if not rows.numel() or not columns.numel():
            continue
        y1 = max(int(rows[0]) - padding, 0)
        y2 = min(int(rows[-1]) + padding + 1, height)
        x1 = max(int(columns[0]) - padding, 0)
        x2 = min(int(columns[-1]) + padding + 1, width)
        crop = (mask[y1:y2, x1:x2].to(torch.uint8) * 255).cpu().numpy()
        crops.append((source_index, crop, (x1, y1)))
    return crops


def clean_mask(mask: np.ndarray, min_area: int = 500, kernel_size: int = 5) -> np.ndarray:
    """Smooth a mask and keep its dominant connected component."""
    source = (np.asarray(mask) > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    source = cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel)
    source = cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source, 8)
    components = sorted(
        ((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)
         if int(stats[i, cv2.CC_STAT_AREA]) >= min_area), reverse=True
    )
    if not components:
        return np.zeros_like(mask, dtype=bool)
    # A second large component usually means the model merged unrelated areas.
    if len(components) > 1 and components[1][0] / components[0][0] > 0.2:
        return np.zeros_like(mask, dtype=bool)
    return labels == components[0][1]


def clean_mask_crop(mask: np.ndarray, min_area: int = 500,
                    kernel_size: int = 5) -> tuple[np.ndarray, tuple[int, int]]:
    """Clean the entire instance within its bounds, retaining its crop offset."""
    source = np.asarray(mask)
    binary = source if source.dtype == np.uint8 else (source > 0).astype(np.uint8)
    x, y, width, height = cv2.boundingRect(binary)
    if width == 0 or height == 0:
        return np.zeros((1, 1), dtype=bool), (0, 0)
    # Closing followed by opening has four primitive operations. Keep enough
    # zero context for their full dependency radius, or the original image edge.
    padding = 4 * (kernel_size // 2)
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1 = min(binary.shape[1], x + width + padding)
    y1 = min(binary.shape[0], y + height + padding)
    cleaned = clean_mask(binary[y0:y1, x0:x1], min_area, kernel_size)
    return cleaned, (x0, y0)


def measure_wire_bending(mask: np.ndarray, roi_boxes, min_ratio: float = 0.75,
                         mask_offset=(0, 0), output_offset=(0, 0)) -> dict:
    """Join the two boundary ports of one cleaned instance into a quadrilateral.

    ROI boxes are half-open. Each port is a cyclic contiguous foreground run
    on the ROI's pixel perimeter; its two endpoints are the edge intersections.
    The denominator counts this instance inside the selected ROI, not polygon
    area. No rotated-rectangle fallback is used for ambiguous intersections.
    """
    result = dict(is_valid=False, is_ok=False, mask_ratio=0.0, rectangle=[],
                  status="empty_mask", roi_id=None, roi_mask_pixel_count=0,
                  covered_mask_pixel_count=0)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not np.any(binary):
        return result
    mx, my = map(int, mask_offset)
    height, width = binary.shape
    overlaps = []
    for raw_box in roi_boxes:
        x1, y1, x2, y2 = map(int, raw_box)
        left, top = max(x1, mx), max(y1, my)
        right, bottom = min(x2, mx + width), min(y2, my + height)
        overlaps.append(int(np.count_nonzero(binary[top-my:bottom-my, left-mx:right-mx]))
                        if left < right and top < bottom else 0)
    if not overlaps or max(overlaps) == 0:
        result["status"] = "outside_rois"
        return result
    selected = int(np.argmax(overlaps))
    x1, y1, x2, y2 = map(int, roi_boxes[selected])
    result.update(roi_id=selected + 1, roi_mask_pixel_count=overlaps[selected])
    if x2 - x1 < 2 or y2 - y1 < 2:
        result["status"] = "degenerate_shape"
        return result

    # Rasterize only this instance in its ROI. Do not clip the ROI itself to
    # the local mask bounds: doing so would invent boundary intersections.
    roi_mask = np.zeros((y2-y1, x2-x1), np.uint8)
    left, top = max(x1, mx), max(y1, my)
    right, bottom = min(x2, mx + width), min(y2, my + height)
    roi_mask[top-y1:bottom-y1, left-x1:right-x1] = binary[
        top-my:bottom-my, left-mx:right-mx]
    h, w = roi_mask.shape
    # Clockwise perimeter, with each corner appearing exactly once.
    perimeter = np.array(
        [(x, 0) for x in range(w)] +
        [(w-1, y) for y in range(1, h)] +
        [(x, h-1) for x in range(w-2, -1, -1)] +
        [(0, y) for y in range(h-2, 0, -1)], dtype=np.int32)
    contact = roi_mask[perimeter[:, 1], perimeter[:, 0]] > 0
    starts = np.flatnonzero(contact & ~np.roll(contact, 1))
    if len(starts) != 2:
        result["status"] = ("ambiguous_intersections" if len(starts) > 2 or np.all(contact)
                            else "insufficient_intersections")
        return result
    endpoints = []
    for start in starts:
        end = int(start)
        while contact[(end + 1) % len(contact)]:
            end = (end + 1) % len(contact)
        endpoints.extend((perimeter[start], perimeter[end]))
    points = np.asarray(endpoints, np.int32)
    hull = cv2.convexHull(points).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) <= 0:
        result["status"] = "degenerate_shape"
        return result
    polygon = np.zeros_like(roi_mask)
    cv2.fillPoly(polygon, [hull], 1)
    covered = int(np.count_nonzero(roi_mask & polygon))
    ratio = covered / overlaps[selected]
    is_ok = bool(ratio >= min_ratio)
    ox, oy = map(int, output_offset)
    result.update(is_valid=True, is_ok=is_ok, mask_ratio=float(ratio),
                  rectangle=(hull + np.array([x1+ox, y1+oy])).tolist(),
                  covered_mask_pixel_count=covered,
                  status="ok" if is_ok else "bending_out_of_range")
    return result


def constrained_rectangle(mask: np.ndarray, roi_boxes: list[tuple[int, int, int, int]],
                           min_ratio: float = 0.75, mask_offset: tuple[int, int] = (0, 0)):
    """Compatibility tuple API; the four points now describe a quadrilateral."""
    result = measure_wire_bending(mask, roi_boxes, min_ratio, mask_offset)
    if not result["is_valid"]:
        return None
    return np.asarray(result["rectangle"], np.int32), result["mask_ratio"], result["is_ok"]
