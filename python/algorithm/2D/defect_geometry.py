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


def constrained_rectangle(mask: np.ndarray, roi_boxes: list[tuple[int, int, int, int]],
                           min_ratio: float = 0.75, mask_offset: tuple[int, int] = (0, 0),
                           coarse_steps: int = 21, fine_steps: int = 11):
    """Fit the longest rotated rectangle contained by the selected ROI."""
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours or not roi_boxes:
        return None
    offset = np.asarray(mask_offset)
    points = np.concatenate(contours) + offset
    (cx, cy), (length, width), angle = cv2.minAreaRect(points)
    if length < width:
        length, width, angle = width, length, angle + 90
    direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    normal = np.array([-direction[1], direction[0]])
    mask_x, mask_y = mask_offset
    mask_height, mask_width = mask_u8.shape
    mask_area = 0
    selected = None
    for roi in roi_boxes:
        x1, y1, x2, y2 = roi
        ix1, iy1 = max(x1, mask_x), max(y1, mask_y)
        ix2, iy2 = min(x2 + 1, mask_x + mask_width), min(y2 + 1, mask_y + mask_height)
        area = (cv2.countNonZero(mask_u8[iy1 - mask_y:iy2 - mask_y,
                                             ix1 - mask_x:ix2 - mask_x])
                if ix1 < ix2 and iy1 < iy2 else 0)
        if area > mask_area:
            mask_area, selected = area, roi
    if selected is None:
        return None
    rx1, ry1, rx2, ry2 = selected
    origin = np.array([cx, cy])
    ranges = []
    for coordinate, component, lower, upper, margin in zip(
        origin, direction, (rx1, ry1), (rx2, ry2),
        (width * abs(direction[1]) / 2, width * abs(direction[0]) / 2),
    ):
        if abs(component) < 1e-6:
            if not lower + margin <= coordinate <= upper - margin:
                return None
        else:
            ranges.append(((lower + margin - coordinate) / component,
                           (upper - margin - coordinate) / component))
    low = max(min(item) for item in ranges) if ranges else -length
    high = min(max(item) for item in ranges) if ranges else length
    if low > high:
        return None

    def evaluate(position):
        center = origin + direction * position
        max_x = (2 * min(center[0] - rx1, rx2 - center[0]) - width * abs(direction[1])) / max(abs(direction[0]), 1e-6)
        max_y = (2 * min(center[1] - ry1, ry2 - center[1]) - width * abs(direction[0])) / max(abs(direction[1]), 1e-6)
        current_length = min(length, max_x, max_y)
        if current_length <= 0:
            return None
        corners = np.array([
            center + direction * current_length / 2 + normal * width / 2,
            center + direction * current_length / 2 - normal * width / 2,
            center - direction * current_length / 2 - normal * width / 2,
            center - direction * current_length / 2 + normal * width / 2,
        ])
        return current_length, corners

    best = None
    best_position = None
    for position in np.linspace(low, high, coarse_steps):
        candidate = evaluate(position)
        if candidate is not None and (best is None or candidate[0] > best[0]):
            best, best_position = candidate, position
    if best is None:
        return None

    coarse_step = (high - low) / max(coarse_steps - 1, 1)
    fine_low = max(low, best_position - coarse_step)
    fine_high = min(high, best_position + coarse_step)
    for position in np.linspace(fine_low, fine_high, fine_steps):
        candidate = evaluate(position)
        if candidate is not None and candidate[0] > best[0]:
            best, best_position = candidate, position

    _, corners = best
    roi_height, roi_width = ry2 - ry1 + 1, rx2 - rx1 + 1
    roi_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
    ix1, iy1 = max(rx1, mask_x), max(ry1, mask_y)
    ix2, iy2 = min(rx2 + 1, mask_x + mask_width), min(ry2 + 1, mask_y + mask_height)
    if ix1 < ix2 and iy1 < iy2:
        roi_mask[iy1 - ry1:iy2 - ry1, ix1 - rx1:ix2 - rx1] = mask_u8[
            iy1 - mask_y:iy2 - mask_y, ix1 - mask_x:ix2 - mask_x
        ]
    rectangle = np.zeros_like(roi_mask)
    local_corners = np.round(corners - np.array([rx1, ry1])).astype(np.int32)
    cv2.fillPoly(rectangle, [local_corners], 255)
    overlap = cv2.countNonZero(cv2.bitwise_and(rectangle, roi_mask))
    ratio = overlap / max(cv2.countNonZero(rectangle), 1)
    return corners.round().astype(np.int32), float(ratio), bool(ratio >= min_ratio)
