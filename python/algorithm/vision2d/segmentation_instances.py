"""Build inspection instances from connected regions of each YOLO class."""
from collections import defaultdict
from math import ceil, floor, isfinite

import cv2
import numpy as np
import torch


def extract_instances(result, image_shape, min_area):
    masks_object = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks_object is None or boxes is None or not len(masks_object.data):
        return []

    masks = masks_object.data
    native = tuple(masks.shape[-2:]) == tuple(image_shape)
    if not native:
        from ultralytics.utils.ops import scale_masks

        masks = scale_masks(masks.unsqueeze(1), image_shape).squeeze(1)

    metadata = torch.stack((boxes.cls, boxes.conf), dim=1)
    xyxy = getattr(boxes, "xyxy", None)
    # Native Ultralytics prediction already zeros pixels outside each source box.
    # Other inputs, including resized masks, use their actual foreground extent.
    use_boxes = native and getattr(result, "orig_shape", None) == image_shape and xyxy is not None
    if use_boxes:
        metadata = torch.cat((metadata, xyxy), dim=1)
    metadata = metadata.detach().cpu().tolist()
    crops = _transfer_crops(masks, metadata if use_boxes else None, image_shape)
    classes = defaultdict(list)
    for index, (crop, left, top) in enumerate(crops):
        x, y, width, height = cv2.boundingRect(crop)
        if width and height:
            class_id, score = metadata[index][:2] if index < len(metadata) else (-1, 1.)
            bounds = (left+x, top+y, left+x+width, top+y+height)
            classes[int(class_id)].append((index, score, bounds, crop[y:y+height, x:x+width]))

    names = getattr(result, "names", {}) or {}
    components = []
    for class_id, predictions in classes.items():
        class_name = names.get(class_id, names.get(str(class_id), class_id)) if isinstance(names, dict) else class_id
        for group in _neighbor_groups(predictions):
            components.extend(_label_group(group, image_shape, min_area, class_id, str(class_name)))

    instances = []
    for instance_id, (_, instance) in enumerate(sorted(components, key=lambda item: item[0]), 1):
        instance["instance_id"] = instance_id
        instances.append(instance)
    return instances


def _transfer_crops(masks, coordinates, image_shape):
    if coordinates is None:
        binary = masks if masks.dtype == torch.uint8 else (masks > .5).to(torch.uint8)
        return [(mask, 0, 0) for mask in binary.cpu().numpy()]

    height, width = image_shape
    buffers, layouts = [], []
    for index, mask in enumerate(masks):
        box = coordinates[index][2:] if index < len(coordinates) else ()
        left, top, right, bottom = 0, 0, width, height
        if box and all(isfinite(value) for value in box):
            x1, y1, x2, y2 = box
            # Round outward with a one-pixel margin to retain native edge pixels.
            left, top = max(0, floor(x1)-1), max(0, floor(y1)-1)
            right, bottom = min(width, ceil(x2)+1), min(height, ceil(y2)+1)
            if right <= left or bottom <= top:
                left, top, right, bottom = 0, 0, width, height
        crop = mask[top:bottom, left:right]
        binary = crop if crop.dtype == torch.uint8 else (crop > .5).to(torch.uint8)
        buffers.append(binary.reshape(-1))
        layouts.append((left, top, bottom-top, right-left))
    packed = torch.cat(buffers).cpu().numpy()
    offset, crops = 0, []
    for left, top, rows, columns in layouts:
        end = offset + rows*columns
        crops.append((packed[offset:end].reshape(rows, columns), left, top))
        offset = end
    return crops


def _touches(first, second):
    # Half-open boxes with adjacent edges or corners can contain 8-connected pixels.
    return (first[0] <= second[2] and second[0] <= first[2] and
            first[1] <= second[3] and second[1] <= first[3])


def _neighbor_groups(predictions):
    groups = []
    for prediction in predictions:
        joined, separate = [prediction], []
        for group in groups:
            if any(_touches(prediction[2], other[2]) for other in group):
                joined.extend(group)
            else:
                separate.append(group)
        groups = [*separate, joined]
    return groups


def _label_group(group, image_shape, min_area, class_id, class_name):
    if len(group) == 1:
        source, score, (left, top, _, _), foreground = group[0]
        ranks = None
    else:
        group.sort(key=lambda item: (item[1], -item[0]))
        left = min(item[2][0] for item in group)
        top = min(item[2][1] for item in group)
        right = max(item[2][2] for item in group)
        bottom = max(item[2][3] for item in group)
        dtype = np.uint8 if len(group) <= 255 else np.int32
        ranks = np.zeros((bottom-top, right-left), dtype=dtype)
        for rank, (_, _, (x1, y1, x2, y2), crop) in enumerate(group, 1):
            np.copyto(ranks[y1-top:y2-top, x1-left:x2-left], rank, where=crop != 0)
        foreground = ranks
        if dtype != np.uint8:
            foreground = np.empty(ranks.shape, dtype=np.uint8)
            np.greater(ranks, 0, out=foreground)

    if foreground.size < min_area:
        return
    _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    for label, (x, y, width, height, area) in enumerate(stats[1:], 1):
        if area < min_area:
            continue
        region = np.s_[y:y+height, x:x+width]
        x0, y0 = int(left+x), int(top+y)
        mask = np.zeros(image_shape, dtype=np.uint8)
        component = mask[y0:y0+height, x0:x0+width]
        # OpenCV 5's scalar compare can reject a 1x1 label region.
        np.equal(labels[region], label, out=component)
        if ranks is not None:
            _, best_rank, _, _ = cv2.minMaxLoc(ranks[region], mask=component)
            source, score = group[int(best_rank) - 1][:2]
        component *= 255
        yield source, dict(class_id=class_id, class_name=class_name, confidence=score, mask=mask,
            _mask_bbox=(x0, y0, int(width), int(height)), _mask_area_px=int(area), _mask_binary_255=True,
            rectangle=None, rectangle_ratio=1., is_ok=True)
