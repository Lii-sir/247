"""Model-free public instance geometry shared by direct calls and CPU workers."""
import cv2
import numpy as np

PUBLIC_INSTANCE_FIELDS = ("instance_id", "class_id", "class_name", "confidence", "rectangle", "rectangle_ratio",
                          "bending_2d", "chip_alignment", "chip_anomaly_2d", "is_ok")


def mask_geometry(mask, offset, epsilon):
    source = np.asarray(mask)
    binary = source if source.dtype == np.uint8 else (source > 0).astype(np.uint8)
    area = int(cv2.countNonZero(binary))
    if area == 0:
        return {"area_px": 0, "bounding_box": [], "contours": []}
    x, y, width, height = cv2.boundingRect(binary)
    ox, oy = int(offset[0]), int(offset[1])
    contours, _ = cv2.findContours(binary[y:y+height, x:x+width], cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE, offset=(x, y))
    serialized = []
    for contour in contours:
        approximate = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
        if len(approximate) >= 3:
            serialized.append([[int(point[0])+ox, int(point[1])+oy] for point in approximate])
    return dict(area_px=area, bounding_box=[x+ox, y+oy, x+width+ox, y+height+oy], contours=serialized)


def public_instance(instance, crop_box, *, geometry=mask_geometry):
    mask = instance.get("mask")
    offset = tuple(crop_box)[:2]
    return {
        "instance_id": int(instance.get("instance_id", 0)),
        "class_id": int(instance.get("class_id", -1)),
        "class_name": str(instance.get("class_name", "")),
        "confidence": float(instance.get("confidence", 0.0)),
        "mask_2d": geometry(mask, offset=offset, epsilon=1.0) if isinstance(mask, np.ndarray)
                   else geometry(np.zeros((1, 1), dtype=np.uint8), (0, 0), 1.0),
        "rectangle": instance.get("rectangle"),
        "rectangle_ratio": float(instance.get("rectangle_ratio", 0.0)),
        "bending_2d": instance.get("bending_2d"),
        "chip_rotation_2d": instance.get("chip_alignment"),
        "chip_anomaly_2d": instance.get("chip_anomaly_2d"),
        "is_ok": bool(instance.get("is_ok", True)),
    }
