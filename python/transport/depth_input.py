"""Packed little-endian uint16 depth input; no image codec or gray conversion."""
import numpy as np


def validate_depth_input(descriptor):
    name = descriptor.get("shm_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("3D input requires shm_name")
    data_format = descriptor.get("data_format", "uint16")
    if not isinstance(data_format, str) or data_format not in {"uint16", "raw_uint16"}:
        raise ValueError("3D input requires raw little-endian uint16; encoded PNG is not accepted")
    values = [descriptor.get(key) for key in ("data_size", "image_width", "image_height")]
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("3D input requires positive integer data_size, image_width and image_height")
    size, width, height = values
    if size != width * height * 2:
        raise ValueError("3D data_size must equal image_width * image_height * 2 (packed uint16)")
    if descriptor.get("image_stride", width * 2) not in (0, width * 2):
        raise ValueError("3D rows must be packed without padding; image_stride must equal image_width * 2")
    return name, size, width, height


def depth_from_uint16(data, width, height):
    if len(data) != width * height * 2:
        raise ValueError("3D raw byte length does not match image_width * image_height * 2")
    # A view over owned bytes: the shared-memory read still copies, but no
    # decoding, rescaling or second pixel-buffer allocation occurs here.
    return np.frombuffer(data, dtype="<u2").reshape(height, width)
