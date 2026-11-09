"""Packed little-endian uint16 depth input; no image codec or gray conversion."""
import numpy as np
import math


def depth_coordinate_space(correction):
    return 'pitch_corrected_board_3d' if correction.get('applied') else 'board_3d'


def input_depth_correction(descriptor):
    """Validate external preprocessing metadata only; never transform pixels."""
    info = descriptor.get('pitch_correction')
    if info is None:
        return dict(applied=False, reason='external_preprocessing_not_reported')
    if not isinstance(info, dict) or type(info.get('applied')) is not bool:
        raise ValueError('pitch_correction requires a boolean applied field')
    if not info['applied']:
        return dict(applied=False, reason=str(info.get('reason', 'not_applied')))
    if info.get('model') != 'smartray_pitch':
        raise ValueError('external pitch_correction model must be smartray_pitch')
    result = dict(model='smartray_pitch', applied=True, enabled_3d=True,
                  reason='external_preprocessing', resampling='nearest_forward_max_z')
    for key in ('angle_deg', 'scan_pitch_mm', 'z_step_mm', 'gray_offset', 'z_origin_mm'):
        value = info.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('external pitch_correction requires finite ' + key)
        result[key] = value
    if abs(result['angle_deg']) >= 60 or result['scan_pitch_mm'] <= 0 or result['z_step_mm'] <= 0:
        raise ValueError('external pitch_correction has invalid angle or resolution')
    for key, choices in (('scan_axis', ('rows', 'columns')),
                         ('transform_direction', ('recover_ideal', 'document_forward'))):
        if info.get(key) not in choices:
            raise ValueError('external pitch_correction has invalid ' + key)
        result[key] = info[key]
    return result


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
