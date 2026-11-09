"""Caller-side depth preparation. Complete this before submitting detection."""
from algorithm.tilt_correction import correct_smartray_depth, production_pitch_parameters
from transport.depth_input import input_depth_correction


def preprocessing_parameters(config):
    from algorithm.config import feature_parameters
    height = feature_parameters(config, 'height_measurement_3d', required=False) or {}
    return production_pitch_parameters(config, float(height.get('gray_offset', 32768.)),
                                       float(height.get('vertical_resolution_mm', .00105)))


def prepare_depth_for_detection(depth, settings):
    corrected, info = correct_smartray_depth(depth, settings)
    return corrected, input_depth_correction(dict(pitch_correction=info))
