"""Describe a depth image already separated by the caller."""
from transport.depth_input import depth_coordinate_space, input_depth_correction, validate_depth_input


def piece_depth_geometry(request, piece_id):
    _, _, _, height = validate_depth_input(request)
    correction = input_depth_correction(request)
    piece = dict(piece_id=piece_id, y1=0, y2=height,
                 depth_coordinate_space=depth_coordinate_space(correction, piece_local=True))
    return piece, correction
