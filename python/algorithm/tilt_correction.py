"""SmartRay Studio4 6.10: X'=X+Z*sin(beta), Y'=Y, Z'=Z*cos(beta).
X is transport. Inverse: Z=Z'/cos(beta), X=X'-Z'*tan(beta).
"""
from __future__ import annotations
import math
import numpy as np

MODEL = 'smartray_pitch'


def production_pitch_parameters(config, gray_offset=32768., z_step_mm=.00105):
    """Optional production block; formal 2D inputs are independent photographs."""
    from algorithm.config import feature_parameters
    values = feature_parameters(config, 'smartray_pitch_3d', required=False) or {}
    if values.get('enabled_2d') or values.get('image_role', 'independent_camera') != 'independent_camera':
        raise ValueError('正式检测接收独立2D照片；smartray_pitch_3d仅支持3D补偿')
    # Old false/default 2D keys remain readable, but are not production parameters.
    values.pop('enabled_2d', None)
    values.pop('image_role', None)
    # Production has one switch. Legacy state keys no longer select a branch.
    values.pop('input_state', None)
    values.setdefault('model', MODEL)
    values.setdefault('z_step_mm', z_step_mm)
    values.setdefault('gray_offset', gray_offset)
    if values.get('enabled_3d'):
        if float(values['z_step_mm']) != z_step_mm or float(values['gray_offset']) != gray_offset:
            raise ValueError('Pitch深度标定须与测高的gray_offset/vertical_resolution_mm一致')
    # Validate through the same algorithm on an empty tiny input, without models.
    correct_smartray_depth(np.zeros((1, 1), np.uint16), values)
    return values


def correct_smartray_depth(depth, settings):
    """Production-only API: no 2D input, mapping or 2D metadata."""
    internal = {**(settings or {}), 'input_state': 'uncorrected'}
    _, corrected, info = correct_smartray_pitch(None, depth, internal)
    hidden = {'enabled_2d', 'requested_2d', 'image_role', 'image_transform_original_to_corrected', 'input_state'}
    return corrected, {key: value for key, value in info.items() if key not in hidden}


def depth_coordinate_space(correction):
    return 'pitch_corrected_board_3d' if correction.get('applied') else 'board_3d'


def _number(settings, key, default):
    value = float(settings.get(key, default))
    if not math.isfinite(value):
        raise ValueError(f'{key}必须是有限数')
    return value


def correct_smartray_pitch(image, depth, settings):
    settings = settings or {}
    want2 = bool(settings.get('enabled_2d', False))
    want3 = bool(settings.get('enabled_3d', False))
    info = dict(model=MODEL, enabled_2d=False, enabled_3d=False, applied=False,
                requested_2d=want2, requested_3d=want3,
                resampling='nearest_forward_max_z',
                image_transform_original_to_corrected=np.eye(3).tolist())
    if not want2 and not want3:
        info['reason'] = 'disabled'
        return image, depth, info
    if settings.get('model') != MODEL:
        raise ValueError('旧版旋转参数不能用于SmartRay；请重新设置SmartRay Pitch参数')
    state = settings.get('input_state', 'unknown')
    if state == 'sdk_corrected':
        info['reason'] = 'already_sdk_corrected'
        return image, depth, info
    if state != 'uncorrected':
        raise ValueError('请先确认输入是否已由Studio/SDK进行Pitch矫正')
    if want2 and not want3:
        raise ValueError('同步强度图必须随3D一起矫正，不能单独拉伸')
    role = settings.get('image_role', 'independent_camera')
    if role not in {'independent_camera', 'smartray_intensity'}:
        raise ValueError('未知2D图像来源')
    if role == 'independent_camera' and want2:
        raise ValueError('独立2D照片不使用SmartRay Pitch公式；请取消同步强度图矫正')
    if role == 'smartray_intensity' and (not want2 or image.shape[:2] != depth.shape):
        raise ValueError('SmartRay强度图必须与深度同尺寸、逐像素对应并同步矫正')
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError('Pitch测试要求原始单通道uint16深度图')
    angle = _number(settings, 'angle_deg', 0)
    if abs(angle) >= 60:
        raise ValueError('Pitch角绝对值须小于60度')
    axis = settings.get('scan_axis', 'rows')
    if axis not in {'rows', 'columns'}:
        raise ValueError('扫描方向必须是rows或columns')
    direction = settings.get('transform_direction', 'recover_ideal')
    if direction not in {'recover_ideal', 'document_forward'}:
        raise ValueError('请选择测量值还原或文档正向')
    trigger = settings.get('trigger_mode', 'external')
    if trigger == 'internal':
        speed = _number(settings, 'transport_rate_mm_s', 0)
        rate = _number(settings, 'scan_rate_hz', 0)
        if speed <= 0 or rate <= 0:
            raise ValueError('内触发需要速度(mm/s)>0和频率(profile/s)>0')
        pitch = speed / rate
    elif trigger == 'external':
        pitch = _number(settings, 'scan_pitch_mm', 0)
    else:
        raise ValueError('触发模式必须是external或internal')
    step = _number(settings, 'z_step_mm', .00105)
    offset = _number(settings, 'gray_offset', 32768)
    origin = _number(settings, 'z_origin_mm', 0)
    if pitch <= 0 or not math.isfinite(pitch) or step <= 0:
        raise ValueError('扫描步距(mm/profile)和Z分辨率必须大于0')
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    tangent = math.tan(math.radians(angle))
    h, w = depth.shape

    def project(start):
        block = depth[start:start+256]
        rows, cols = np.nonzero((block > 0) & (block < 65535))
        rows = rows + start
        z = (depth[rows, cols].astype(float)-offset)*step + origin
        if direction == 'recover_ideal':
            shift, corrected_z = -z*tangent, z/cosine
        else:
            shift, corrected_z = z*sine, z*cosine
        rr, cc = rows.copy(), cols.copy()
        if axis == 'rows':
            rr = np.rint(rows + shift/pitch).astype(np.int64)
        else:
            cc = np.rint(cols + shift/pitch).astype(np.int64)
        raw = np.rint((corrected_z-origin)/step + offset)
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w) & (raw >= 1) & (raw <= 65534)
        return len(rows), rr[valid]*w+cc[valid], rows[valid]*w+cols[valid], raw[valid].astype(np.uint16)

    output = np.zeros(h*w, np.uint16)
    input_count = projected_count = 0
    for start in range(0, h, 256):
        count, dst, _, raw = project(start)
        input_count += count
        projected_count += len(dst)
        np.maximum.at(output, dst, raw)
    if want2:
        # Copy the intensity of the winning depth sample; deterministic ties.
        winners = np.full(h*w, -1, np.int64)
        for start in range(0, h, 256):
            _, dst, src, raw = project(start)
            select = raw == output[dst]
            np.maximum.at(winners, dst[select], src[select])
        corrected = np.zeros_like(image)
        present = winners >= 0
        flat_shape = (h*w,) + image.shape[2:]
        corrected.reshape(flat_shape)[present] = image.reshape(flat_shape)[winners[present]]
        image = corrected
        # A depth-dependent remap is not a 3x3 homography.
        info['image_transform_original_to_corrected'] = None
    count = int(np.count_nonzero(output))
    info.update(enabled_2d=want2, enabled_3d=True, applied=True, angle_deg=angle,
                input_state=state, image_role=role, scan_axis=axis, scan_pitch_mm=pitch,
                trigger_mode=trigger, transform_direction=direction, z_step_mm=step,
                gray_offset=offset, z_origin_mm=origin, input_valid_points=input_count,
                output_valid_points=count, discarded_points=input_count-projected_count,
                collision_points=projected_count-count,
                coordinate_mapping='depth_dependent_transport_shift')
    return image, output.reshape(h,w), info
