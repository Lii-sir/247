"""SmartRay Studio4 6.10: X'=X+Z*sin(beta), Y'=Y, Z'=Z*cos(beta).
X is transport. Inverse: Z=Z'/cos(beta), X=X'-Z'*tan(beta).
"""
from __future__ import annotations
import math
import time
from functools import lru_cache
import numpy as np
from algorithm.instance_tasks import InstanceTasks
from algorithm.timing import record_duration, timed_call, note, current_timings
from transport.depth_input import depth_coordinate_space

MODEL = 'smartray_pitch'


@lru_cache(maxsize=8)
def _depth_lookup(angle, pitch, step, offset, origin, direction):
    """Immutable per-calibration tables, using the original float operation order."""
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    tangent = math.tan(math.radians(angle))
    codes = np.arange(65536, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        z = (codes-offset)*step + origin
        if direction == 'recover_ideal':
            shift, corrected_z = -z*tangent, z/cosine
        else:
            shift, corrected_z = z*sine, z*cosine
        shifts = shift/pitch
        raw = np.rint((corrected_z-origin)/step + offset)
    valid = (codes > 0) & (codes < 65535) & (raw >= 1) & (raw <= 65534)
    values = np.zeros(65536, np.uint16)
    values[valid] = raw[valid].astype(np.uint16)
    shifts.setflags(write=False)
    values.setflags(write=False)
    return shifts, values


def _output_buffer(depth, out, image=None):
    if out is None:
        return np.empty(depth.shape, np.uint16)
    if (not isinstance(out, np.ndarray) or out.dtype != np.uint16 or out.shape != depth.shape
            or not out.flags.c_contiguous or not out.flags.writeable):
        raise ValueError('Pitch输出须为同尺寸、可写、C连续的uint16数组')
    if np.shares_memory(depth, out) or (isinstance(image, np.ndarray) and np.shares_memory(image, out)):
        raise ValueError('Pitch输入输出不能共享像素内存')
    return out


def _unchanged_depth(depth, out, image):
    if out is None:
        return depth
    target = _output_buffer(depth, out, image)
    np.copyto(target, depth)
    return target


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


def correct_smartray_depth(depth, settings, *, tasks=None, out=None):
    """Production-only API: no 2D input, mapping or 2D metadata."""
    internal = {**(settings or {}), 'input_state': 'uncorrected'}
    _, corrected, info = correct_smartray_pitch(None, depth, internal, tasks=tasks, out=out)
    hidden = {'enabled_2d', 'requested_2d', 'image_role', 'image_transform_original_to_corrected', 'input_state'}
    return corrected, {key: value for key, value in info.items() if key not in hidden}


def _number(settings, key, default):
    value = float(settings.get(key, default))
    if not math.isfinite(value):
        raise ValueError(f'{key}必须是有限数')
    return value


def correct_smartray_pitch(image, depth, settings, *, tasks=None, out=None):
    prepare_started = time.perf_counter()
    settings = settings or {}
    want2 = bool(settings.get('enabled_2d', False))
    want3 = bool(settings.get('enabled_3d', False))
    info = dict(model=MODEL, enabled_2d=False, enabled_3d=False, applied=False,
                requested_2d=want2, requested_3d=want3,
                resampling='nearest_forward_max_z',
                image_transform_original_to_corrected=np.eye(3).tolist())
    if not want2 and not want3:
        info['reason'] = 'disabled'
        return image, _unchanged_depth(depth, out, image), info
    if settings.get('model') != MODEL:
        raise ValueError('旧版旋转参数不能用于SmartRay；请重新设置SmartRay Pitch参数')
    state = settings.get('input_state', 'unknown')
    if state == 'sdk_corrected':
        info['reason'] = 'already_sdk_corrected'
        return image, _unchanged_depth(depth, out, image), info
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
    h, w = depth.shape

    workers = settings.get('projection_workers', 1)
    if type(workers) is not int or workers < 1:
        raise ValueError('projection_workers必须为正整数')
    lookup_started = time.perf_counter()
    shifts, values = _depth_lookup(angle, pitch, step, offset, origin, direction)
    record_duration('pitch_lookup_prepare_ms', (time.perf_counter()-lookup_started)*1000)
    allocate_started = time.perf_counter()
    output = _output_buffer(depth, out, image)
    output_flat = output.reshape(-1)
    corrected_image = np.empty_like(image) if want2 else image
    record_duration('pitch_output_allocate_ms', (time.perf_counter()-allocate_started)*1000)
    owned_tasks = tasks is None
    if owned_tasks:
        tasks = InstanceTasks(workers, thread_name_prefix="pitch")
    # Collisions can only happen along the scan axis. Each strip spans that
    # entire axis and owns a disjoint part of the output; no concurrent scatter
    # ever writes into another strip, even for very large depth shifts.
    extent = w if axis == 'rows' else h
    strip_count = min(workers, tasks.workers, extent)
    boundaries = np.linspace(0, extent, strip_count + 1, dtype=int)
    spans = list(zip(boundaries[:-1], boundaries[1:]))
    record_duration('pitch_prepare_ms', (time.perf_counter()-prepare_started)*1000)

    def project_strip(span):
        started = tick = time.perf_counter()
        durations = {}
        # Local clocks avoid a shared dictionary/log lock on every substep.
        def checkpoint(name):
            nonlocal tick
            now = time.perf_counter()
            durations[name] = (now-tick)*1000
            tick = now
        left, right = span
        source = depth[:, left:right] if axis == 'rows' else depth[left:right, :]
        target = output[:, left:right] if axis == 'rows' else output[left:right, :]
        target.fill(0)
        row_offset, col_offset = (0, left) if axis == 'rows' else (left, 0)
        if want2:
            image_target = corrected_image[:, left:right] if axis == 'rows' else corrected_image[left:right, :]
            image_target.fill(0)
        th, tw = source.shape
        checkpoint('pitch_strip_clear_work_ms')
        rows, cols = np.nonzero((source > 0) & (source < 65535))
        input_count = len(rows)
        checkpoint('pitch_valid_points_work_ms')
        codes = source[rows, cols]
        mapped = shifts[codes]
        raw = values[codes]
        checkpoint('pitch_lookup_work_ms')
        # Keep q + continuous shift before rint. Rounding only the shift would
        # change half-pixel ties and coordinate-addition rounding.
        np.add(mapped, rows if axis == 'rows' else cols, out=mapped)
        np.rint(mapped, out=mapped)
        checkpoint('pitch_round_coordinates_work_ms')
        valid = (mapped >= 0) & (mapped < (th if axis == 'rows' else tw)) & (raw != 0)
        dst = mapped[valid].astype(np.intp)
        if axis == 'rows':
            dst *= w
            dst += cols[valid] + col_offset
        else:
            dst += (rows[valid] + row_offset)*w
        raw = raw[valid]
        checkpoint('pitch_filter_indices_work_ms')
        np.maximum.at(output_flat, dst, raw)
        checkpoint('pitch_scatter_max_work_ms')
        if want2:
            # Reuse the same projected points instead of projecting a second
            # time. Ties choose the greatest source index along the scan axis,
            # exactly as in the original whole-board algorithm.
            src = rows[valid]*tw + cols[valid]
            winners = np.full(th*tw, -1, np.int64)
            select = raw == output_flat[dst]
            rr, cc = np.divmod(dst, w)
            local_dst = (rr-row_offset)*tw + cc-col_offset
            np.maximum.at(winners, local_dst[select], src[select])
            target_rows, target_cols = np.nonzero(winners.reshape(th, tw) >= 0)
            chosen = winners.reshape(th, tw)[target_rows, target_cols]
            source_rows, source_cols = np.divmod(chosen, tw)
            corrected_image[target_rows+row_offset, target_cols+col_offset] = image[
                source_rows+row_offset, source_cols+col_offset]
        checkpoint('pitch_intensity_work_ms')
        output_count = int(np.count_nonzero(target))
        checkpoint('pitch_output_count_work_ms')
        return dict(span=[int(left), int(right)], timings=durations,
                            wall_ms=(time.perf_counter()-started)*1000,
                            start_delay_ms=(started-dispatched)*1000,
                            input_valid_points=input_count, projected_points=len(dst),
                            output_valid_points=output_count)

    dispatched = time.perf_counter()
    work_before = current_timings().get('pitch_projection_work_ms', 0.)
    try:
        projected = timed_call('pitch_projection_wall_ms', tasks.map,
                               project_strip, spans, timing_stage='pitch_projection_work_ms')
    finally:
        if owned_tasks:
            timed_call('pitch_executor_close_ms', tasks.close)
    merge_started = time.perf_counter()
    # Workers already wrote their disjoint pixels. Only combine scalar stats.
    input_count = sum(detail['input_valid_points'] for detail in projected)
    projected_count = sum(detail['projected_points'] for detail in projected)
    count = sum(detail['output_valid_points'] for detail in projected)
    record_duration('pitch_output_merge_ms', (time.perf_counter()-merge_started)*1000)
    details = projected
    if details:
        for stage in details[0]['timings']:
            record_duration(stage, sum(detail['timings'][stage] for detail in details))
        work = current_timings().get('pitch_projection_work_ms', 0.)-work_before
        measured = sum(sum(detail['timings'].values()) for detail in details)
        record_duration('pitch_projection_other_work_ms', max(0., work-measured))
        record_duration('pitch_projection_max_strip_ms', max(detail['wall_ms'] for detail in details))
        record_duration('pitch_projection_start_delay_max_ms', max(detail['start_delay_ms'] for detail in details))
        note('pitch_projection_details', strips=details)
    if want2:
        image = corrected_image
        info['image_transform_original_to_corrected'] = None
    info.update(projection_workers=strip_count)
    info.update(enabled_2d=want2, enabled_3d=True, applied=True, angle_deg=angle,
                input_state=state, image_role=role, scan_axis=axis, scan_pitch_mm=pitch,
                trigger_mode=trigger, transform_direction=direction, z_step_mm=step,
                gray_offset=offset, z_origin_mm=origin, input_valid_points=input_count,
                output_valid_points=count, discarded_points=input_count-projected_count,
                collision_points=projected_count-count,
                coordinate_mapping='depth_dependent_transport_shift')
    return image, output, info
