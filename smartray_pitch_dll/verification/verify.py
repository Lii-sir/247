"""Synthetic DLL-vs-production parity only; no sensor, models or benchmarks."""
from pathlib import Path
import os
import math
import shutil
import subprocess
import sys
import importlib.util

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT / 'python'))
from algorithm.tilt_correction import correct_smartray_depth
reference_spec = importlib.util.spec_from_file_location('frozen_pitch_reference',
    ROOT / 'visualization/tests/fixtures/pitch_reference.py')
reference = importlib.util.module_from_spec(reference_spec)
reference_spec.loader.exec_module(reference)


def run(*args):
    subprocess.run(list(map(str, args)), check=True)


def main():
    compiler = Path(os.environ['WINDIR']) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    work = PROJECT / 'bin/verification'
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT / 'bin/Release/SmartRay.PitchCorrection.dll', work)
    runner = work / 'PitchRunner.exe'
    run(compiler, '/nologo', '/optimize+', '/warnaserror+', '/out:' + str(runner),
        '/reference:' + str(work / 'SmartRay.PitchCorrection.dll'), HERE / 'PitchRunner.cs')
    run(runner, 'contracts')
    rng = np.random.default_rng(781)
    random = rng.integers(1, 65535, (71, 53), dtype=np.uint16)
    random[::7] = 0
    random[::11, ::3] = 65535
    all_codes = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    cases = []
    for axis in ('rows', 'columns'):
        for direction in ('recover_ideal', 'document_forward'):
            for angle in (-15., 0., 6., 59.):
                cases.append((random, dict(scan_axis=axis, transform_direction=direction,
                                          angle_deg=angle, projection_workers=3)))
    cases += [(all_codes, dict(angle_deg=15., projection_workers=4)),
              (all_codes, dict(angle_deg=-6., scan_axis='columns', projection_workers=4)),
              (random, dict(angle_deg=6., z_origin_mm=1.25, gray_offset=32000., z_step_mm=.002)),
              (random, dict(angle_deg=15., enabled_3d=False)),
              (random[:, :1].copy(), dict(angle_deg=6., projection_workers=20)),
              (random[:1].copy(), dict(angle_deg=-6., scan_axis='columns', projection_workers=20)),
              (random, dict(angle_deg=6., scan_pitch_mm=1e-300)),
              (np.zeros((7, 9), np.uint16), dict(angle_deg=6.)),
              (np.full((7, 9), 65535, np.uint16), dict(angle_deg=6.))]
    # Exactly half-pixel shifts must round the coordinate, not the shift alone.
    half_pitch = 2 * (.00105 * math.tan(math.radians(6.)))
    for axis in ('rows', 'columns'):
        cases.append((np.full((8, 9), 32769, np.uint16),
                      dict(scan_axis=axis, angle_deg=6., scan_pitch_mm=half_pitch,
                           projection_workers=3)))
        # Adjacent floats around the tie and larger integer coordinates catch
        # invalid optimizations that pre-round the shift rather than q+shift.
        shape = (301, 9) if axis == 'rows' else (9, 301)
        for pitch in (half_pitch, np.nextafter(half_pitch, 0.), np.nextafter(half_pitch, np.inf)):
            cases.append((np.full(shape, 32769, np.uint16), dict(scan_axis=axis,
                angle_deg=6., scan_pitch_mm=float(pitch), projection_workers=4)))
    for index, (source, overrides) in enumerate(cases):
        config = dict(model='smartray_pitch', enabled_3d=True, scan_axis='rows',
            transform_direction='recover_ideal', angle_deg=6., scan_pitch_mm=.02,
            z_step_mm=.00105, gray_offset=32768., z_origin_mm=0., projection_workers=1)
        config.update(overrides)
        # Compare both optimized implementations with a frozen pre-optimization
        # formula, so a matching regression in Python and DLL cannot pass.
        reference_config = {**config, 'projection_workers': 1}
        with np.errstate(over='ignore', invalid='ignore'):
            expected = reference.correct_smartray_pitch(None, source,
                {**reference_config, 'input_state': 'uncorrected'})[1]
            optimized, info = correct_smartray_depth(source, reference_config)
        np.testing.assert_array_equal(expected, optimized, err_msg=f'Python case {index}: {config}')
        input_path, output_path = work / 'input.raw', work / 'output.raw'
        source.astype('<u2').tofile(input_path)
        run(runner, input_path, output_path, source.shape[1], source.shape[0],
            config['angle_deg'], config['scan_pitch_mm'], config['z_step_mm'],
            config['gray_offset'], config['z_origin_mm'],
            int(config['scan_axis'] == 'columns'), int(config['transform_direction'] == 'document_forward'),
            config['projection_workers'], int(config['enabled_3d']))
        actual = np.fromfile(output_path, dtype='<u2').reshape(source.shape)
        np.testing.assert_array_equal(expected, actual, err_msg=f'case {index}: {config}')
        print(f'PASS {index+1}: {source.shape} {config["scan_axis"]} {config["angle_deg"]} deg')
    print(f'{len(cases)} synthetic pixel-exact parity cases passed; no hardware tests.')


if __name__ == '__main__':
    main()
