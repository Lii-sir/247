"""Exact comparison with frozen Pitch projection, no real images or models."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from algorithm.tilt_correction import correct_smartray_pitch, correct_smartray_depth
from algorithm.instance_tasks import InstanceTasks
from algorithm.detect_board import DetectBoard
from algorithm.detect_3D import Detect3D
from algorithm.config import load_runtime_config
from fixtures.pitch_reference import correct_smartray_pitch as reference


class PitchParallelTests(unittest.TestCase):
    def test_projection_subtimings_are_merged_and_keep_pixels_unchanged(self):
        from algorithm.timing import timing_context, timed_call
        depth=np.random.default_rng(51).integers(32000,34000,(33,17),dtype=np.uint16)
        for workers in (1,3):
            settings=self.settings(projection_workers=workers)
            expected=reference(None,depth,settings)[1]
            timings,logs={},[]
            with timing_context(timings,phase='depth',log=logs.append):
                _,actual,info=timed_call('smartray_pitch_ms',correct_smartray_pitch,None,depth,settings)
            np.testing.assert_array_equal(actual,expected)
            records=[json.loads(line.removeprefix('[TIMING] ')) for line in logs]
            details=[row['strips'] for row in records if row['event']=='pitch_projection_details'][0]
            self.assertEqual(workers,len(details))
            self.assertEqual(info['input_valid_points'],sum(row['input_valid_points'] for row in details))
            stages=details[0]['timings'].keys()
            for stage in stages:
                self.assertAlmostEqual(timings[stage],sum(row['timings'][stage] for row in details))
                self.assertGreaterEqual(timings[stage],0)
            self.assertAlmostEqual(timings['pitch_projection_work_ms'],
                sum(timings[key] for key in stages)+timings['pitch_projection_other_work_ms'],places=7)
            self.assertEqual(timings['pitch_projection_max_strip_ms'],max(row['wall_ms'] for row in details))
            self.assertGreaterEqual(timings['smartray_pitch_ms'],timings['pitch_projection_wall_ms'])
            self.assertIn('pitch_prepare_ms',timings)
            self.assertIn('pitch_output_allocate_ms',timings)
            self.assertIn('pitch_output_merge_ms',timings)

    def test_timing_does_not_break_empty_depth_and_disabled_has_no_projection(self):
        from algorithm.timing import timing_context
        for shape in ((0,3),(3,0)):
            depth=np.zeros(shape,np.uint16)
            with timing_context({},phase='depth'):
                _,actual,_=correct_smartray_pitch(None,depth,self.settings())
            self.assertEqual(actual.shape,shape)
        timings={}
        with timing_context(timings,phase='depth'):
            correct_smartray_depth(np.zeros((3,4),np.uint16),{})
        self.assertEqual({},timings)

    def settings(self,**values):
        return dict(model='smartray_pitch',enabled_3d=True,input_state='uncorrected',
            scan_pitch_mm=.02,angle_deg=15.,**values)

    def test_exact_depth_intensity_and_metadata_for_axes_angles_and_workers(self):
        rng=np.random.default_rng(506)
        depth=rng.integers(32000,34000,(270,31),dtype=np.uint16)
        depth[rng.random(depth.shape)<.15]=0;depth[rng.random(depth.shape)<.03]=65535
        image=rng.integers(0,256,(*depth.shape,3),dtype=np.uint8)
        for axis in ('rows','columns'):
            for direction in ('recover_ideal','document_forward'):
                for angle in (-35.,0.,15.):
                    for intensity in (False,True):
                        settings=self.settings(scan_axis=axis,transform_direction=direction,
                            enabled_2d=intensity,image_role='smartray_intensity' if intensity else 'independent_camera')
                        settings['angle_deg']=angle
                        expected_image,expected_depth,expected_info=reference(image,depth,settings)
                        for workers in (1,4,40):
                            actual_image,actual_depth,actual_info=correct_smartray_pitch(image,depth,
                                {**settings,'projection_workers':workers})
                            with self.subTest(axis=axis,direction=direction,angle=angle,intensity=intensity,workers=workers):
                                np.testing.assert_array_equal(actual_depth,expected_depth)
                                np.testing.assert_array_equal(actual_image,expected_image)
                                for key,value in expected_info.items():self.assertEqual(actual_info[key],value,key)

    def test_collision_ties_clipping_nonzero_origin_and_empty_depth(self):
        rng=np.random.default_rng(42)
        for case in range(12):
            depth=rng.integers(1,65535,(32,27),dtype=np.uint16)
            if case%3==0:depth[:]=0
            if case%3==1:depth[:,::2]=65535
            image=np.arange(depth.size,dtype=np.uint16).reshape(depth.shape)
            settings=self.settings(enabled_2d=True,image_role='smartray_intensity',scan_axis=('rows','columns')[case%2],
                                   gray_offset=32000.,z_origin_mm=case*.001)
            settings['angle_deg']=(-1 if case%2 else 1)*59.
            settings['scan_pitch_mm']=.001 if case%4 else 1.
            expected=reference(image,depth,settings)
            actual=correct_smartray_pitch(image,depth,{**settings,'projection_workers':8})
            np.testing.assert_array_equal(actual[0],expected[0]);np.testing.assert_array_equal(actual[1],expected[1])
            for key,value in expected[2].items():self.assertEqual(actual[2][key],value)

    def test_persistent_pool_reuse_parallel_boards_and_readonly_inputs(self):
        tasks=InstanceTasks(4,thread_name_prefix='pitch');self.addCleanup(tasks.close)
        depth=np.random.default_rng(3).integers(32000,34000,(270,40),dtype=np.uint16)
        depth.flags.writeable=False
        settings=self.settings(projection_workers=4)
        expected=reference(None,depth,settings)[1]
        executor=tasks._pool
        with ThreadPoolExecutor(max_workers=2) as parents:
            futures=[parents.submit(correct_smartray_depth,depth,settings,tasks=tasks) for _ in range(2)]
            for future in futures:np.testing.assert_array_equal(future.result(timeout=10)[0],expected)
        self.assertIs(tasks._pool,executor)

    def test_equal_corrected_depth_collision_keeps_last_source_across_old_block_boundary(self):
        for axis in ('rows','columns'):
            shape=(520,12) if axis=='rows' else (12,520)
            depth=np.zeros(shape,np.uint16);image=np.zeros(shape,np.uint8)
            first=(255,5) if axis=='rows' else (5,255)
            last=(256,5) if axis=='rows' else (5,256)
            destination=(257,5) if axis=='rows' else (5,257)
            depth[first]=32770;depth[last]=32769
            image[first]=11;image[last]=22
            settings=self.settings(enabled_2d=True,image_role='smartray_intensity',scan_axis=axis,
                transform_direction='document_forward',projection_workers=4)
            settings.update(angle_deg=59.,scan_pitch_mm=math.sin(math.radians(59.))*.00105)
            actual=correct_smartray_pitch(image,depth,settings)
            expected=reference(image,depth,settings)
            np.testing.assert_array_equal(actual[0],expected[0]);np.testing.assert_array_equal(actual[1],expected[1])
            self.assertEqual(actual[0][destination],22)
            self.assertEqual(actual[2]['collision_points'],1)

    def test_external_preprocessor_owns_pitch_work_instead_of_detectors(self):
        board=DetectBoard(lambda _:None);standalone=Detect3D()
        self.addCleanup(board.release);self.addCleanup(standalone.release)
        config=load_runtime_config((ROOT/'config/config.json').read_text(encoding='utf-8-sig'),str(ROOT/'config/config.json'))
        config['smartray_pitch_3d']['parameters'].update(projection_workers=3,enabled_3d=True)
        board._load_config(json.dumps(config))
        self.assertFalse(hasattr(board,'_pitch_tasks'))
        response=standalone.handle_init(dict(config_text=json.dumps(config)))
        self.assertTrue(response['ok']);self.assertFalse(hasattr(standalone,'_pitch_tasks'))
        standalone.release()
        self.assertTrue(standalone.handle_init(dict(config_text=json.dumps(config)))['ok'])
        depth=np.full((25,30),33000,np.uint16)
        from algorithm.depth_preprocessing import preprocessing_parameters, prepare_depth_for_detection
        settings=preprocessing_parameters(config)
        actual=prepare_depth_for_detection(depth,settings)[0]
        expected=reference(None,depth,{**settings,'input_state':'uncorrected'})[1]
        np.testing.assert_array_equal(actual,expected)

    def test_bad_workers_and_disabled_or_sdk_input(self):
        depth=np.full((3,4),33000,np.uint16)
        for workers in (0,-1,True,2.5,'4'):
            with self.assertRaisesRegex(ValueError,'projection_workers'):
                correct_smartray_pitch(None,depth,self.settings(projection_workers=workers))
        for state,enabled in (('sdk_corrected',True),('unknown',False)):
            settings=self.settings();settings.update(input_state=state,enabled_3d=enabled,projection_workers=4)
            self.assertIs(correct_smartray_pitch(None,depth,settings)[1],depth)


if __name__=='__main__':unittest.main()
