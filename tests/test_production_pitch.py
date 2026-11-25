"""External Pitch preparation and detection-only timing/coordinate contracts."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests')]
from algorithm.application.board import detect_board as board_module
from algorithm.application.board.board_pipeline import BoardPipeline
from algorithm.vision3d.depth.depth_preprocessing import prepare_depth_for_detection, preprocessing_parameters
from algorithm.vision3d.depth.tilt_correction import correct_smartray_depth
from transport.depth_input import input_depth_correction
from pipeline_fakes import FakeDetector, depth_piece_request


def settings():
    return dict(model='smartray_pitch',enabled_3d=True,angle_deg=30,scan_pitch_mm=.01,
                z_step_mm=.00105,gray_offset=32768.,z_origin_mm=0.)


def raw_board():
    depth=np.zeros((100,20),np.uint16);depth[30:60]=32958
    return depth


class ProductionPitchTests(unittest.TestCase):
    def setUp(self):
        self.raw=raw_board()
        self.corrected,self.info=prepare_depth_for_detection(self.raw,settings())
        self.detector=board_module.DetectBoard(lambda _:None)
        self.addCleanup(self.detector.release)
        self.detector.expected_piece_count=1

    def test_external_parameters_validate_calibration_and_keep_switch(self):
        config=json.loads((ROOT/'config/3D/config.json').read_text(encoding='utf-8-sig'))
        config['smartray_pitch_3d']['parameters'].update(enabled_3d=False)
        self.assertFalse(preprocessing_parameters(config)['enabled_3d'])
        for bad in ({'scan_pitch_mm':0},{'enabled_2d':True},{'z_step_mm':.002}):
            config['smartray_pitch_3d']['parameters']={**settings(),**bad}
            with self.assertRaises(ValueError):preprocessing_parameters(config)

    def test_formal_load_has_no_pitch_runtime_or_validation(self):
        from algorithm.runtime.config import load_runtime_config
        config=load_runtime_config((ROOT/'config/config.json').read_text(encoding='utf-8-sig'),str(ROOT/'config/config.json'))
        config['smartray_pitch_3d']['parameters']={'enabled_3d':True,'scan_pitch_mm':0}
        with patch('algorithm.vision3d.depth.tilt_correction.correct_smartray_depth',side_effect=AssertionError('detector corrected')):
            self.detector._load_config(json.dumps(config))
        self.assertFalse(hasattr(self.detector,'_pitch_tasks'))
        self.assertFalse(hasattr(self.detector,'prepare_3d_depth'))
        self.assertFalse(hasattr(self.detector,'pitch_config'))

    def test_external_prepare_matches_original_formula_without_mutating_source(self):
        expected=correct_smartray_depth(self.raw,settings())[0]
        before=self.raw.copy()
        actual,info=prepare_depth_for_detection(self.raw,settings())
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(self.raw,before)
        self.assertTrue(info['applied'])
        self.assertFalse(any(key.endswith('_ms') for key in info))
        unchanged,info=prepare_depth_for_detection(self.raw,{'enabled_3d':False})
        self.assertIs(unchanged,self.raw);self.assertFalse(info['applied'])

    def test_metadata_validation_does_not_transform_pixels_or_accept_bad_claims(self):
        self.assertEqual(input_depth_correction({})['applied'],False)
        for info in ({}, {'applied':1}, {**self.info,'angle_deg':float('nan')},
                     {**self.info,'z_step_mm':0}, {**self.info,'model':'other'}):
            with self.assertRaises(ValueError):input_depth_correction({'pitch_correction':info})
        accepted=input_depth_correction({'pitch_correction':{**self.info,'smartray_pitch_ms':123}})
        self.assertNotIn('smartray_pitch_ms',accepted)

    def test_concurrent_pipeline_consumes_prepared_depth_without_second_correction(self):
        outer=self
        class Detector(FakeDetector):
            def __init__(self):super().__init__(count=1)
            def _read_3d(self,req):return outer.corrected
            def prepare_3d_depth(self,*args):raise AssertionError('second correction')
            def measure_prepared_piece(self,pid,entry,depth,piece,correction=None):
                np.testing.assert_array_equal(depth,outer.corrected)
                outer.assertTrue(correction['applied'])
                outer.assertEqual(piece['y1'], 0)
                return {**super().measure_prepared_piece(pid,entry,depth,piece,correction),
                        'depth_coordinate_space': piece['depth_coordinate_space']}
        with ThreadPoolExecutor(2) as cpu,ThreadPoolExecutor(1) as gpu:
            pipe=BoardPipeline(Detector(),cpu,gpu)
            try:
                self.assertTrue(pipe.submit_2d({'board_id':'A','piece_id':1}).result(5)['ok'])
                height, width = self.corrected.shape
                response = pipe.submit_3d_piece(depth_piece_request('A', pitch_correction=self.info,
                    image_height=height, image_width=width, data_size=height*width*2)).result(5)
            finally:pipe.close()
        self.assertTrue(response['ok'],response)
        self.assertEqual(response['results_piece']['depth_coordinate_space'],'pitch_corrected_piece_3d')
        self.assertFalse(any('pitch_' in key for key in response['timings']))

if __name__=='__main__':unittest.main()
