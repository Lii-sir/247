"""External Pitch preparation and detection-only timing/coordinate contracts."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests')]
from algorithm import detect_board as board_module, detect_3D as standalone_module
from algorithm.board_pipeline import BoardPipeline
from algorithm.depth_preprocessing import prepare_depth_for_detection, preprocessing_parameters
from algorithm.tilt_correction import correct_smartray_depth
from transport.depth_input import input_depth_correction
from pipeline_fakes import FakeDetector


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
        self.detector.segmentation_config=board_module.SegmentationConfig(expected_piece_count=1,max_internal_gap_height=0)

    def test_external_parameters_validate_calibration_and_keep_switch(self):
        config=json.loads((ROOT/'config/3D/config.json').read_text(encoding='utf-8-sig'))
        config['smartray_pitch_3d']['parameters'].update(enabled_3d=False)
        self.assertFalse(preprocessing_parameters(config)['enabled_3d'])
        for bad in ({'scan_pitch_mm':0},{'enabled_2d':True},{'z_step_mm':.002}):
            config['smartray_pitch_3d']['parameters']={**settings(),**bad}
            with self.assertRaises(ValueError):preprocessing_parameters(config)

    def test_formal_load_has_no_pitch_runtime_or_validation(self):
        from algorithm.config import load_runtime_config
        config=load_runtime_config((ROOT/'config/config.json').read_text(encoding='utf-8-sig'),str(ROOT/'config/config.json'))
        config['smartray_pitch_3d']['parameters']={'enabled_3d':True,'scan_pitch_mm':0}
        with patch('algorithm.tilt_correction.correct_smartray_depth',side_effect=AssertionError('detector corrected')):
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
                outer.assertTrue(correction['applied']);outer.assertNotEqual(piece['y1'],30)
                return super().measure_prepared_piece(pid,entry,depth,piece,correction)
        with ThreadPoolExecutor(2) as cpu,ThreadPoolExecutor(1) as gpu:
            pipe=BoardPipeline(Detector(),cpu,gpu)
            try:
                future=pipe.submit_3d({'board_id':'A','pitch_correction':self.info})
                self.assertTrue(pipe.submit_2d({'board_id':'A','piece_id':1}).result(5)['ok'])
                response=future.result(5)
            finally:pipe.close()
        self.assertTrue(response['ok'],response)
        self.assertEqual(response['results_board']['depth_coordinate_space'],'pitch_corrected_board_3d')
        self.assertFalse(any('pitch_' in key for key in response['timings']))

    def test_sync_board_passes_prepared_board_and_metadata_to_measurement(self):
        detector=self.detector
        detector._piece_cache={'A':{1:{}}};detector._read_3d=Mock(return_value=self.corrected)
        def measure(pid,entry,depth,piece,correction):
            self.assertIs(depth,self.corrected);self.assertTrue(correction['applied'])
            self.assertNotEqual(piece['y1'],30)
            return {'registration':{'is_valid':True},'is_ok':True}
        detector.measure_prepared_piece=Mock(side_effect=measure)
        response=detector.handle_detect_3d_board({'board_id':'A','pitch_correction':self.info})
        self.assertTrue(response['ok'],response)
        self.assertNotIn('smartray_pitch_ms',response['timings'])

    def test_batch_uses_as_received_depth_and_corrected_coordinate_metadata(self):
        detector=self.detector
        registration=dict(is_valid=True,affine_valid=True,scale=1,tx=0,ty=0,homography_2d_to_3d=np.eye(3).tolist())
        detector._measure_piece=Mock(return_value=dict(is_ok=True,wire_measurements=[],solder_measurements=[],reference_planes={}))
        detector.model=object();detector._synchronize_model=Mock()
        detector._validate_descriptors=Mock(return_value=[{'piece_id':1}])
        detector._read_2d=Mock(return_value=np.zeros((8,8,3),np.uint8));detector._read_3d=Mock(return_value=self.corrected)
        detector._stream_crop=Mock(return_value=(np.zeros((8,8,3),np.uint8),(0,0,8,8),None,[]))
        detector._predict_batch=Mock(return_value=[object()]);detector._extract_instances=Mock(return_value=[])
        with patch.object(board_module,'register_platform_masks',return_value=registration),patch.object(board_module,'refine_affine_registration',return_value=registration):
            response=detector.handle_detect({'images_2d':[],'image_3d':{'pitch_correction':self.info}})
        self.assertTrue(response['ok'],response)
        self.assertTrue(response['results_board']['pitch_correction']['applied'])
        self.assertNotIn('smartray_pitch_ms',response['timings'])
        observed=detector._measure_piece.call_args.args[0]
        self.assertEqual(int(observed.max()),int(self.corrected.max()))

    def test_standalone_ignores_pitch_config_and_never_recooks_input(self):
        config=json.loads((ROOT/'config/3D/config.json').read_text(encoding='utf-8-sig'))
        config['segmentation_3d']['parameters']={'expected_piece_count':1,'mode':'auto','fixed_rois':[]}
        config['smartray_pitch_3d']['parameters']=settings()
        detector=standalone_module.Detect3D();self.addCleanup(detector.release)
        self.assertTrue(detector.handle_init({'config_text':json.dumps(config)})['ok'])
        for depth,info,space in ((self.corrected,self.info,'pitch_corrected_board_3d'),(self.raw,None,'board_3d')):
            raw=depth.astype('<u2').tobytes()
            def measure(observed,pieces):
                np.testing.assert_array_equal(observed,depth)
                return [{**pieces[0],'is_ok':True}]
            with patch.object(standalone_module,'read_shared_memory',return_value=raw),patch.object(standalone_module,'measure_depth',side_effect=measure):
                response=detector.detect('test',len(raw),'uint16',20,100,pitch_correction=info)
            self.assertTrue(response['ok'],response)
            self.assertEqual(response['results_3d']['depth_coordinate_space'],space)
            self.assertNotIn('smartray_pitch_ms',response['timings'])


if __name__=='__main__':unittest.main()
