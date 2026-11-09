"""Production Pitch ordering/coordinate contracts without models or hardware."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'tests'))
from algorithm import detect_board as board_module
from algorithm import detect_3D as standalone_module
from algorithm.board_pipeline import BoardPipeline
from algorithm.height_measurement import height_points_to_board
from algorithm.tilt_correction import correct_smartray_pitch,correct_smartray_depth,production_pitch_parameters
from pipeline_fakes import FakeDetector


def settings():
    return dict(model='smartray_pitch',enabled_3d=True,enabled_2d=False,input_state='uncorrected',
                image_role='independent_camera',angle_deg=30,scan_pitch_mm=.01,
                z_step_mm=.00105,gray_offset=32768.,z_origin_mm=0.)


def raw_board():
    d=np.zeros((100,20),np.uint16);d[30:60]=32958
    return d


class ProductionPitchTests(unittest.TestCase):
    def setUp(self):
        self.raw=raw_board();_,self.corrected,self.info=correct_smartray_pitch(None,self.raw,settings())
        self.detector=board_module.DetectBoard(lambda text:None)
        self.detector.expected_piece_count=1
        self.detector.segmentation_config=board_module.SegmentationConfig(expected_piece_count=1,max_internal_gap_height=0)
        self.detector.pitch_config=settings()

    def test_config_default_disabled_and_enabled_invalid_is_rejected(self):
        config=json.loads((ROOT/'config/3D/config.json').read_text(encoding='utf-8-sig'))
        config['smartray_pitch_3d']['parameters']['enabled_3d']=False
        self.assertFalse(production_pitch_parameters(config)['enabled_3d'])
        self.assertNotIn('perspective_registration_3d',config)
        for bad in ({'scan_pitch_mm':0},{'enabled_2d':True},{'z_step_mm':.002}):
            config['smartray_pitch_3d']['parameters']={**settings(),**bad}
            with self.assertRaises(ValueError):production_pitch_parameters(config)

    def test_three_d_only_config_uses_fixed_model_and_omits_two_d_metadata(self):
        config=json.loads((ROOT/'config/3D/config.json').read_text(encoding='utf-8-sig'))
        values=config['smartray_pitch_3d']['parameters']
        for key in ('model','enabled_2d','image_role','input_state'):
            self.assertNotIn(key,values)
        values.update(enabled_3d=True,angle_deg=30,scan_pitch_mm=.01)
        params=production_pitch_parameters(config)
        self.assertEqual(params['model'],'smartray_pitch')
        self.assertNotIn('enabled_2d',params);self.assertNotIn('image_role',params)
        depth,info=correct_smartray_depth(self.raw,params)
        np.testing.assert_array_equal(depth,self.corrected)
        for key in ('enabled_2d','requested_2d','image_role','image_transform_original_to_corrected'):
            self.assertNotIn(key,info)
        self.assertNotIn('input_state',params);self.assertNotIn('input_state',info)
        for legacy in ('unknown','sdk_corrected'):
            config['smartray_pitch_3d']['parameters']['input_state']=legacy
            normalized=production_pitch_parameters(config)
            self.assertNotIn('input_state',normalized)
            processed,info=correct_smartray_depth(self.raw,{**normalized,'input_state':legacy})
            np.testing.assert_array_equal(processed,self.corrected);self.assertTrue(info['applied'])
        params['enabled_3d']=False
        skipped,info=correct_smartray_depth(self.raw,params)
        self.assertIs(skipped,self.raw);self.assertEqual(info['reason'],'disabled')

    def test_formal_config_loads_without_perspective_block(self):
        from algorithm.config import load_runtime_config
        config=load_runtime_config((ROOT/'config/config.json').read_text(encoding='utf-8-sig'),str(ROOT/'config/config.json'))
        config['smartray_pitch_3d']['parameters']['enabled_3d']=False
        self.detector._load_config(json.dumps(config))
        self.assertFalse(self.detector.pitch_config['enabled_3d'])
        self.assertFalse(hasattr(self.detector,'perspective_registration_config'))

    def test_prepare_depth_is_same_algorithm_and_does_not_mutate_input(self):
        before=self.raw.copy();d,info=self.detector.prepare_3d_depth(self.raw)
        np.testing.assert_array_equal(d,self.corrected);np.testing.assert_array_equal(self.raw,before)
        self.assertTrue(info['applied'])

    def test_concurrent_depth_correction_precedes_segmentation_and_joint(self):
        outer=self
        class Detector(FakeDetector):
            def __init__(self):super().__init__(count=1);self.calls=0
            def _read_3d(self,req):return outer.raw
            def prepare_3d_depth(self,depth):self.calls+=1;return outer.detector.prepare_3d_depth(depth)
            def measure_prepared_piece(self,pid,entry,depth,piece,correction=None):
                np.testing.assert_array_equal(depth,outer.corrected)
                outer.assertTrue(correction['applied']);outer.assertNotEqual(piece['y1'],30)
                return super().measure_prepared_piece(pid,entry,depth,piece,correction)
        detector=Detector()
        with ThreadPoolExecutor(2) as cpu,ThreadPoolExecutor(1) as gpu:
            pipe=BoardPipeline(detector,cpu,gpu)
            future=pipe.submit_3d({'board_id':'A'})
            self.assertTrue(pipe.submit_2d({'board_id':'A','piece_id':1}).result(5)['ok'])
            response=future.result(5)
        self.assertTrue(response['ok']);self.assertEqual(detector.calls,1)
        self.assertEqual(response['results_board']['depth_coordinate_space'],'pitch_corrected_board_3d')

    def test_sync_board_passes_corrected_board_once_to_measurement(self):
        detector=self.detector;detector._piece_cache={'A':{1:{}}};detector._read_3d=Mock(return_value=self.raw)
        def measure(pid,entry,depth,piece,correction):
            np.testing.assert_array_equal(depth,self.corrected);self.assertTrue(correction['applied'])
            self.assertNotEqual(piece['y1'],30)
            return {'registration':{'is_valid':True},'is_ok':True}
        detector.measure_prepared_piece=Mock(side_effect=measure)
        response=detector.handle_detect_3d_board({'board_id':'A'})
        self.assertTrue(response['ok'],response);detector.measure_prepared_piece.assert_called_once()
        self.assertTrue(response['results_board']['pitch_correction']['applied'])

    def test_batch_and_stream_measurements_use_affine_and_corrected_coordinates(self):
        detector=self.detector
        registration=dict(is_valid=True,affine_valid=True,scale=1,tx=0,ty=0,homography_2d_to_3d=np.eye(3).tolist())
        def measure(depth,*args,**kwargs):
            self.assertEqual(int(depth.max()),int(self.corrected.max()))
            return dict(is_ok=True,wire_measurements=[{'height_3d':{'point':{'x':1,'y':2},'is_valid':True}}],solder_measurements=[],reference_planes={})
        detector._measure_piece=Mock(side_effect=measure)
        entry=dict(registration_mask=np.zeros((8,8),np.uint8),gray_image=np.zeros((8,8),np.uint8),instances=[],image_shape=[8,8],crop_box=(0,0,8,8),roi_boxes=[])
        piece=board_module.segment_depth_image(self.corrected,detector.segmentation_config)[0]
        with patch.object(board_module,'register_platform_masks',return_value=registration),patch.object(board_module,'refine_affine_registration',return_value=registration):
            result=detector.measure_prepared_piece(1,entry,self.corrected,piece,self.info)
            self.assertEqual(result['wire_measurements'][0]['height_3d']['point']['y'],piece['y1']+2)
            self.assertEqual(result['height_point_coordinate_space'],'pitch_corrected_board_3d')
            self.assertEqual(height_points_to_board(result),result)
            detector.model=object();detector._synchronize_model=Mock()
            detector._validate_descriptors=Mock(return_value=[{'piece_id':1}])
            detector._read_2d=Mock(return_value=np.zeros((8,8,3),np.uint8));detector._read_3d=Mock(return_value=self.raw)
            detector._stream_crop=Mock(return_value=(np.zeros((8,8,3),np.uint8),(0,0,8,8),None,[]))
            detector._predict_batch=Mock(return_value=[object()]);detector._extract_instances=Mock(return_value=[])
            response=detector.handle_detect({'images_2d':[],'image_3d':{}})
        self.assertTrue(response['ok'],response)
        result=response['results_board']['pieces'][0]
        self.assertEqual(result['height_point_coordinate_space'],'pitch_corrected_board_3d')
        self.assertTrue(result['pitch_correction']['applied'])
        self.assertFalse(hasattr(board_module,'estimate_perspective'))

    def test_standalone_depth_corrected_before_segment_and_measure(self):
        config=json.loads((ROOT/'config/3D/config.json').read_text(encoding='utf-8-sig'))
        # This synthetic single-piece fixture must not inherit production's
        # calibrated ten-piece fixed ROIs.
        config['segmentation_3d']['parameters']={'expected_piece_count':1,'mode':'auto','fixed_rois':[]}
        config['smartray_pitch_3d']['parameters']=settings()
        detector=standalone_module.Detect3D()
        self.assertTrue(detector.handle_init({'config_text':json.dumps(config)})['ok'])
        raw=self.raw.astype('<u2').tobytes()
        def measure(depth,pieces):
            np.testing.assert_array_equal(depth,self.corrected);self.assertNotEqual(pieces[0]['y1'],30)
            return [{**pieces[0],'is_ok':True}]
        with patch.object(standalone_module,'read_shared_memory',return_value=raw),patch.object(standalone_module,'measure_depth',side_effect=measure):
            response=detector.detect('test',len(raw),'uint16',self.raw.shape[1],self.raw.shape[0])
        self.assertTrue(response['ok'],response)
        self.assertEqual(response['results_3d']['depth_coordinate_space'],'pitch_corrected_board_3d')


if __name__=='__main__':unittest.main()
