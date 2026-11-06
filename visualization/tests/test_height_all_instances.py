"""All detected height targets are measured, independently of 2D wire ROIs."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from algorithm import detect_board as board


class AllHeightTargetsTests(unittest.TestCase):
    def setUp(self):
        self.detector=board.DetectBoard(lambda _:None)
        self.addCleanup(self.detector._instance_tasks.close)
        self.detector.min_component_area=4
        self.detector.wire_rois=[(0,0,8,8)]
        self.detector.height_config=replace(self.detector.height_config,
            platform_roi_offsets=((0.,0.,2.,2.),),plane_min_valid_points=4,chip_min_valid_points=4)
        self.registration=dict(scale=1.,tx=0.,ty=0.,is_valid=True,homography_2d_to_3d=np.eye(3).tolist())
        self.crop=(0,0,200,160)
        self.depth=np.full((160,200),32768,np.uint16)
        self.items=[]
        for pid,kind,box,value in [(1,'wire',(30,40,110,52),33000),(2,'wire',(40,65,150,77),33050),
                                    (3,'thin',(30,90,150,102),33100),(4,'bond',(140,120,160,136),33200)]:
            x1,y1,x2,y2=box;mask=np.zeros(self.depth.shape,np.uint8);mask[y1:y2,x1:x2]=255
            self.depth[mask>0]=value
            self.items.append(dict(instance_id=pid,class_name=kind,mask=mask,
                _mask_2d_array=mask,_mask_3d_array=mask.copy(),
                bending_2d={'is_valid':False,'is_ok':False,'status':'outside_rois'}))

    def measure(self,items=None):
        return self.detector._measure_piece(self.depth,self.items if items is None else items,
            self.registration,crop_box=self.crop,circle=(10,10,5))

    def test_every_wire_thin_and_solder_outside_2d_rois_is_measured(self):
        result=self.measure()
        self.assertEqual([r['instance_id'] for r in result['wire_measurements']],[1,2,3])
        self.assertEqual([(r['source_instance_id'],r['endpoint']) for r in result['solder_measurements']],
                         [(3,'start'),(3,'end'),(4,'bond')])
        self.assertTrue(result['is_ok'])
        for record in result['wire_measurements']+result['solder_measurements']:
            self.assertTrue(record['height_3d']['is_valid'])
            self.assertGreater(record['height_3d']['point']['x'],8)

    def test_changing_2d_roi_does_not_change_height_masks_or_points(self):
        result=self.measure(copy.deepcopy(self.items))
        self.detector.wire_rois=[(0,0,200,160)]
        self.assertEqual(result,self.measure(copy.deepcopy(self.items)))

    def test_full_wire_region_can_select_high_point_beyond_former_roi(self):
        self.detector.wire_rois=[(30,40,60,52)]
        self.depth[40:52,80:110]=33400
        result=self.measure()['wire_measurements'][0]
        self.assertGreaterEqual(result['height_3d']['point']['x'],80)
        self.assertEqual(result['mask_2d']['bounding_box'],[30,40,110,52])

    def test_uncached_bending_is_not_computed_by_height_processing(self):
        items=copy.deepcopy(self.items)
        for item in items:item.pop('bending_2d')
        with patch.object(board._geometry,'measure_wire_bending',side_effect=AssertionError('2D only')):
            result=self.measure(items)
        self.assertEqual(len(result['wire_measurements']),3)
        self.assertTrue(result['is_ok'])

    def test_empty_or_cleaned_away_wire_has_invalid_record_not_silent_skip(self):
        items=copy.deepcopy(self.items)
        items[0]['_mask_2d_array']=np.zeros(self.depth.shape,np.uint8)
        items[1]['_mask_2d_array']=np.zeros(self.depth.shape,np.uint8)
        items[1]['_mask_2d_array'][66,40:100]=255
        result=self.measure(items)
        self.assertEqual([r['instance_id'] for r in result['wire_measurements']],[1,2,3])
        for record in result['wire_measurements'][:2]:
            self.assertFalse(record['height_3d']['is_valid'])
            self.assertIsNone(record['height_3d']['point'])

    def test_serial_parallel_measurements_equal_without_mask_mutation(self):
        serial=self.measure(copy.deepcopy(self.items))
        originals=copy.deepcopy(self.items)
        self.detector._instance_tasks.configure(4)
        actual=self.measure()
        self.assertEqual(actual,serial)
        for item,before in zip(self.items,originals):
            np.testing.assert_array_equal(item['_mask_2d_array'],before['_mask_2d_array'])
            np.testing.assert_array_equal(item['_mask_3d_array'],before['_mask_3d_array'])

    def test_production_joint_entry_measures_all_and_converts_points_to_board(self):
        depth=np.pad(self.depth,((10,0),(0,0)))
        entry=dict(image_shape=[200,160],registration_mask=np.ones(self.depth.shape,np.uint8),
            gray_image=np.zeros(self.depth.shape,np.uint8),instances=copy.deepcopy(self.items),
            roi_boxes=self.detector.wire_rois,crop_box=self.crop,circle=(10,10,5),wire_roi_checks=[])
        with patch.object(board,'register_platform_masks',return_value=self.registration), \
             patch.object(board,'refine_affine_registration',return_value=self.registration):
            result=self.detector.measure_prepared_piece(1,entry,depth,dict(y1=10,y2=170))
        self.assertEqual([r['instance_id'] for r in result['wire_measurements']],[1,2,3])
        self.assertGreaterEqual(result['wire_measurements'][0]['height_3d']['point']['y'],50)
        self.assertTrue(all(item['height_3d'] is not None for item in result['masks']))


if __name__=='__main__':unittest.main()
