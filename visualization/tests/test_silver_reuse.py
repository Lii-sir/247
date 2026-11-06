"""Synthetic equivalence and ownership checks for silver/2D geometry."""
import importlib
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'tests'))
from algorithm.detect_board import DetectBoard
from reference_silver_continuity import measure_continuity as reference
silver=importlib.import_module('algorithm.2D.silver_inspection')


class SilverReuseTests(unittest.TestCase):
    def fixture(self):
        shape=(150,210);chip=np.zeros(shape,np.uint8)
        chip[31:61,45:85]=255;chip[73:104,119:152]=255
        silver_mask=cv2.dilate(chip,np.ones((17,17),np.uint8));wire=np.zeros_like(chip);wire[:,100:105]=255
        return shape,[dict(class_name='chip',mask=chip),dict(class_name='silver',mask=silver_mask),dict(class_name='wire',mask=wire)]

    def test_component_crops_preserve_full_reference_and_mask_shapes(self):
        rng=np.random.default_rng(788)
        for case in range(24):
            shape,items=self.fixture()
            items[0]['mask']=np.roll(items[0]['mask'],case%3-1,axis=1)
            items[1]['mask'][rng.random(shape)<.2]=0
            config=silver.ContinuitySettings(min_chip_area_px=5,outward_length_px=case%12+1,
                occlusion_dilation_px=case%3,sector_count=(4,72,720)[case%3])
            actual=silver.measure_continuity(items,shape,config);expected=reference(items,shape,config)
            self.assertEqual(actual.summary(),expected.summary())
            for name in ('chip_mask','ring_mask','silver_mask','occlusion_mask','missing_mask'):
                np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))

    def test_one_inspection_reuses_union_and_chip_components(self):
        shape,items=self.fixture()
        original=silver._union_masks
        with patch.object(silver,'_union_masks',wraps=original) as unions, \
             patch.object(silver.cv2,'connectedComponentsWithStats',wraps=cv2.connectedComponentsWithStats) as ccl:
            result=silver.inspect_masks(items,shape,dict(overflow=dict(boundary=dict(mode='chip_expansion',min_chip_area_px=5)),
                continuity=dict(min_chip_area_px=5)))
        requests=[tuple(call.args[1]) for call in unions.call_args_list]
        self.assertEqual(requests.count(('chip',)),1);self.assertEqual(requests.count(('silver',)),1)
        self.assertEqual(ccl.call_count,2) # Shared chip labels and overflow candidates.
        self.assertLess(ccl.call_args_list[0].args[0].size,np.prod(shape))
        self.assertTrue(result['enabled'])

    def test_cache_does_not_cross_image_generations(self):
        shape,items=self.fixture()
        expected=silver.measure_continuity(items,shape,silver.ContinuitySettings()).summary()
        items[1]['mask'][:]=0
        with self.assertRaisesRegex(ValueError,'silver'):
            silver.measure_continuity(items,shape,silver.ContinuitySettings())
        self.assertGreater(expected['silver_area_px'],0)

    def test_public_geometry_parallel_matches_serial_and_never_writes_instances(self):
        detector=DetectBoard(lambda _:None);self.addCleanup(detector._instance_tasks.close)
        _,items=self.fixture()
        for i,item in enumerate(items):item.update(instance_id=i+1,class_id=i,confidence=.9,is_ok=True);item['mask'].flags.writeable=False
        keys=[set(item) for item in items]
        expected=detector._public_2d_instances(items,(100,200,310,350))
        detector._instance_tasks.configure(3)
        original=detector._public_2d_instance;barrier=threading.Barrier(3)
        def child(item,crop):barrier.wait(timeout=3);return original(item,crop)
        with patch.object(detector,'_public_2d_instance',side_effect=child):
            actual=detector._public_2d_instances(items,(100,200,310,350))
        self.assertEqual(actual,expected)
        self.assertEqual([set(item) for item in items],keys)


if __name__=='__main__':unittest.main()
