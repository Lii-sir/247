"""CPU synthetic regressions for production instance concurrency; no models."""
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from algorithm.detect_board import DetectBoard
from algorithm.instance_tasks import InstanceTasks
from algorithm.timing import record_duration, timing_context, timed_call
from algorithm.config import load_runtime_config
from algorithm import detect_board as board
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures.wire_reference import LegacyWireReference


class InstanceParallelTests(unittest.TestCase):
    def setUp(self):
        self.detector = DetectBoard(lambda message: None)
        self.addCleanup(self.detector._instance_tasks.close)

    def test_tasks_overlap_keep_order_and_reuse_pool(self):
        pool = self.detector._instance_tasks
        pool.configure(2)
        executor = pool._pool
        barrier = threading.Barrier(2)

        def task(value):
            barrier.wait(timeout=3)
            return value, threading.current_thread().name

        for _ in range(2):
            result = pool.map(task, [2, 1])
            self.assertEqual([v for v, _ in result], [2, 1])
            self.assertTrue(all(name.startswith('instance') for _, name in result))
            self.assertIs(pool._pool, executor)

    def test_parallel_mask_mapping_equals_serial_without_input_mutation(self):
        instances = []
        for index in range(4):
            mask = np.zeros((64, 90), np.uint8)
            mask[5+index:20+index, 10+index:35+index] = 255
            mask.flags.writeable = False
            instances.append(dict(instance_id=index+1, class_name='wire', mask=mask))
        registration = dict(scale=.5, tx=3., ty=2.,
            homography_2d_to_3d=[[.5,.01,3.],[-.01,.51,2.],[0.,0.,1.]])
        before = copy.deepcopy(registration)
        expected = self.detector._align_instances(instances, (60,70), registration, (10,20,100,84))
        self.detector._instance_tasks.configure(3)
        actual = self.detector._align_instances(instances, (60,70), registration, (10,20,100,84))
        self.assertEqual(registration, before)
        for a, b in zip(actual, expected):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                if isinstance(a[key], np.ndarray):
                    np.testing.assert_array_equal(a[key], b[key])
                else:
                    self.assertEqual(a[key], b[key])
        self.assertEqual(self.detector._align_instances([], (60,70), registration), [])

    def test_child_timing_isolated_between_concurrent_pieces(self):
        self.detector._instance_tasks.configure(4)
        barrier = threading.Barrier(4)

        def piece(pid):
            totals = {}
            def child(value):
                barrier.wait(timeout=3)
                record_duration('synthetic_ms', value)
                return pid, value
            with timing_context(totals, phase='joint', piece_id=pid):
                values = self.detector._instance_tasks.map(child, [pid,pid], timing_stage='work_ms')
            return values, totals

        with ThreadPoolExecutor(max_workers=2) as parents:
            results = list(parents.map(piece, [1,2]))
        for pid, (values, totals) in enumerate(results, 1):
            self.assertEqual(values, [(pid,pid),(pid,pid)])
            self.assertEqual(totals['synthetic_ms'], 2*pid)
            self.assertGreaterEqual(totals['work_ms'], 0)

    def test_error_waits_for_running_siblings_and_retains_timings(self):
        pool = self.detector._instance_tasks
        pool.configure(2)
        barrier = threading.Barrier(2)
        finished = threading.Event()

        def child(value):
            barrier.wait(timeout=3)
            record_duration('child_ms', 1)
            if value == 0:
                raise ValueError('synthetic failure')
            finished.set()
            return value

        totals = {}
        with timing_context(totals, phase='joint'):
            with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                pool.map(child, [0,1])
        self.assertTrue(finished.is_set())
        self.assertEqual(totals['child_ms'], 2)
        self.assertEqual(pool.map(lambda value: value+1, [1,2]), [2,3])

    def test_configuration_serial_mode_and_release(self):
        config = load_runtime_config((ROOT/'config/config.json').read_text(encoding='utf-8-sig'),
                                     str(ROOT/'config/config.json'))
        config['smartray_pitch_3d']['parameters']['enabled_3d'] = False
        config['common_3d']['parameters']['threading']['instance_workers'] = 3
        self.detector._load_config(json.dumps(config))
        self.assertEqual(self.detector._instance_tasks.workers, 3)
        executor = self.detector._instance_tasks._pool
        self.detector._load_config(json.dumps(config))
        self.assertIs(self.detector._instance_tasks._pool, executor)
        for invalid in (0,-1,True,2.5,'2'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.detector._instance_tasks.configure(invalid)
        self.detector._instance_tasks.configure(1)
        self.assertTrue(executor._shutdown)
        name = threading.current_thread().name
        self.assertEqual(self.detector._instance_tasks.map(lambda _: threading.current_thread().name,[1,2]),[name,name])
        self.detector._instance_tasks.configure(2)
        with patch('algorithm.detect_board.torch.cuda.is_available', return_value=False):
            self.detector.release()
        self.assertIsNone(self.detector._instance_tasks._pool)
        self.detector._load_config(json.dumps(config))
        self.assertEqual(self.detector._instance_tasks.workers,3)


class WireParallelTests(unittest.TestCase):
    def setUp(self):
        self.detector = DetectBoard(lambda _: None)
        self.reference = LegacyWireReference(lambda _: None)
        self.addCleanup(self.detector._instance_tasks.close)
        self.detector._instance_tasks.configure(4)
        for detector in (self.detector, self.reference):
            detector.min_component_area = 4
            detector.height_config = replace(detector.height_config,
                plane_min_valid_points=4, chip_min_valid_points=4)

    @staticmethod
    def fixture():
        depth = np.full((140, 180), 33000, np.uint16)
        masks = []
        for index in range(4):
            mask = np.zeros((120,160), np.uint8)
            cv2.line(mask, (0, 15+index*23), (150, 25+index*23), 255, 7)
            masks.append(dict(instance_id=index+1, class_name='thin' if index%2 else 'wire',
                              _mask_2d_array=mask, _mask_3d_array=np.zeros_like(depth,np.uint8),
                              bending_2d={'is_valid':True,'is_ok':True}))
        return depth, masks

    def test_local_wire_path_equals_frozen_full_frame_reference(self):
        rng = np.random.default_rng(194)
        for case in range(28):
            depth, items = self.fixture()
            # Compare with the old algorithm over the entire instance domain;
            # empty instances now have explicit invalid records instead of skips.
            if case%4 == 0:
                items[0]['_mask_2d_array'][60:75, 20:95] = 255
            if case%5 == 0:
                items[2]['_mask_2d_array'][:] = 0
            depth[rng.random(depth.shape)<.03] = 0
            roi = [(0,0,160,120)]
            a = float(rng.uniform(.7,1.1))
            matrix = [[a,float(rng.uniform(-.015,.015)),float(rng.uniform(-5,5))],
                      [float(rng.uniform(-.015,.015)),a,float(rng.uniform(-3,3))],[0,0,1]]
            registration = dict(scale=a,tx=matrix[0][2],ty=matrix[1][2],homography_2d_to_3d=matrix)
            radius = case%3
            for detector in (self.detector,self.reference):
                detector.height_config = replace(detector.height_config,
                    wire_measurement_erode_px=radius,thin_measurement_erode_px=radius)
            args=(registration,dict(coefficients=[.02,-.01,32768.]),roi,(7,9,167,129))
            expected=self.reference._measure_wire_regions(depth,copy.deepcopy(items),*args)
            for workers in (1,4):
                self.detector._instance_tasks.configure(workers)
                before=copy.deepcopy(items)
                actual=self.detector._measure_wire_regions(depth,items,registration,args[1],crop_box=args[3])
                expected_by_id={item['instance_id']:item for item in expected}
                self.assertEqual([item['instance_id'] for item in actual],[1,2,3,4])
                with self.subTest(case=case,workers=workers):
                    for item in actual:
                        if item['instance_id'] in expected_by_id:
                            previous=expected_by_id[item['instance_id']]
                            self.assertEqual({k:v for k,v in item.items() if k!='region_id'},
                                             {k:v for k,v in previous.items() if k!='region_id'})
                        else:
                            self.assertFalse(item['height_3d']['is_valid'])
                            self.assertIsNone(item['height_3d']['point'])
                for item,original in zip(items,before):
                    np.testing.assert_array_equal(item['_mask_2d_array'],original['_mask_2d_array'])

    def test_wire_tasks_overlap_preserve_all_instance_order(self):
        depth,items=self.fixture()
        barrier=threading.Barrier(4)
        def task(depth,wire,*args):
            barrier.wait(timeout=3)
            return {'instance_id':wire['instance_id']}
        with patch.object(self.detector,'_measure_wire_region',side_effect=task):
            result=self.detector._measure_wire_regions(depth,items,{},None)
        self.assertEqual(result,[{'instance_id':i,'region_id':i} for i in range(1,5)])

    def test_thin_split_once_per_piece_and_reused_for_endpoints(self):
        depth,items=self.fixture()
        original=copy.deepcopy(items)
        registration=dict(scale=1.,tx=0.,ty=0.,is_valid=True)
        with patch.object(board,'split_thin_mask',wraps=board.split_thin_mask) as split:
            actual=self.detector._measure_piece(depth,items,registration,crop_box=(0,0,160,120))
        self.assertEqual(split.call_count,2)
        expected=self.reference._measure_piece(depth,original,registration,[(0,0,160,120)],(0,0,160,120))
        self.assertEqual(actual,expected)
        for item in items:
            if item['class_name']=='thin':
                self.assertIn('_thin_wire_mask_2d',item)
                self.assertEqual(len(item['_thin_solder_masks_2d']),2)

    def test_cleaning_is_local_and_roi_clipping_is_removed(self):
        mask=np.zeros((1000,1500),np.uint8);mask[400:410,700:800]=255
        item=dict(instance_id=1,class_name='wire',_mask_2d_array=mask,bending_2d={'is_ok':True})
        with patch.object(board._geometry,'clean_mask',wraps=board._geometry.clean_mask) as clean:
            self.detector._measure_wire_regions(np.full((1000,1500),33000,np.uint16),[item],
                dict(scale=1,tx=0,ty=0),dict(coefficients=[0,0,32768]),crop_box=(0,0,1500,1000))
        self.assertLess(clean.call_args.args[0].size,mask.size//20)
        self.assertFalse(hasattr(self.detector,'_mask_in_wire_rois'))


class SolderParallelTests(unittest.TestCase):
    def setUp(self):
        self.detector=DetectBoard(lambda _:None)
        self.reference=LegacyWireReference(lambda _:None)
        self.addCleanup(self.detector._instance_tasks.close)
        self.detector._instance_tasks.configure(4)
        for detector in (self.detector,self.reference):
            detector.min_component_area=4
            detector.height_config=replace(detector.height_config,plane_min_valid_points=4,
                chip_min_valid_points=4,platform_roi_offsets=((-2.,-3.,-1.,0.),))

    @staticmethod
    def fixture():
        shape=(180,240)
        depth=np.full(shape,33000,np.uint16)
        items=[]
        for pid,kind in ((1,'thin'),(2,'bond'),(3,'chip'),(4,'chip'),(5,'wire'),(6,'bond')):
            mask=np.zeros(shape,np.uint8)
            if pid==1:cv2.line(mask,(30,120),(175,85),255,7)
            elif pid==2:mask[80:90,150:166]=255
            elif pid==3:
                mask[30:140,100:220]=255;mask[70:100,140:180]=0
            elif pid==4:mask[65:110,135:185]=255
            elif pid==5:cv2.line(mask,(30,155),(210,160),255,7)
            else:mask[20:30,20:40]=255
            if kind=='chip':depth[mask>0]=33500
            elif kind in {'bond','thin'}:depth[mask>0]=33900
            items.append(dict(instance_id=pid,class_name=kind,_mask_2d_array=mask,
                              _mask_3d_array=mask.copy()))
            if kind in {'wire','thin'}:items[-1]['bending_2d']={'is_valid':True,'is_ok':True}
        return depth,items

    def test_complete_measurement_equals_frozen_reference_serial_and_parallel(self):
        rng=np.random.default_rng(221)
        for case in range(12):
            depth,items=self.fixture()
            depth[rng.random(depth.shape)<.02]=0
            registration=dict(scale=1.,tx=0.,ty=0.,is_valid=True)
            if case%3==0:
                registration.update(scale=.95,tx=1.5,ty=-1.25,
                    homography_2d_to_3d=[[.95,.005,1.5],[-.005,.96,-1.25],[0,0,1]])
                for item in items:
                    item['_mask_3d_array']=board.align_crop_mask_to_depth(
                        item['_mask_2d_array'],(0,0,240,180),depth.shape,registration)
            for detector in (self.detector,self.reference):
                detector.height_config=replace(detector.height_config,
                    bond_measurement_erode_px=case%3,thin_solder_measurement_erode_px=case%2,
                    thin_measurement_erode_px=case%2)
            if case==11:
                for detector in (self.detector,self.reference):
                    detector.height_config=replace(detector.height_config,plane_min_valid_points=100000)
            args=(registration,[(0,0,240,180)],(0,0,240,180),(30,70,10))
            originals=copy.deepcopy(items)
            expected=self.reference._measure_piece(depth,originals,*args)
            for workers in (1,4):
                self.detector._instance_tasks.configure(workers)
                actual_items=copy.deepcopy(items)
                actual=self.detector._measure_piece(depth,actual_items,args[0],crop_box=args[2],circle=args[3])
                with self.subTest(case=case,workers=workers):
                    self.assertEqual(actual,expected)
                    for actual_item,expected_item,raw in zip(actual_items,originals,items):
                        self.assertEqual(self.detector._public_instance(actual_item),
                                         self.reference._public_instance(expected_item))
                        np.testing.assert_array_equal(actual_item['_mask_2d_array'],raw['_mask_2d_array'])
                        np.testing.assert_array_equal(actual_item['_mask_3d_array'],raw['_mask_3d_array'])

    def test_solder_workers_overlap_read_prepared_cache_and_parent_writes_last(self):
        depth,items=self.fixture()
        barrier=threading.Barrier(4)
        parent=threading.current_thread().ident
        threads=[]
        original=self.detector._measure_solder
        def measure(*args):
            self.assertNotEqual(threading.current_thread().ident,parent)
            for item in items:
                self.assertNotIn('height_3d',item)
                self.assertNotIn('bond_location',item)
                if item['class_name']=='chip':
                    self.assertIn('_filled_chip_region_2d',item)
            barrier.wait(timeout=3)
            threads.append(threading.current_thread().name)
            return original(*args)
        totals={}
        with patch.object(self.detector,'_measure_solder',side_effect=measure), \
             patch.object(board,'_filled_external_region',wraps=board._filled_external_region) as filled, \
             timing_context(totals,phase='joint',piece_id=1):
            result=self.detector._measure_piece(depth,items,dict(scale=1,tx=0,ty=0,is_valid=True),
                crop_box=(0,0,240,180),circle=(30,70,10))
        self.assertEqual(filled.call_count,2)
        self.assertEqual(len(set(threads)),4)
        self.assertEqual([item['measurement_id'] for item in result['solder_measurements']],[4,5,6,18])
        self.assertTrue(all('height_3d' in item for item in items))
        for key in ('height_prepare_chip_regions_ms','height_solder_wall_ms','height_solder_work_ms','height_solder_ms'):
            self.assertGreaterEqual(totals[key],0)

    def test_chip_matching_is_read_only_with_or_without_prepared_cache(self):
        _,items=self.fixture()
        chips=[item for item in items if item['class_name']=='chip']
        point=np.zeros((180,240),np.uint8);point[85,155]=255
        for prepare in (False,True):
            current=copy.deepcopy(chips)
            if prepare:self.detector._prepare_chip_regions(current)
            readonly=[MappingProxyType(chip) for chip in current]
            with ThreadPoolExecutor(max_workers=4) as pool:
                results=list(pool.map(lambda _:self.detector._matching_chip({'_mask_2d_array':point},readonly),range(8)))
            self.assertTrue(all(chip['instance_id']==4 and inside for chip,inside in results))
            if not prepare:self.assertTrue(all('_filled_chip_region_2d' not in chip for chip in current))

    def test_concurrent_pieces_share_leaf_pool_without_cache_or_output_leaks(self):
        fixtures=[]
        expected=[]
        for pid in range(2):
            depth,items=self.fixture()
            depth=depth+pid*50
            args=(dict(scale=1,tx=0,ty=0,is_valid=True),[(0,0,240,180)],(0,0,240,180),(30,70,10))
            expected.append(self.reference._measure_piece(depth,copy.deepcopy(items),*args))
            for item in items:
                item['_mask_2d_array'].flags.writeable=False
                item['_mask_3d_array'].flags.writeable=False
            fixtures.append((depth,items,args))
        def piece(pid):
            depth,items,args=fixtures[pid]
            totals={}
            with timing_context(totals,phase='joint',piece_id=pid+1):
                result=self.detector._measure_piece(depth,items,args[0],crop_box=args[2],circle=args[3])
            return result,totals
        with ThreadPoolExecutor(max_workers=2) as parents:
            futures=[parents.submit(piece,pid) for pid in range(2)]
            results=[future.result(timeout=10) for future in futures]
        for (result,totals),reference in zip(results,expected):
            self.assertEqual(result,reference)
            self.assertIn('height_solder_work_ms',totals)
        self.assertIsNot(fixtures[0][1][2]['_filled_chip_region_2d'],fixtures[1][1][2]['_filled_chip_region_2d'])

    def test_failed_solder_tasks_drain_before_return_and_skip_parent_updates(self):
        depth,items=self.fixture()
        barrier=threading.Barrier(4)
        finished=[]
        def task(job,*args):
            barrier.wait(timeout=3)
            finished.append((job[0]['instance_id'],job[1]))
            if job[1]=='start':raise ValueError('synthetic solder failure')
            return {'source_instance_id':job[0]['instance_id'],'endpoint':job[1]}
        with patch.object(self.detector,'_measure_solder_task',side_effect=task):
            with self.assertRaisesRegex(ValueError,'synthetic solder failure'):
                self.detector._measure_piece(depth,items,dict(scale=1,tx=0,ty=0,is_valid=True),
                    crop_box=(0,0,240,180),circle=(30,70,10))
        self.assertEqual(len(finished),4)
        self.assertTrue(all('height_3d' not in item for item in items))


if __name__ == '__main__':
    unittest.main()
