"""Synthetic sharing/lifetime checks: no camera, models or CPU affinity changes."""
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'tests'))
from algorithm.registration_process import SharedBoardInputs, JointSharedInputs
from algorithm.cpu_2d_process import Cpu2DProcesses
from algorithm.joint_process import JointMeasurementProcesses
from algorithm.board_pipeline import BoardPipeline
from pipeline_fakes import FakeDetector
from test_joint_process import fixture

share_instances = import_module('algorithm.2D.instance_geometry').share_instances


class SharedInstanceInputsTests(unittest.TestCase):
    def owners(self):
        board, masks = SharedBoardInputs(), SharedBoardInputs()
        self.addCleanup(board.close)
        self.addCleanup(masks.close)
        return board, masks

    def test_both_submission_paths_publish_same_mask_names_in_either_order(self):
        depth, entry, config, registration = fixture()
        for joint_first in (False, True):
            with self.subTest(joint_first=joint_first):
                board, masks = self.owners()
                cpu = Cpu2DProcesses.__new__(Cpu2DProcesses)
                joint = JointMeasurementProcesses.__new__(JointMeasurementProcesses)
                # Inspect the real payload builders without starting processes.
                def capture(pool, inputs, payload, *args, **kwargs):
                    inputs.acquire()
                    try:
                        return payload()
                    finally:
                        inputs.release()
                cpu.submit_task = joint.submit_task = capture
                prepared = dict(joint_entry=entry, silver_boundary=object(),
                    crop=np.zeros((180, 240, 3), np.uint8), crop_box=entry['crop_box'], circle=None)
                metadata = dict(board_id='A', piece_id=1)
                def cpu_payload():
                    return cpu.submit_piece(None, prepared, {}, None, {}, metadata, lambda: True,
                                            shared_inputs=masks)
                def joint_payload():
                    return joint.submit_piece(None, JointSharedInputs(board, masks), 1, entry, depth,
                        dict(piece_id=1, y1=11, y2=191), registration, {}, config, 1,
                        (0, 0, 240, 180), [], {}, metadata, lambda: True)
                if joint_first:
                    b, a = joint_payload(), cpu_payload()
                else:
                    a, b = cpu_payload(), joint_payload()
                self.assertEqual(a['instances'], b['instances'])
                self.assertEqual(a['arrays'], {k: v for k, v in b['arrays'].items() if k != 'depth'})
                self.assertEqual(len(entry['instances']), len(masks._blocks))
                self.assertEqual(1, len(board._blocks))
                descriptor = a['arrays']['mask_0']
                handle = SharedMemory(name=descriptor['name'])
                try:
                    actual = np.ndarray(descriptor['shape'], dtype=descriptor['dtype'], buffer=handle.buf)
                    np.testing.assert_array_equal(actual, entry['instances'][0]['mask'])
                    del actual
                finally:
                    handle.close()

    def test_concurrent_publication_creates_one_block_and_leases_outlive_close(self):
        board, masks = self.owners()
        instances = [dict(mask=np.ones((7, 9), np.uint8))]
        joint = JointSharedInputs(board, masks)
        masks.acquire()  # 2D reader
        joint.acquire()  # 3D reader
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(share_instances, instances, owner, cache_prefix=('instance_mask', 1))
                    for owner in (masks, joint)]
            first, second = [job.result(3) for job in jobs]
        self.assertEqual(first, second)
        self.assertEqual(1, len(masks._blocks))
        name = first[0]['mask_0']['name']
        board.close()
        masks.close()
        masks.release()
        handle = SharedMemory(name=name)
        handle.close()
        joint.release()
        with self.assertRaises(FileNotFoundError):
            SharedMemory(name=name)
        self.assertEqual({}, masks._blocks)
        self.assertEqual(0, board._leases)

    def test_partial_acquire_rolls_back_depth_lease(self):
        board, masks = self.owners()
        masks.close()
        with self.assertRaises(RuntimeError):
            JointSharedInputs(board, masks).acquire()
        self.assertEqual(0, board._leases)

    def test_piece_replacement_uses_new_owner_and_board_release_closes_it(self):
        with ThreadPoolExecutor(2) as cpu, ThreadPoolExecutor(1) as gpu:
            pipe = BoardPipeline(FakeDetector(count=1, delay=0), cpu, gpu)
            self.addCleanup(pipe.close)
            self.assertTrue(pipe.submit_2d(dict(board_id='A', piece_id=1)).result(3)['ok'])
            old = pipe._boards['A'].mask_inputs[1]
            old.prepare_named(dict(mask=np.ones((3, 4), np.uint8)))
            self.assertTrue(pipe.submit_2d(dict(board_id='A', piece_id=1)).result(3)['ok'])
            new = pipe._boards['A'].mask_inputs[1]
            self.assertIsNot(old, new)
            self.assertTrue(old._closed)
            self.assertEqual({}, old._blocks)
            pipe.release_board('A')
            self.assertTrue(new._closed)


if __name__ == '__main__':
    unittest.main()
