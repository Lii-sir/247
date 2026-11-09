"""Synthetic raw-depth transport checks; no camera, model or performance tests."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from importlib import import_module
import mmap
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'tests'), str(ROOT/'visualization')]
from transport.depth_input import validate_depth_input, depth_from_uint16
from algorithm.detect_board import DetectBoard
from algorithm.board_pipeline import BoardPipeline
from algorithm.timing import timing_context
from pipeline_fakes import FakeDetector
from detect_and_view import mapped_file


class RawDepthInputTests(unittest.TestCase):
    def shared(self, depth, stack):
        raw = depth.astype('<u2').tobytes()
        name = 'raw-depth-test-' + uuid.uuid4().hex
        mapping = mmap.mmap(-1, len(raw), tagname=name, access=mmap.ACCESS_WRITE)
        stack.callback(mapping.close)
        mapping.write(raw)
        return dict(shm_name=name, data_size=len(raw), data_format='uint16',
                    image_width=depth.shape[1], image_height=depth.shape[0])

    def test_known_little_endian_pixels_keep_full_uint16_range_without_codec(self):
        expected = np.array([[0, 1, 256], [32768, 65534, 65535]], np.uint16)
        with ExitStack() as stack:
            descriptor = self.shared(expected, stack)
            timings = {}
            with patch.object(cv2, 'imdecode', side_effect=AssertionError('codec called')), \
                    timing_context(timings, phase='depth'):
                actual = DetectBoard._read_3d(descriptor)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(np.uint16, actual.dtype)
        self.assertFalse(actual.flags.writeable)
        self.assertIn('depth_array_view_ms', timings)
        self.assertNotIn('depth_image_decode_ms', timings)
        # The bytes read from shared memory own the array after the mapping closes.
        self.assertEqual(65535, int(actual[1, 2]))

    def test_invalid_layout_is_rejected_before_reading_shared_memory(self):
        base = dict(shm_name='unused', data_size=12, image_width=3, image_height=2)
        cases = [dict(data_format=value) for value in ('png', 'file', 'uint16_be', None, [])]
        cases += [dict(image_width=value) for value in (0, -1, True, 3.0, '3', None)]
        cases += [dict(image_height=0), dict(data_size=11), dict(data_size=14),
                  dict(shm_name=''), dict(image_stride=8)]
        with patch('algorithm.detect_board.read_shared_memory', side_effect=AssertionError('read before validation')):
            for change in cases:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    DetectBoard._read_3d({**base, **change})

    def test_view_does_not_allocate_or_modify_pixel_storage_and_checks_length(self):
        raw = b'\x00\x00\x01\x00\x00\x01\xff\xff'
        actual = depth_from_uint16(raw, 2, 2)
        self.assertIs(actual.base.base, raw)
        self.assertEqual([[0, 1], [256, 65535]], actual.tolist())
        for size in (0, 7, 9):
            with self.assertRaises(ValueError):
                depth_from_uint16(bytes(size), 2, 2)

    def test_standalone_3d_uses_raw_depth_and_reports_view_timing(self):
        module = import_module('algorithm.detect_3D')
        segment = import_module('algorithm.3D.segment')
        depth = np.zeros((20, 3), np.uint16)
        depth[:5] = depth[10:15] = 33000
        detector = module.Detect3D()
        self.addCleanup(detector.release)
        detector.segmentation_config = segment.SegmentationConfig(expected_piece_count=2, max_internal_gap_height=0)
        with ExitStack() as stack:
            request = self.shared(depth, stack)
            with patch.object(module, 'measure_depth', side_effect=lambda image, pieces:
                    [{**item, 'is_ok': True} for item in pieces]):
                result = detector.handle_detect(request)
        self.assertTrue(result['ok'], result)
        self.assertEqual(2, len(result['results_3d']['pieces']))
        self.assertIn('depth_array_view_ms', result['timings'])
        self.assertNotIn('decode_ms', result['timings'])
        self.assertFalse(detector.handle_detect({**request, 'data_format': 'png'})['ok'])

    def test_formal_streaming_pipeline_accepts_raw_depth_then_joins_pieces(self):
        class Detector(FakeDetector):
            _read_3d = staticmethod(DetectBoard._read_3d)
        depth = np.zeros((20, 3), np.uint16)
        depth[:5] = depth[10:15] = 33000
        with ExitStack() as stack, ThreadPoolExecutor(2) as cpu, ThreadPoolExecutor(1) as gpu:
            descriptor = self.shared(depth, stack)
            pipe = BoardPipeline(Detector(count=2), cpu, gpu)
            try:
                board = pipe.submit_3d(dict(board_id='raw', **descriptor))
                for pid in (1, 2):
                    self.assertTrue(pipe.submit_2d(dict(board_id='raw', piece_id=pid)).result(3)['ok'])
                result = board.result(3)
                self.assertTrue(result['ok'], result)
                self.assertEqual(2, len(result['results_board']['pieces']))
                self.assertIn('depth_read_ms', result['timings'])
                self.assertNotIn('depth_image_decode_ms', result['timings'])
            finally:
                pipe.close()

    def test_offline_sample_adapter_sends_packed_pixels_and_rejects_display_images(self):
        expected = np.array([[0, 1, 256], [32768, 65534, 65535]], np.uint16)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            path = Path(folder)/'odd-width.png'
            cv2.imwrite(str(path), expected)
            descriptor = mapped_file(stack, path, '3d')
            self.assertEqual('uint16', descriptor['data_format'])
            self.assertEqual(12, descriptor['data_size'])
            np.testing.assert_array_equal(DetectBoard._read_3d(descriptor), expected)
            for image in (np.zeros((2, 3), np.uint8), np.zeros((2, 3, 3), np.uint16)):
                cv2.imwrite(str(path), image)
                with self.assertRaises(ValueError):
                    mapped_file(stack, path, '3d')

    def test_offline_pitch_prepares_pixels_before_algorithm_reads_shared_input(self):
        from algorithm.depth_preprocessing import prepare_depth_for_detection
        original=np.zeros((50,3),np.uint16);original[20:35]=32958
        settings=dict(model='smartray_pitch',enabled_3d=True,angle_deg=30,scan_pitch_mm=.01)
        expected,info=prepare_depth_for_detection(original,settings)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            path=Path(folder)/'raw.png';cv2.imwrite(str(path),original)
            descriptor=mapped_file(stack,path,'3d',pitch_settings=settings)
            actual=DetectBoard._read_3d(descriptor)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(info,descriptor['pitch_correction'])
            np.testing.assert_array_equal(cv2.imread(str(path),cv2.IMREAD_UNCHANGED),original)


if __name__ == '__main__':
    unittest.main()
