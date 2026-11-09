"""Synthetic exact-parity, buffer ownership and persistent preprocessing checks."""
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(Path(__file__).resolve().parent)]
from algorithm.tilt_correction import correct_smartray_pitch, _depth_lookup
from algorithm.depth_preprocessing import DepthPreprocessor
from fixtures.pitch_reference import correct_smartray_pitch as reference


class PitchBufferTests(unittest.TestCase):
    def settings(self, **values):
        return dict(model='smartray_pitch',enabled_3d=True,input_state='uncorrected',
                    angle_deg=15.,scan_pitch_mm=.02,projection_workers=3,**values)

    def test_all_uint16_codes_match_frozen_formula_in_both_axes_and_directions(self):
        depth=np.arange(65536,dtype=np.uint16).reshape(256,256)
        for axis in ('rows','columns'):
            for direction in ('recover_ideal','document_forward'):
                for angle in (-15.,59.):
                    settings=self.settings(scan_axis=axis,transform_direction=direction)
                    settings['angle_deg']=angle
                    expected=reference(None,depth,settings)
                    actual=correct_smartray_pitch(None,depth,settings)
                    np.testing.assert_array_equal(actual[1],expected[1])
                    for key,value in expected[2].items():self.assertEqual(actual[2][key],value,key)

    def test_lookup_is_bounded_immutable_and_invalidated_by_calibration(self):
        _depth_lookup.cache_clear()
        first=_depth_lookup(15.,.02,.00105,32768.,0.,'recover_ideal')
        second=_depth_lookup(15.,.02,.00105,32768.,0.,'recover_ideal')
        self.assertIs(first,second)
        self.assertFalse(first[0].flags.writeable);self.assertFalse(first[1].flags.writeable)
        changed=_depth_lookup(-15.,.02,.00105,32768.,0.,'recover_ideal')
        self.assertIsNot(first,changed)
        self.assertFalse(np.array_equal(first[0],changed[0]))
        for angle in range(12):_depth_lookup(float(angle),.02,.00105,32768.,0.,'recover_ideal')
        self.assertLessEqual(_depth_lookup.cache_info().currsize,8)

    def test_reused_output_clears_stale_pixels_and_preserves_input(self):
        depth=np.full((60,17),33000,np.uint16)
        before=depth.copy();out=np.full_like(depth,65535)
        expected=reference(None,depth,self.settings())[1]
        result=correct_smartray_pitch(None,depth,self.settings(),out=out)
        self.assertIs(result[1],out)
        np.testing.assert_array_equal(out,expected);np.testing.assert_array_equal(depth,before)
        correct_smartray_pitch(None,np.zeros_like(depth),self.settings(),out=out)
        self.assertFalse(np.any(out))

    def test_noncontiguous_intensity_and_half_pixel_ties_match_reference(self):
        half_pitch=2*(.00105*math.tan(math.radians(6.)))
        depth=np.full((17,22),32769,np.uint16)[::-1,::2]
        image=np.arange(17*22*3,dtype=np.uint8).reshape(17,22,3)[::-1,::2]
        for axis in ('rows','columns'):
            settings=self.settings(scan_axis=axis,enabled_2d=True,image_role='smartray_intensity')
            settings.update(angle_deg=6.,scan_pitch_mm=half_pitch)
            expected=reference(image,depth,settings)
            actual=correct_smartray_pitch(image,depth,settings,out=np.empty(depth.shape,np.uint16))
            np.testing.assert_array_equal(actual[0],expected[0])
            np.testing.assert_array_equal(actual[1],expected[1])

    def test_bad_or_aliased_output_is_rejected(self):
        depth=np.ones((5,6),np.uint16)
        readonly=np.empty_like(depth);readonly.setflags(write=False)
        for out in (depth,depth.view(),readonly,np.empty((5,6),np.uint8),
                    np.empty((5,12),np.uint16)[:,::2],np.empty((5,5),np.uint16)):
            with self.assertRaises(ValueError):correct_smartray_pitch(None,depth,self.settings(),out=out)

    def test_runtime_reuses_pool_and_encoding_storage_but_returns_stable_bytes(self):
        depth=np.full((60,17),33000,np.uint16)
        settings=self.settings()
        processor=DepthPreprocessor(settings)
        settings['angle_deg']=float('nan')  # Constructor snapshot is independent.
        with processor:
            first,_=processor.encode(depth)
            buffer,pool=processor._buffer,processor._tasks._pool
            second,_=processor.encode(np.zeros_like(depth))
            self.assertIs(buffer,processor._buffer);self.assertIs(pool,processor._tasks._pool)
            self.assertFalse(np.any(np.frombuffer(second,np.uint16)))
            np.testing.assert_array_equal(np.frombuffer(first,np.uint16).reshape(depth.shape),
                                          reference(None,depth,self.settings())[1])
            processor.encode(np.ones((12,13),np.uint16))
            self.assertIsNot(buffer,processor._buffer)
        self.assertTrue(pool._shutdown)
        with self.assertRaises(RuntimeError):processor.encode(depth)

    def test_concurrent_runtime_calls_have_independent_snapshots_and_prepare_outputs(self):
        sources=[np.full((50,19),d,np.uint16) for d in (32900,33100)]
        with DepthPreprocessor(self.settings()) as processor, ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(processor.encode,depth) for depth in sources]
            for depth,future in zip(sources,futures):
                raw,_=future.result(5)
                np.testing.assert_array_equal(np.frombuffer(raw,np.uint16).reshape(depth.shape),
                                              reference(None,depth,self.settings())[1])
            a,_=processor.prepare(sources[0]);b,_=processor.prepare(sources[1])
            self.assertFalse(np.shares_memory(a,b))


if __name__=='__main__':unittest.main()
