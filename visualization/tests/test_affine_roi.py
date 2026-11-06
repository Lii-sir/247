"""ROI scoring versus the actual full OpenCV warp, using synthetic masks."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from algorithm import registration as reg


class AffineRoiTests(unittest.TestCase):
    def test_roi_pixels_equal_full_warp_for_native_and_ipp_paths(self):
        rng=np.random.default_rng(812)
        source=rng.integers(0,2,(90,130),dtype=np.uint8)
        rois=[(0,0,27,29),(17,13,117,97),(110,70,150,110)]
        grids=[reg._affine_roi_grid(box) for box in rois]
        matrices=[np.array([[1,0,tx],[0,1,ty]],float) for tx,ty in [(0,0),(.5,.5),(-.5,-.5),(10000,0)]]
        matrices += [np.array([[rng.uniform(.4,1.2),rng.uniform(-.15,.15),rng.uniform(-30,30)],
                              [rng.uniform(-.15,.15),rng.uniform(.4,1.2),rng.uniform(-30,30)]]) for _ in range(100)]
        original=cv2.ipp.useIPP()
        try:
            for ipp in (False,original):
                cv2.ipp.setUseIPP(ipp)
                for index,matrix in enumerate(matrices):
                    expected=cv2.warpAffine(source,matrix,(150,110),flags=cv2.INTER_NEAREST)
                    patches=list(reg._warp_affine_rois(source,matrix,rois,grids,150))
                    for box,actual in zip(rois,patches):
                        x0,y0,x1,y1=box
                        with self.subTest(ipp=ipp,index=index,box=box):
                            np.testing.assert_array_equal(actual,expected[y0:y1,x0:x1])
        finally:cv2.ipp.setUseIPP(original)

    def fixture(self):
        image=np.zeros((80,120),np.uint8);image[8:65,15:100]=200;image[20:38,40:60]=0
        depth=cv2.warpAffine((image>50).astype(np.uint16)*33000,np.array([[.8,.01,3.],[-.01,.81,2.]]),(110,75))
        coarse=dict(scale=.8,tx=2.,ty=3.)
        parameters=dict(preview_scale=1.,morphology_kernel_size=1,rois=[[0,0,35,70],[45,5,105,74]],maxiter=2)
        return image,depth,coarse,parameters

    def test_optimizer_scores_matrices_and_stop_diagnostics(self):
        image,depth,coarse,parameters=self.fixture()
        results=[reg.refine_affine_registration(image,depth,coarse,{**parameters,'score_mode':mode}) for mode in ('full','roi')]
        for field in ('affine_score_before','affine_score_after','affine_zone_losses','affine_matrix_2d_to_3d','affine_function_evaluations','affine_stop_reason'):
            self.assertEqual(results[0][field],results[1][field],field)
        for result in results:
            self.assertGreater(result['affine_function_evaluations'],0)
            self.assertEqual(result['affine_score_calls'],result['affine_function_evaluations']+2)
            self.assertIsInstance(result['affine_optimizer_success'],bool)
            self.assertIsInstance(result['affine_optimizer_status'],int)
            self.assertGreaterEqual(result['affine_score_mean_ms'],0.)

    def test_roi_mode_never_allocates_full_warp_and_invalid_mode_is_rejected(self):
        image,depth,coarse,parameters=self.fixture()
        def optimize(fn,x0,**kwargs):
            fn(x0)
            return SimpleNamespace(x=x0,nit=0,nfev=1,success=False,status=2,message='iteration limit')
        with patch.object(reg,'minimize',side_effect=optimize),patch.object(reg.cv2,'warpAffine',side_effect=AssertionError('full warp forbidden')):
            result=reg.refine_affine_registration(image,depth,coarse,parameters)
        self.assertEqual(result['affine_stop_reason'],'iteration limit')
        self.assertEqual(result['affine_function_evaluations'],1)
        invalid=reg.refine_affine_registration(image,depth,coarse,{**parameters,'score_mode':'bad'})
        self.assertIn('score_mode',invalid['affine_error'])

    def test_optimizer_exception_preserves_diagnostics(self):
        image,depth,coarse,parameters=self.fixture()
        def fail(fn,x0,**kwargs):fn(x0);raise ValueError('synthetic stop')
        with patch.object(reg,'minimize',side_effect=fail):
            result=reg.refine_affine_registration(image,depth,coarse,parameters)
        self.assertFalse(result['affine_optimizer_success'])
        self.assertEqual(result['affine_stop_reason'],'synthetic stop')
        self.assertEqual(result['affine_function_evaluations'],1)
        self.assertEqual(result['affine_score_calls'],2)


if __name__=='__main__':unittest.main()
