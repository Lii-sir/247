"""Synthetic affine search regressions: no models, real inputs or benchmarks."""
from pathlib import Path
import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from algorithm import registration as reg


class AffineOptimizationTests(unittest.TestCase):
    def fixture(self):
        image=np.zeros((70,100),np.uint8);image[10:60,20:80]=200;image[20:30,30:40]=0
        depth=(image>50).astype(np.uint16)*33000
        return image,depth,dict(scale=1.,tx=0.,ty=0.),dict(preview_scale=1.,morphology_kernel_size=1,
            rois=[[0,0,45,70],[45,0,100,70]],maxiter=2,ftol=.01)

    def test_cache_reuses_exact_parameters_preserves_scores_and_nfev(self):
        image,depth,coarse,values=self.fixture();evaluated=[]
        def optimize(fn,x0,**kwargs):
            p=x0.copy();p[4]+=1
            scores=[fn(x0),fn(x0.copy()),fn(p),fn(x0)]
            evaluated.append(scores)
            return SimpleNamespace(x=p,nit=1,nfev=4,success=True,status=0,message='done')
        results=[]
        for enabled in (False,True):
            with patch.object(reg,'minimize',side_effect=optimize):
                results.append(reg.refine_affine_registration(image,depth,coarse,{**values,'cache_enabled':enabled}))
        self.assertEqual(evaluated[0],evaluated[1])
        self.assertEqual(results[0]['affine_matrix_2d_to_3d'],results[1]['affine_matrix_2d_to_3d'])
        self.assertEqual(results[1]['affine_function_evaluations'],4)
        self.assertEqual(results[1]['affine_score_calls'],6)
        self.assertEqual(results[1]['affine_cache_hits'],4)
        self.assertEqual(results[1]['affine_cache_misses'],2)
        self.assertEqual(results[1]['affine_warp_evaluations'],2)
        self.assertEqual(results[0]['affine_warp_evaluations'],6)

    def test_cache_is_per_image_and_keeps_distinct_nearby_parameter_keys(self):
        source=np.zeros((10,12),np.uint8);target=source.copy()
        p=np.array([1,1,0,0,1000,400.],float)
        objective=reg._AffineScore(source,target,[(0,0,12,10)],.005,.02,cache_enabled=True)
        value,zones=objective(p,True);zones[0]=999
        self.assertEqual(objective(p,True)[1],[0.])
        q=p.copy();q[4]=np.nextafter(q[4],np.inf)
        objective(q)
        self.assertEqual(objective.misses,2)
        other=reg._AffineScore(source,np.ones_like(target),[(0,0,12,10)],.005,.02,cache_enabled=True)
        self.assertNotEqual(other(p),value)
        self.assertEqual(other.hits,0)

    def test_real_synthetic_powell_cache_does_not_change_solution(self):
        args=self.fixture();results=[reg.refine_affine_registration(*args[:3],{**args[3],'cache_enabled':v}) for v in (False,True)]
        for key in ('affine_function_evaluations','affine_score_before','affine_score_after','affine_zone_losses','affine_matrix_2d_to_3d'):
            self.assertEqual(results[0][key],results[1][key],key)
        self.assertGreater(results[1]['affine_cache_hits'],0)


if __name__=='__main__':unittest.main()
