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

    def calibration(self,args,linear=None):
        image,depth,coarse,values=args
        context=reg.affine_coordinate_context(image.shape,depth.shape)
        matrix=np.eye(3)
        matrix[:2,:2]=linear if linear is not None else [[1.005,0],[0,1.]]
        matrix[:2,2]=[90000.,-70000.] # Deliberately irrelevant old translation.
        return dict(format='registration_lab_matrix',version=1,matrix=matrix.tolist(),context=copy.deepcopy(context)),context

    def test_calibration_reuses_only_linear_and_preserves_coarse_anchor_and_bounds(self):
        args=self.fixture();image,_,coarse,values=args
        profile,context=self.calibration(args)
        p0=reg._affine_parameters_from_matrix(reg._registration_matrix(coarse),1.)
        candidate=p0.copy();candidate[0]=1.005
        depth=cv2.warpAffine((image>50).astype(np.uint16)*33000,reg._full_affine(candidate,1.),(100,70),flags=cv2.INTER_NEAREST)
        seen=[]
        def optimize(fn,start,**kwargs):
            seen.append((start.copy(),kwargs['bounds']))
            fn(start)
            return SimpleNamespace(x=start,nit=1,nfev=1,success=True,status=0,message='done')
        saved=copy.deepcopy(profile)
        with patch.object(reg,'minimize',side_effect=optimize):
            result=reg.refine_affine_registration(image,depth,coarse,
                {**values,'penalty_weight':0.,'calibration':{'enabled':True,'profile':profile}},context=context)
        np.testing.assert_array_equal(seen[0][0],candidate)
        self.assertEqual(seen[0][1][4],(965.,1035.))
        self.assertEqual(result['affine_initialization']['source'],'calibration_linear')
        self.assertLess(result['affine_initial_score'],result['affine_score_before'])
        self.assertEqual(profile,saved)

    def test_calibration_context_failures_retain_coarse_seed(self):
        args=self.fixture();profile,context=self.calibration(args)
        invalid=[]
        for key,value in [('source_space','other'),('image_size',[101,70]),('depth_size',[101,70]),('preprocessing',{})]:
            item=copy.deepcopy(profile);item['context'][key]=value;invalid.append(item)
        item=copy.deepcopy(profile);item['matrix'][2][0]=.01;invalid.append(item)
        item=copy.deepcopy(profile);item['matrix'][0][0]=2.;invalid.append(item)
        for item in invalid+[None]:
            with self.subTest(profile=item):
                result=reg.refine_affine_registration(*args[:3],
                    {**args[3],'calibration':{'enabled':True,'profile':item}},context=context)
                self.assertEqual(result['affine_initialization']['source'],'coarse')
                self.assertFalse(result['affine_initialization']['profile_valid'])

    def test_calibration_allows_piece_height_changes_but_needs_context_and_lower_loss(self):
        args=self.fixture();profile,context=self.calibration(args,[[1,0],[0,1]])
        profile['context']['depth_size'][1]+=11
        profile['context']['segmentation']={'mode':'auto'}
        context['segmentation']={'mode':'fixed'}
        values={**args[3],'calibration':{'enabled':True,'profile':profile}}
        accepted=reg.refine_affine_registration(*args[:3],values,context=context)
        self.assertTrue(accepted['affine_initialization']['profile_valid'])
        self.assertEqual(accepted['affine_initialization']['source'],'coarse')
        missing=reg.refine_affine_registration(*args[:3],values)
        self.assertIn('context_missing',missing['affine_initialization']['reason'])

    def test_multiscale_matrix_uses_actual_source_and_target_ratios(self):
        source=np.ones((73,101),np.uint8);target=np.ones((55,93),np.uint8)
        full=reg._AffineScore(source,target,[(0,0,93,55)],0.,.02,cache_enabled=True)
        low=full.resized(.5)
        p=np.array([.8,.9,.001,-.002,500.,300.])
        with patch.object(reg.cv2,'warpAffine',wraps=cv2.warpAffine) as warp:
            low(p)
        matrix=reg._scaled_affine(p)
        src=np.diag([low.source.shape[1]/101,low.source.shape[0]/73,1.])
        dst=np.diag([low.target.shape[1]/93,low.target.shape[0]/55,1.])
        expected=dst @ np.vstack([matrix,[0,0,1]]) @ np.linalg.inv(src)
        np.testing.assert_allclose(warp.call_args.args[1],expected[:2],rtol=0,atol=1e-12)

    def test_multiscale_caches_do_not_share_losses(self):
        source=np.zeros((8,8),np.uint8);target=source.copy();target[1,1]=1
        full=reg._AffineScore(source,target,[(0,0,8,8)],0.,.02,cache_enabled=True)
        low=full.resized(.5)
        p=np.array([1,1,0,0,1000,400.])
        self.assertNotEqual(full(p),low(p))
        self.assertEqual(full.misses,1);self.assertEqual(low.misses,1)
        self.assertIsNot(full.cache,low.cache)

    def test_multiscale_keeps_canonical_parameters_and_sums_stage_calls(self):
        args=self.fixture();seen=[]
        def optimize(fn,x0,**kwargs):
            seen.append((x0.copy(),kwargs['bounds']))
            fn(x0)
            return SimpleNamespace(x=x0.copy(),nit=1,nfev=1,success=True,status=0,message='done')
        with patch.object(reg,'minimize',side_effect=optimize):
            result=reg.refine_affine_registration(*args[:3],{**args[3],'cache_enabled':True,'multiscale':{'enabled':True}})
        self.assertEqual([stage['name'] for stage in result['affine_stages']],['search','refine'])
        self.assertEqual(result['affine_function_evaluations'],2)
        np.testing.assert_array_equal(seen[0][0],seen[1][0])
        self.assertEqual(seen[0][1][4],(965.,1035.))
        self.assertTrue(result['affine_multiscale']['applied'])
        self.assertEqual(result['affine_score_mode'],'full')

    def test_refine_boundary_triggers_original_full_search(self):
        args=list(self.fixture());args[0][:]=0;args[1][:]=0
        seen=[]
        def optimize(fn,x0,**kwargs):
            seen.append(kwargs['bounds'])
            x=np.array([b[0] for b in kwargs['bounds']]) if len(seen)==2 else x0.copy()
            fn(x)
            return SimpleNamespace(x=x,nit=1,nfev=1,success=True,status=0,message='done')
        with patch.object(reg,'minimize',side_effect=optimize):
            result=reg.refine_affine_registration(*args[:3],{**args[3],'penalty_weight':0.,'multiscale':{'enabled':True}})
        self.assertEqual([stage['name'] for stage in result['affine_stages']],['search','refine','fallback'])
        self.assertEqual(seen[0],seen[2])
        self.assertEqual(result['affine_multiscale']['fallback_reason'],'refine_window_boundary')
        self.assertEqual(result['affine_function_evaluations'],3)

    def test_multiscale_error_falls_back_and_final_score_never_uses_low_loss(self):
        args=self.fixture();calls=[]
        def optimize(fn,x0,**kwargs):
            calls.append(1)
            if len(calls)==1:raise ValueError('low stage failed')
            fn(x0)
            return SimpleNamespace(x=x0,nit=1,nfev=1,success=True,status=0,message='done')
        with patch.object(reg,'minimize',side_effect=optimize):
            result=reg.refine_affine_registration(*args[:3],{**args[3],'multiscale':{'enabled':True}})
        self.assertTrue(result['affine_multiscale']['fallback_used'])
        self.assertEqual(result['affine_multiscale']['fallback_reason'],'search_error')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['affine_score_after'],0.)

    def test_real_synthetic_multiscale_preserves_best_full_seed(self):
        args=self.fixture()
        args[1][:]=cv2.warpAffine(args[1],np.array([[1,0,2.],[0,1,0.]]),(100,70),flags=cv2.INTER_NEAREST)
        result=reg.refine_affine_registration(*args[:3],{**args[3],'cache_enabled':True,
            'multiscale':{'enabled':True,'search_maxiter':2,'refine_maxiter':2}})
        self.assertLessEqual(result['affine_score_after'],result['affine_initial_score'])
        self.assertTrue(np.isfinite(result['affine_matrix_2d_to_3d']).all())

    def test_budget_uses_best_trial_instead_of_last_completed_powell_position(self):
        args=list(self.fixture());args[1][:]=cv2.warpAffine(args[1],np.array([[1,0,2.],[0,1,0.]]),(100,70),flags=cv2.INTER_NEAREST)
        def optimize(fn,x0,**kwargs):
            better=x0.copy();better[4]+=2
            fn(x0);fn(better)
            return SimpleNamespace(x=x0.copy(),nit=0,nfev=2,success=False,status=1,message='maxfev reached')
        with patch.object(reg,'minimize',side_effect=optimize):
            result=reg.refine_affine_registration(*args[:3],{**args[3],'cache_enabled':True,'maxfev':2})
        self.assertEqual(result['affine_function_evaluations'],2)
        self.assertEqual(result['affine_score_after'],0.)
        self.assertEqual(result['tx'],2.)
        self.assertTrue(result['affine_stages'][0]['used_best_observed'])
        self.assertFalse(result['affine_optimizer_success'])
        self.assertTrue(result['affine_valid'])

    def test_total_budget_caps_all_stages_and_reports_blocked_fallback(self):
        args=list(self.fixture());args[0][:]=0;args[1][:]=0
        limits=[]
        def optimize(fn,x0,**kwargs):
            limit=kwargs['options']['maxfev'];limits.append(limit)
            point=x0.copy() if len(limits)==1 else np.array([b[0] for b in kwargs['bounds']])
            for _ in range(limit):fn(point)
            return SimpleNamespace(x=point,nit=0,nfev=limit,success=False,status=1,message='maxfev reached')
        with patch.object(reg,'minimize',side_effect=optimize):
            result=reg.refine_affine_registration(*args[:3],{**args[3],'penalty_weight':0.,'total_maxfev':3,
                'multiscale':{'enabled':True,'search_maxfev':2,'refine_maxfev':5}})
        self.assertEqual(limits,[2,1])
        self.assertEqual(result['affine_function_evaluations'],3)
        self.assertEqual(result['affine_budget']['used'],3)
        self.assertTrue(result['affine_budget']['exhausted'])
        self.assertFalse(result['affine_multiscale']['fallback_used'])
        self.assertEqual(result['affine_multiscale']['fallback_not_run'],'total_evaluation_budget_exhausted')
        self.assertTrue(result['affine_stages'][-1]['skipped'])

    def test_real_synthetic_maxfev_is_independent_of_quality(self):
        args=self.fixture()
        result=reg.refine_affine_registration(*args[:3],{**args[3],'cache_enabled':True,'maxfev':6})
        self.assertEqual(result['affine_function_evaluations'],6)
        self.assertEqual(result['affine_optimizer_status'],1)
        self.assertFalse(result['affine_optimizer_success'])
        self.assertTrue(result['affine_valid'])
        self.assertEqual(result['affine_score_after'],0.)

    def test_invalid_budgets_are_rejected_and_errors_do_not_keep_coarse_validity(self):
        args=self.fixture()
        for field in ('maxfev','total_maxfev'):
            for value in (0,-1,True,1.5,'150'):
                result=reg.refine_affine_registration(*args[:3],{**args[3],field:value})
                self.assertIn(field,result['affine_error'])
                self.assertFalse(result['is_valid'])
        with patch.object(reg,'minimize',side_effect=ValueError('all stages failed')):
            result=reg.refine_affine_registration(*args[:3],{**args[3],'multiscale':{'enabled':True}})
        self.assertFalse(result['is_valid'])
        self.assertIn('failed',result['affine_error'])

    def test_saved_report_comparison_supports_formal_envelope_and_lab_export(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('affine_comparison',ROOT/'visualization/compare_affine_results.py')
        comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
        record=dict(affine_score_after=.1,affine_function_evaluations=10,matrix_2d_to_3d=[[1,0,0],[0,1,0]])
        first={'response':{'results_board':{'pieces':[{'piece_id':1,'registration':record,'timings':{'affine_registration_ms':100.}}]}}}
        second={'inputs':{'piece_id':1},'stages':[{'registration':record}],'timings':{'仿射配准':50.}}
        result=comparison.compare(first,second)['pieces'][0]
        self.assertEqual(result['wall_reduction_fraction'],.5)
        self.assertTrue(result['matrix_identical'])
        second['inputs']['piece_id']=2
        with self.assertRaises(ValueError):comparison.compare(first,second)

    def test_default_options_and_explicit_disabled_options_are_equivalent(self):
        args=self.fixture()
        first=reg.refine_affine_registration(*args[:3],args[3])
        second=reg.refine_affine_registration(*args[:3],{**args[3],'cache_enabled':True,'maxfev':None,
            'total_maxfev':None,'calibration':{'enabled':False,'profile':None},'multiscale':{'enabled':False}})
        for key in ('affine_matrix_2d_to_3d','affine_score_after','affine_function_evaluations'):
            self.assertEqual(first[key],second[key],key)


if __name__=='__main__':unittest.main()
