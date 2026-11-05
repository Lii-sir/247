"""Registration score equivalence tests; no camera or GPU required."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from algorithm import registration


class AffineScoreTests(unittest.TestCase):
    def test_only_selected_regions_contribute_to_objective(self):
        gray=np.zeros((20,30),np.uint8)
        depth=np.zeros((20,30),np.uint16)
        depth[:10,10:20]=200  # Only R2 disagrees with the source mask.
        coarse=dict(scale=1.,tx=0.,ty=0.,homography_2d_to_3d=np.eye(3).tolist())
        base=dict(preview_scale=1.,morphology_kernel_size=1,penalty_weight=0.,
                  rois=[[0,0,10,10],[10,0,20,10],[20,0,30,10]])
        for selected,ids,expected in ((None,[1,2,3],1/3),([1],[1],0.),([2],[2],1.),([3,1],[1,3],0.)):
            values=dict(base)
            if selected is not None:values['selected_regions']=selected
            captured=[]
            def optimize(fn,x0,**kwargs):
                captured.append(fn(x0));return SimpleNamespace(x=x0,nit=0)
            with patch.object(registration,'minimize',side_effect=optimize):
                result=registration.refine_affine_registration(gray,depth,coarse,values)
            self.assertEqual(result['affine_scoring_region_ids'],ids)
            self.assertEqual(len(result['affine_zone_losses']),len(ids))
            self.assertAlmostEqual(result['affine_score_before'],expected)
            self.assertAlmostEqual(captured[0],expected)

    def test_invalid_selection_is_rejected_and_ids_survive_clipping(self):
        raw=[[0,0,10,10],[10,0,20,10],[20,0,30,10]]
        for selected in ([],[0],[4],[True],[1,1],'1'):
            with self.assertRaises(ValueError):registration._affine_regions(raw,30,20,selected)
        clipped=[[0,0,0,5],[-5,0,10,12],[20,0,40,12]]
        self.assertEqual(registration._affine_regions(clipped,30,20,[2]),[(2,(0,0,10,12))])

    def test_binary_mismatch_score_matches_int16_absolute_mean(self):
        image = np.zeros((40, 50), dtype=np.uint8)
        image[5:30, 8:42] = 200
        depth = np.zeros((20, 30), dtype=np.uint16)
        depth[2:18, 3:14] = 100
        depth[5:16, 16:27] = 100
        rois = [[2, 1, 13, 10], [14, 4, 29, 19]]
        coarse = {
            "scale": 1.0,
            "tx": 0.0,
            "ty": 0.0,
            "homography_2d_to_3d": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
        captured = {}

        def run_objective(objective, x0, **kwargs):
            captured["score"] = objective(x0)
            return SimpleNamespace(x=x0, nit=0)

        with patch.object(registration, "minimize", side_effect=run_objective):
            result = registration.refine_affine_registration(
                image, depth, coarse,
                {"preview_scale": 0.5, "rois": rois, "penalty_weight": 0.0},
            )

        source = cv2.resize((image > 50).astype(np.uint8), None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)
        warped = cv2.warpAffine(
            source, registration._scaled_affine(
                registration._affine_parameters_from_matrix(
                    registration._registration_matrix(coarse), 0.5
                )
            ), (30, 20), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        target = (depth > 0).astype(np.uint8)
        expected_losses = [
            float(np.abs(warped[y0:y1, x0:x1].astype(np.int16)
                         - target[y0:y1, x0:x1].astype(np.int16)).mean())
            for x0, y0, x1, y1 in rois
        ]
        self.assertEqual(result["affine_zone_losses"], expected_losses)
        self.assertAlmostEqual(captured["score"], float(np.mean(expected_losses)))
        self.assertAlmostEqual(result["affine_score_before"], float(np.mean(expected_losses)))


if __name__ == "__main__":
    unittest.main()
