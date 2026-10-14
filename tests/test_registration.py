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
