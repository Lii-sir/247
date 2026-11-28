"""Allocation-path parity on synthetic data; no performance/real-image tests."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from algorithm.vision3d.alignment import _affine_native
from algorithm.vision3d.alignment import registration as reg


def legacy_score(score, parameters, details=False):
    """Previous allocating full-warp and patch != target scoring formula."""
    matrix = reg._scaled_affine(parameters)
    if score.ratios != (1., 1., 1., 1.):
        sx, sy, tx, ty = score.ratios
        matrix[:, 0] /= sx
        matrix[:, 1] /= sy
        matrix[0, :] *= tx
        matrix[1, :] *= ty
    warped = cv2.warpAffine(score.source, matrix, score.target.shape[::-1],
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    zones = tuple(float(np.count_nonzero(warped[y0:y1, x0:x1] != target) / target.size)
                  for (x0, y0, x1, y1), target in zip(score.rois, score.targets))
    penalty = score.penalty_weight * ((parameters[2]/max(score.shear_limit, 1e-6))**2
                                     +(parameters[3]/max(score.shear_limit, 1e-6))**2)
    value = float(np.mean(zones)+penalty)
    return (value, list(zones)) if details else value


class AffineBufferTests(unittest.TestCase):
    def scorer(self, **kwargs):
        rng = np.random.default_rng(81)
        source = rng.integers(0, 2, (73, 101), dtype=np.uint8)
        target = rng.integers(0, 2, (67, 97), dtype=np.uint8)
        return reg._AffineScore(source, target, [(0, 0, 31, 67), (25, 7, 97, 57)], .005, .02, **kwargs)

    def parameters(self):
        matrices = [np.array([[1, 0, x], [0, 1, y]], float)
                    for x, y in [(0, 0), (.5, .5), (-.5, -.5), (10000, -10000), (0, 0)]]
        rng = np.random.default_rng(62)
        matrices += [np.array([[rng.uniform(.8, 1.1), rng.uniform(-.02, .02), rng.uniform(-8, 8)],
                              [rng.uniform(-.02, .02), rng.uniform(.8, 1.1), rng.uniform(-8, 8)]])
                     for _ in range(30)]
        return [reg._affine_parameters_from_matrix(matrix, 1.) for matrix in matrices]

    def test_scores_match_allocating_reference_and_full_buffer_is_reused(self):
        for score in (self.scorer(), self.scorer().resized(.5)):
            warp = score._warp_buffer
            differences = list(score._difference_buffers)
            original_source, original_target = score.source.copy(), score.target.copy()
            for parameters in self.parameters():
                expected = legacy_score(score, parameters, True)
                with patch.object(reg.cv2, 'warpAffine', wraps=cv2.warpAffine) as call:
                    actual = score(parameters, True)
                self.assertEqual(expected, actual)
                self.assertIs(call.call_args.kwargs['dst'], warp)
                for box, target, difference in zip(score.rois, score.targets, differences):
                    x0, y0, x1, y1 = box
                    np.testing.assert_array_equal(difference, warp[y0:y1, x0:x1] != target)
            self.assertIs(score._warp_buffer, warp)
            self.assertTrue(all(a is b for a, b in zip(differences, score._difference_buffers)))
            np.testing.assert_array_equal(score.source, original_source)
            np.testing.assert_array_equal(score.target, original_target)

    def test_cache_keeps_scalar_values_not_mutable_buffer_views(self):
        score = self.scorer(cache_enabled=True)
        parameters = self.parameters()
        expected = score(parameters[0], True)
        score(parameters[1])
        actual = score(parameters[0], True)
        self.assertEqual(expected, actual)
        actual[1][0] = -999
        self.assertEqual(expected, score(parameters[0], True))
        self.assertEqual(2, score.warps)
        self.assertEqual(2, score.hits)

    def test_buffers_are_private_to_piece_and_resolution(self):
        a, b = self.scorer(), self.scorer()
        low = a.resized(.5)
        for left, right in ((a, b), (a, low)):
            self.assertFalse(np.shares_memory(left._warp_buffer, right._warp_buffer))
            self.assertFalse(np.shares_memory(left._difference_buffers[0], right._difference_buffers[0]))
        roi = self.scorer(score_mode='roi')
        self.assertIsNone(roi._warp_buffer)
        for parameters in self.parameters():
            self.assertEqual(legacy_score(a, parameters, True), roi(parameters, True))

    def test_synthetic_powell_solution_and_evaluation_count_match_legacy(self):
        image = np.zeros((70, 100), np.uint8)
        image[10:60, 20:80] = 200
        image[20:30, 30:40] = 0
        depth = (image > 50).astype(np.uint16)*33000
        values = dict(preview_scale=1., morphology_kernel_size=1,
            rois=[[0, 0, 45, 70], [45, 0, 100, 70]], maxiter=2, ftol=.01)
        args = image, depth, dict(scale=1., tx=0., ty=0.), values
        actual = reg.refine_affine_registration(*args)
        def allocating(score, parameters, details=False):
            score.calls += 1
            score.misses += 1
            score.warps += 1
            return legacy_score(score, parameters, details)
        with patch.object(reg._AffineScore, '__call__', allocating):
            expected = reg.refine_affine_registration(*args)
        for key in ('affine_function_evaluations', 'affine_score_before', 'affine_score_after',
                    'affine_zone_losses', 'affine_matrix_2d_to_3d', 'affine_valid', 'affine_stop_reason'):
            self.assertEqual(expected[key], actual[key], key)

    def test_built_rust_kernel_matches_opencv_scores_at_borders_and_ties(self):
        cv_rng = np.random.default_rng(342)
        source = cv_rng.integers(0, 2, (251, 270), dtype=np.uint8)
        target = cv_rng.integers(0, 2, (261, 257), dtype=np.uint8)
        rois = [(0, 0, 134, 261), (103, 7, 257, 255), (254, 23, 257, 200)]
        scorer = reg._AffineScore(source, target, rois, .005, .02)
        if scorer._native_score is None:
            self.skipTest("Rust DLL or supported OpenCV/CPU unavailable")
        for parameters in self.parameters():
            with self.subTest(parameters=parameters):
                self.assertEqual(legacy_score(scorer, parameters, True), scorer(parameters, True))
        self.assertGreater(scorer.native_calls, 0)

    def test_binary_blocks_match_dense_scoring_and_nonbinary_uses_dense_path(self):
        source = np.zeros((251, 270), dtype=np.uint8)
        source[13:146, 12:242] = 1
        source[70:133, 102:130] = 0
        target = np.zeros((261, 257), dtype=np.uint8)
        target[23:156, 16:248] = 1
        rois = [(0, 0, 257, 261), (3, 7, 244, 96), (254, 5, 257, 100)]
        library = _affine_native._load_library()
        if library is None or library[1] is None or library[2] is None:
            self.skipTest("Rust DLL with block scorer or supported CPU unavailable")
        kernel, batch, (build, blocks) = library
        counting_blocks = Mock(wraps=blocks)
        with patch.dict('os.environ', {'YINXKJ_DISABLE_RUST_AFFINE_BLOCKS': '0'}):
            with patch.object(_affine_native, '_load_library', return_value=(kernel, batch, None)):
                dense = _affine_native.make_scorer(source, target, rois)
            if dense is None:
                self.skipTest("Supported OpenCV/CPU unavailable")
            with patch.object(_affine_native, '_load_library', return_value=(
                    kernel, batch, (build, counting_blocks))):
                scorer = _affine_native.make_scorer(source, target, rois)
                nonbinary = _affine_native.make_scorer(source * 255, target * 255, rois)
            matrices = [reg._scaled_affine(parameters) for parameters in self.parameters()]
            matrices += [np.array([[-1., .02, 240.5], [.04, 1., -5.5]])]
            for matrix in matrices:
                with self.subTest(matrix=matrix):
                    self.assertEqual(dense(matrix), scorer(matrix))
                    self.assertEqual(dense(matrix), nonbinary(matrix))
            self.assertEqual(counting_blocks.call_count, sum(dense(m) is not None for m in matrices))


if __name__ == '__main__':
    unittest.main()
