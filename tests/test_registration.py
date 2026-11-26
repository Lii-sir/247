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
from algorithm.vision3d.alignment import registration


class MaskMappingInputTests(unittest.TestCase):
    def test_normalized_readonly_masks_reuse_source_and_match_legacy_pixels(self):
        base = np.zeros((18, 30), np.uint8)
        base[3:14, 6:25] = 255
        for source in (base, base[:, ::2]):
            original = source.copy()
            source.setflags(write=False)
            for matrix in (None, [[.91, .02, -2.5], [-.01, .95, 1.5], [.0002, -.0001, 1.]]):
                values = dict(scale=.9, tx=-2.5, ty=1.5)
                if matrix is not None:
                    values['homography_2d_to_3d'] = matrix
                expected = registration.align_crop_mask_to_depth(source, (4, 6, 34, 24), (24, 38), values)
                with patch.object(cv2, 'warpPerspective', wraps=cv2.warpPerspective) as warp:
                    actual = registration.align_crop_mask_to_depth(
                        source, (4, 6, 34, 24), (24, 38), values, binary_255=True)
                self.assertIs(warp.call_args.args[0], source)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(source, original)

    def test_unmarked_uint8_and_other_formats_keep_positive_value_contract(self):
        examples = [np.array([[0, 1, 2], [17, 254, 255]], np.uint8),
                    np.array([[True, False], [False, True]]),
                    np.array([[-3, 0], [1, 256]], np.int16),
                    np.array([[np.nan, -1.], [0., .2]], np.float32)]
        values = dict(scale=1., tx=0., ty=0.)
        for source in examples:
            for trusted in (False, True) if source.dtype != np.uint8 else (False,):
                actual = registration.align_crop_mask_to_depth(
                    source, (0, 0, 3, 2), source.shape, values, binary_255=trusted)
                np.testing.assert_array_equal(actual, (source > 0).astype(np.uint8)*255)

    def test_formal_producer_marks_masks_and_shared_payload_preserves_internal_flag(self):
        from importlib import import_module
        from algorithm.application.board.detect_board import DetectBoard
        from algorithm.execution.processes.registration_process import SharedBoardInputs
        import torch
        detector = DetectBoard.__new__(DetectBoard)
        detector.min_component_area = 1
        result = SimpleNamespace(masks=SimpleNamespace(data=torch.tensor([[[0., .8], [1., .3]]])),
            boxes=SimpleNamespace(cls=torch.tensor([1]), conf=torch.tensor([.9])), names={1: 'wire'})
        instance = detector._extract_instances(result, (2, 2))[0]
        self.assertIs(instance['_mask_binary_255'], True)
        np.testing.assert_array_equal(instance['mask'], np.array([[0, 255], [255, 0]], np.uint8))
        geometry = import_module('algorithm.vision2d.instance_geometry')
        owner = SharedBoardInputs()
        try:
            _, records = geometry.share_instances([instance], owner)
            self.assertIs(records[0]['fields']['_mask_binary_255'], True)
            self.assertNotIn('_mask_binary_255', geometry.public_instance(instance, (0, 0, 2, 2)))
        finally:
            owner.close()

    def test_class_unions_keep_components_separate_and_use_local_confidence(self):
        import torch
        from algorithm.vision2d.segmentation_instances import extract_instances

        masks = np.zeros((4, 24, 30), np.uint8)
        masks[0, 2:6, 2:6] = 1
        masks[0, 2:4, 24:26] = 1
        masks[1, 5:10, 5:10] = 1
        masks[2, 14:19, 20:25] = 1
        masks[2, 22, 0] = 1  # Below min_area after connected-component labeling.
        masks[3, 5:9, 5:9] = 1  # Overlaps wire but belongs to silver.
        result = SimpleNamespace(masks=SimpleNamespace(data=torch.from_numpy(masks)),
            boxes=SimpleNamespace(cls=torch.tensor([1, 1, 1, 2]), conf=torch.tensor([.65, .9, .35, .8])),
            names={1: "wire", 2: "silver"})
        instances = extract_instances(result, masks.shape[1:], 4)
        self.assertEqual([1, 2, 3, 4], [item["instance_id"] for item in instances])
        self.assertEqual(["wire", "wire", "wire", "silver"], [item["class_name"] for item in instances])
        np.testing.assert_allclose([.65, .9, .35, .8], [item["confidence"] for item in instances])
        self.assertEqual([(24, 2, 2, 2), (2, 2, 8, 8), (20, 14, 5, 5), (5, 5, 4, 4)],
                         [item["_mask_bbox"] for item in instances])
        self.assertEqual([4, 40, 25, 16], [item["_mask_area_px"] for item in instances])
        self.assertEqual(255, instances[1]["mask"][5, 5])
        self.assertEqual(255, instances[3]["mask"][5, 5])
        np.testing.assert_array_equal(masks, result.masks.data.numpy())
        result.orig_shape = masks.shape[1:]
        result.boxes.xyxy = torch.tensor([[1.5, 1.5, 26., 6.], [5., 5., 10., 10.],
                                           [0., 14., 25., 23.], [5., 5., 9., 9.]])
        cropped = extract_instances(result, masks.shape[1:], 4)
        self.assertEqual(len(instances), len(cropped))
        for original, packed in zip(instances, cropped):
            np.testing.assert_array_equal(original["mask"], packed["mask"])
            self.assertEqual(original["confidence"], packed["confidence"])

    def test_non_native_masks_exclude_letterbox_padding(self):
        import torch
        from algorithm.vision2d.segmentation_instances import extract_instances

        masks = torch.zeros((1, 8, 8))
        masks[0, :2, :] = 1
        masks[0, 3:5, 4:6] = 1
        result = SimpleNamespace(masks=SimpleNamespace(data=masks),
            boxes=SimpleNamespace(cls=torch.tensor([1]), conf=torch.tensor([.9])), names={1: "wire"})
        instance, = extract_instances(result, (4, 8), 1)
        self.assertEqual((4, 1, 2, 2), instance["_mask_bbox"])
        self.assertEqual(4, instance["_mask_area_px"])

    def test_diagonally_adjacent_crops_are_joined_before_area_filtering(self):
        import torch
        from algorithm.vision2d.segmentation_instances import extract_instances

        masks = torch.zeros((2, 10, 10), dtype=torch.uint8)
        masks[0, 2, 2] = 1
        masks[1, 3, 3] = 1
        result = SimpleNamespace(masks=SimpleNamespace(data=masks), orig_shape=(10, 10),
            boxes=SimpleNamespace(cls=torch.tensor([1, 1]), conf=torch.tensor([.7, .8]),
                                  xyxy=torch.tensor([[2, 2, 3, 3], [3, 3, 4, 4]])), names={1: "wire"})
        instance, = extract_instances(result, (10, 10), 2)
        self.assertEqual((2, 2, 2, 2), instance["_mask_bbox"])
        self.assertEqual(2, instance["_mask_area_px"])
        self.assertAlmostEqual(.8, instance["confidence"])


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
