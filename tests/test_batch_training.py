"""验证 EfficientAD 批量训练新增的数学语义。"""

from __future__ import annotations

import unittest

import torch

from efficientad_ccd import load_runtime, make_loader
from self_efficientad.torch_model import student_teacher_hard_loss


class BatchTrainingTests(unittest.TestCase):
    def test_per_image_matches_global_for_single_image(self) -> None:
        torch.manual_seed(42)
        distance = torch.rand(1, 8, 7, 7)
        global_loss = student_teacher_hard_loss(distance, mode="global")
        per_image_loss = student_teacher_hard_loss(distance, mode="per_image")
        self.assertTrue(torch.equal(global_loss, per_image_loss))

    def test_per_image_gives_each_image_its_own_quantile(self) -> None:
        distance = torch.zeros(2, 1, 10, 10)
        distance[0, 0, 0, 0] = 10.0
        distance[1, 0, 0, 0] = 2.0
        loss = student_teacher_hard_loss(distance, mode="per_image")
        self.assertAlmostEqual(float(loss), 6.0, places=6)

    def test_unknown_hard_loss_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            student_teacher_hard_loss(torch.ones(1, 1, 2, 2), mode="unknown")

    def test_training_loader_uses_full_batches_and_drops_remainder(self) -> None:
        load_runtime()
        config = {
            "image_size": 256,
            "batch_size": 2,
            "num_workers": 0,
            "device": "cpu",
        }
        records = [{"path": "unused", "label": 0}] * 5
        loader = make_loader(records, config, training=True)
        batches = list(loader.batch_sampler)
        self.assertEqual([len(batch) for batch in batches], [2, 2])

    def test_per_image_matches_global_for_single_image_shapes(self) -> None:
        torch.manual_seed(7)
        for shape in ((1, 1, 8, 8), (1, 3, 5, 7), (1, 8, 2, 11)):
            distance = torch.rand(shape)
            self.assertTrue(torch.equal(
                student_teacher_hard_loss(distance, mode="global"),
                student_teacher_hard_loss(distance, mode="per_image"),
            ))


if __name__ == "__main__":
    unittest.main()
