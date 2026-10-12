"""Rectangular image-size configuration, preprocessing and all-backbone alignment."""

from __future__ import annotations

import gc
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from ccd_efficientad import cli
from ccd_efficientad.mask import default_mask_record, read_rgb
from ccd_efficientad.models.torch_model import EfficientAdModel


BACKBONES = (
    "pdn_small", "pdn_medium", "resnet18_layer2", "resnet18_layer3",
    "resnet50_layer1", "resnet50_layer1v2", "resnet50_layer2", "resnet50_layer3",
)


class RectangularInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cli.load_runtime()
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_configuration_normalizes_rectangles_and_preserves_legacy_square(self):
        self.assertEqual(cli.config_input_size({"image_size": 384}), (384, 384))
        self.assertEqual(cli.config_input_size({"input_size": [288, 512]}), (288, 512))
        self.assertEqual(cli.config_input_size({"input_height": 320, "input_width": 640}), (320, 640))
        self.assertEqual(cli.args_input_size(cli.build_parser().parse_args(["train"])), (256, 256))
        args = cli.build_parser().parse_args(["train", "--image-width", "512", "--image-height", "288"])
        self.assertEqual(cli.args_input_size(args), (288, 512))
        args = cli.build_parser().parse_args(["train", "--image-size", "1024"])
        self.assertEqual(cli.args_input_size(args), (1024, 1024))

    def test_dimension_validation_rejects_missing_pair_small_or_unaligned_sizes(self):
        parser = cli.build_parser()
        cases = (
            (["--image-width", "512"], "必须同时指定"),
            (["--image-height", "288"], "必须同时指定"),
            (["--image-width", "512", "--image-height", "288", "--image-size", "512"], "不能与"),
            (["--image-width", "512", "--image-height", "270"], "32 的倍数"),
            (["--image-width", "512", "--image-height", "224"], "不少于 256"),
            (["--image-width", "0", "--image-height", "288"], "正整数"),
        )
        for extras, message in cases:
            with self.subTest(extras=extras):
                args = parser.parse_args(["train", *extras])
                with self.assertRaisesRegex(ValueError, message):
                    cli.args_input_size(args)
        with self.assertRaisesRegex(ValueError, "正整数"):
            cli.config_input_size({"input_size": [288, True]})
        with self.assertRaisesRegex(ValueError, "不一致"):
            cli.config_input_size({"input_size": [288, 512], "input_height": 512, "input_width": 288})

    def test_resume_cannot_change_or_transpose_rectangular_dimensions(self):
        saved = {"optimizer_state": {}, "config": {"input_size": [288, 512]},
                 "manifest": {"category": "CCD1"}}
        args = cli.build_parser().parse_args(["train", "--resume", "last.pt", "--device", "cpu",
                                             "--image-width", "288", "--image-height", "512"])
        with patch.object(cli, "read_checkpoint", return_value=saved), \
                patch.object(cli, "new_output") as output, self.assertRaisesRegex(ValueError, "续训不能更改输入宽高"):
            cli.train_one(args, "CCD1")
        output.assert_not_called()

    def test_imagenette_loader_uses_saved_height_and_width_including_batch(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            category = root / "auxiliary" / "dummy"
            category.mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (333, 247), (80 + index, 100, 120)).save(category / f"{index}.png")
            config = {"imagenette_dir": str(category.parent), "model_size": "small", "backbone": "pdn_small",
                      "device": "cpu", "num_workers": 0, "lr": 1e-4, "weight_decay": 1e-5, "batch_size": 2,
                      "hard_loss_mode": "per_image",
                      "input_size": [288, 512], "image_size": 512}
            model = cli.new_model(config)
            cli.prepare_assets(model, config, load_teacher=False)
            images, _ = next(iter(model.imagenet_loader))
            self.assertEqual(tuple(images.shape), (2, 3, 288, 512))
            del model

    def test_dataset_resizes_image_and_mask_to_h_w_without_padding(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            mask_path = root / "mask.png"
            pixels = np.zeros((180, 330, 3), dtype=np.uint8)
            pixels[:, :110] = (255, 0, 0)
            Image.fromarray(pixels).save(source)
            mask = np.zeros((180, 330), dtype=np.uint8)
            mask[:, :110] = 255
            Image.fromarray(mask).save(mask_path)
            stat = source.stat()
            record = {"path": str(source), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "label": 0}
            record["circle"] = default_mask_record(read_rgb(source), {"default_mask": str(mask_path)})
            image_tensor, label, ignore_mask = cli.SnapshotDataset([record], (288, 512))[0]
            self.assertEqual(tuple(image_tensor.shape), (3, 288, 512))
            self.assertEqual(label, 0)
            self.assertEqual(tuple(ignore_mask.shape), (1, 288, 512))
            self.assertTrue(ignore_mask[:, :, :170].all())
            self.assertFalse(ignore_mask[:, :, 340:].any())
            no_mask_record = dict(record)
            no_mask_record.pop("circle")
            _, _, no_mask = cli.SnapshotDataset([no_mask_record], (288, 512))[0]
            self.assertFalse(no_mask.any())

    def test_box_coordinates_scale_x_and_y_independently(self):
        box = {"x0": 128, "y0": 72, "x1": 384, "y1": 216, "source_scales": [1]}
        mapped = cli.boxes_to_original([box], (288, 512), (1022, 511))[0]
        self.assertEqual((mapped["x0"], mapped["y0"], mapped["x1"], mapped["y1"]),
                         (255, 127, 767, 384))
        self.assertEqual(mapped["source_scales"], [1])

    def test_all_backbones_align_rectangular_features_and_restore_rectangular_maps(self):
        for backbone in BACKBONES:
            model = EfficientAdModel(backbone=backbone, teacher_pretrained=False).eval()
            for height, width in ((288, 512), (512, 288)):
                image = torch.rand(1, 3, height, width)
                with self.subTest(backbone=backbone, input_size=(height, width)), torch.inference_mode():
                    student, distance = model.compute_student_teacher_distance(image)
                    teacher_grid = distance.shape[-2:]
                    border = 8 if backbone.startswith("pdn_") else 2
                    self.assertEqual(tuple(teacher_grid),
                                     (height // model.feature_stride - border, width // model.feature_stride - border))
                    ae = model.autoencoder_features(image, teacher_grid)
                    self.assertEqual(student[:, model.teacher_out_channels:].shape[-2:], teacher_grid)
                    self.assertEqual(ae.shape[-2:], teacher_grid)
                    map_st, map_ae = model.compute_maps(image, student, distance, normalize=False)
                    self.assertEqual(tuple(map_st.shape), (1, 1, height, width))
                    self.assertEqual(tuple(map_ae.shape), (1, 1, height, width))
                    prediction = model(image)
                    self.assertEqual(tuple(prediction.anomaly_map.shape), (1, 1, height, width))
                    self.assertTrue(torch.isfinite(prediction.anomaly_map).all())
            # Validate the training path for every backbone, including all
            # ResNet-50 stages; representative smaller models also backprop below.
            model.train()
            with torch.no_grad():
                losses = model(image, torch.rand_like(image))
            self.assertTrue(all(torch.isfinite(loss).item() for loss in losses))
            del model, image, student, distance, ae, map_st, map_ae, prediction, losses
            gc.collect()

    def test_representative_pdns_and_resnet_train_with_rectangular_batches(self):
        height, width = 288, 512
        image, auxiliary = torch.rand(1, 3, height, width), torch.rand(1, 3, height, width)
        for backbone in ("pdn_small", "pdn_medium", "resnet18_layer2", "resnet18_layer3"):
            with self.subTest(backbone=backbone):
                model = EfficientAdModel(backbone=backbone, hard_loss_mode="per_image").train()
                losses = model(image, auxiliary)
                self.assertEqual(len(losses), 3)
                self.assertTrue(all(torch.isfinite(loss) for loss in losses))
                sum(losses).backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.student.parameters()))
                del model, losses
                gc.collect()


if __name__ == "__main__":
    unittest.main()
