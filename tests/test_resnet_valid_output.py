"""Valid final ResNet outputs, post-score zero borders and checkpoint compatibility."""

import gc
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import lightning
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import efficientad_ccd as cli
from export_branch_maps import extract_maps
from self_efficientad import EfficientAd
from self_efficientad.backbones import BACKBONE_SPECS
from self_efficientad.torch_model import EfficientAdModel


RESNETS = tuple(name for name in BACKBONE_SPECS if name.startswith("resnet"))


class ResNetValidOutputTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    @torch.inference_mode()
    def test_all_stages_keep_trunk_padding_and_align_valid_outputs(self):
        for name in RESNETS:
            model = EfficientAdModel(backbone=name).eval()
            spec = BACKBONE_SPECS[name]
            self.assertEqual(model.resnet_feature_mode, "valid")
            self.assertEqual(model.student.head.padding, (0, 0))
            self.assertEqual(model.ae.decoder[-1].padding, (0, 0))
            for trunk in (model.teacher.features, model.student.features):
                self.assertEqual(trunk[0].padding, (3, 3))
                self.assertEqual(trunk[3].padding, 1)
                for layer in trunk.modules():
                    if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3):
                        self.assertEqual(layer.padding, (1, 1))
                    # No instance-level forward monkey patches on residual blocks.
                    self.assertNotIn("forward", vars(layer))
            for h, w in ((224, 224), (256, 256), (257, 289)):
                with self.subTest(backbone=name, size=(h, w)):
                    image = torch.rand(1, 3, h, w)
                    normalized = (image - image.new_tensor([.485, .456, .406])[None, :, None, None])
                    normalized = normalized / image.new_tensor([.229, .224, .225])[None, :, None, None]
                    if hasattr(model.teacher, "forward_features"):
                        full_teacher = model.teacher.forward_features(normalized)
                    else:
                        full_teacher = model.teacher.features(normalized)
                    teacher = model.teacher(image)
                    torch.testing.assert_close(teacher, full_teacher[..., 1:-1, 1:-1], rtol=0, atol=0)
                    student, distance = model.compute_student_teacher_distance(image)
                    grid = ((h + spec.feature_stride - 1) // spec.feature_stride - 2,
                            (w + spec.feature_stride - 1) // spec.feature_stride - 2)
                    self.assertEqual(tuple(teacher.shape), (1, spec.out_channels, *grid))
                    self.assertEqual(tuple(student.shape), (1, 2 * spec.out_channels, *grid))
                    if hasattr(model.student, "forward_features"):
                        full_student = model.student.forward_features(normalized)
                    else:
                        full_student = model.student.features(normalized)
                    reference = F.conv2d(full_student, model.student.head.weight,
                                         model.student.head.bias, padding=0)
                    torch.testing.assert_close(student, reference, rtol=0, atol=0)
                    ae = model.autoencoder_features(image, grid)
                    self.assertEqual(ae.shape, teacher.shape)
                    raw_st = distance.mean(1, keepdim=True)
                    raw_ae = (ae - student[:, spec.out_channels:]).square().mean(1, keepdim=True)
                    maps = model.compute_maps(image, student, distance, normalize=False)
                    for raw, actual in zip((raw_st, raw_ae), maps, strict=True):
                        padded = model.pad_map_to_feature_grid(raw, (h, w))
                        torch.testing.assert_close(padded[..., 1:-1, 1:-1], raw, rtol=0, atol=0)
                        self.assertEqual(padded[..., 0, :].count_nonzero().item(), 0)
                        self.assertEqual(padded[..., -1, :].count_nonzero().item(), 0)
                        self.assertEqual(padded[..., :, 0].count_nonzero().item(), 0)
                        self.assertEqual(padded[..., :, -1].count_nonzero().item(), 0)
                        expected = F.interpolate(F.pad(raw, (1, 1, 1, 1)),
                                                 size=(h, w), mode="bilinear", align_corners=False)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertTrue(torch.isfinite(actual).all())
            del model

    @torch.inference_mode()
    def test_disable_map_padding_and_reject_wrong_grid(self):
        model = EfficientAdModel(backbone="resnet50_layer1v2", pad_maps=False).eval()
        image = torch.rand(1, 3, 224, 224)
        student, distance = model.compute_student_teacher_distance(image)
        st, _ = model.compute_maps(image, student, distance, normalize=False)
        expected = F.interpolate(distance.mean(1, keepdim=True), size=(224, 224), mode="bilinear")
        torch.testing.assert_close(st, expected, rtol=0, atol=0)
        model.pad_maps = True
        with self.assertRaisesRegex(ValueError, "高宽各小 2"):
            model.pad_map_to_feature_grid(torch.zeros(1, 1, 56, 56), (224, 224))

    @torch.inference_mode()
    def test_pdn_retains_original_features_and_four_cell_padding(self):
        for padding in (False, True):
            model = EfficientAdModel(backbone="pdn_small", padding=padding).eval()
            image = torch.rand(1, 3, 256, 256)
            student, distance = model.compute_student_teacher_distance(image)
            self.assertEqual(student.shape[-2:], (64, 64) if padding else (56, 56))
            st, _ = model.compute_maps(image, student, distance, normalize=False)
            expected = F.interpolate(F.pad(distance.mean(1, keepdim=True), (4, 4, 4, 4)),
                                     size=(256, 256), mode="bilinear")
            torch.testing.assert_close(st, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_export_matches_both_calibrated_branches(self):
        model = EfficientAdModel(backbone="resnet50_layer1v2").eval()
        for key, value in {"qa_st": 1., "qb_st": 3., "qa_ae": 2., "qb_ae": 5.}.items():
            model.quantiles[key].fill_(value)
        image = torch.rand(1, 3, 224, 224)
        maps = extract_maps(model, image)
        self.assertEqual(maps["st_native"].shape, (54, 54))
        np.testing.assert_allclose(maps["fused_calibrated"], model(image).anomaly_map[0, 0].numpy(),
                                   rtol=0, atol=0)

    def test_new_224_layer3_losses_backpropagate(self):
        model = EfficientAdModel(backbone="resnet50_layer3", hard_loss_mode="per_image").train()
        image = torch.rand(1, 3, 224, 224)
        losses = model(image, torch.rand_like(image))
        self.assertTrue(all(torch.isfinite(value) for value in losses))
        sum(losses).backward()
        for module in (model.student, model.ae):
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for gradient in model.student.head.weight.grad.chunk(2):
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))

    def test_mode_validation(self):
        for mode in ("typo", "", False):
            with self.assertRaisesRegex(ValueError, "resnet_feature_mode"):
                EfficientAdModel(backbone="resnet50_layer1", resnet_feature_mode=mode)
        with self.assertRaisesRegex(ValueError, "旧结构"):
            EfficientAdModel(backbone="resnet18_layer2", resnet_architecture_version=1,
                             resnet_feature_mode="valid")

    @torch.inference_mode()
    def test_cli_and_lightning_restore_native_legacy_and_valid_checkpoints(self):
        cli.load_runtime()
        for name in ("resnet18_layer2", "resnet50_layer1v2"):
            for mode in ("native", "valid"):
                with self.subTest(backbone=name, mode=mode), TemporaryDirectory() as directory:
                    config = dict(backbone=name, resnet_feature_mode=mode, model_size="small",
                                  imagenette_dir="unused", lr=1e-4, weight_decay=1e-5, device="cpu")
                    original = cli.new_model(config).eval()
                    image = torch.rand(1, 3, 256, 256)
                    expected = original.model(image).anomaly_map
                    path = Path(directory) / "model.pt"
                    # Writer must capture the actual mode even if omitted by its caller.
                    config.pop("resnet_feature_mode")
                    cli.save_checkpoint(path, original, config, {}, 0)
                    payload = cli.read_checkpoint(path)
                    self.assertEqual(payload["config"]["resnet_feature_mode"], mode)
                    if mode == "native":
                        payload["config"].pop("resnet_feature_mode")
                        torch.save(payload, path)
                    payload = cli.read_checkpoint(path)
                    self.assertEqual(payload["config"]["resnet_feature_mode"], mode)
                    restored = cli.new_model(payload["config"]).eval()
                    restored.model.load_state_dict(payload["model_state"])
                    torch.testing.assert_close(restored.model(image).anomaly_map, expected, rtol=0, atol=0)
                    del restored, payload
                    hparams = dict(original.hparams, teacher_pretrained=True)
                    if mode == "native":
                        hparams.pop("resnet_feature_mode")
                    path = Path(directory) / "model.ckpt"
                    torch.save(dict(state_dict=original.state_dict(), hyper_parameters=hparams,
                                    **{"pytorch-lightning_version": lightning.__version__}), path)
                    with patch("torchvision.models.resnet.ResNet50_Weights.get_state_dict",
                               side_effect=AssertionError("unexpected download")), \
                         patch("torchvision.models.resnet.ResNet18_Weights.get_state_dict",
                               side_effect=AssertionError("unexpected download")):
                        restored = EfficientAd.load_from_checkpoint(path).eval()
                    self.assertEqual(restored.model.resnet_feature_mode, mode)
                    self.assertEqual(restored.hparams["resnet_feature_mode"], mode)
                    self.assertEqual(restored.model.student.head.padding, (0, 0) if mode == "valid" else (1, 1))
                    self.assertEqual(restored.model.pad_maps, mode == "valid")
                    torch.testing.assert_close(restored.model(image).anomaly_map, expected, rtol=0, atol=0)
                    del original, restored


if __name__ == "__main__":
    unittest.main()
