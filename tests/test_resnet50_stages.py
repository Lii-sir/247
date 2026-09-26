"""ResNet-50 stage selection must adapt the teacher, whole student and AE."""

import gc
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import lightning
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck

import efficientad_ccd as cli
from self_efficientad import EfficientAd
from self_efficientad.backbones import BACKBONE_SPECS, load_default_teacher_weights
from self_efficientad.torch_model import EfficientAdModel


class ResNet50StageTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_cli_and_registry_expose_both_shallower_stages(self):
        for stage, channels, stride in ((1, 256, 4), (2, 512, 8)):
            name = f"resnet50_layer{stage}"
            with self.subTest(backbone=name):
                self.assertIn(name, BACKBONE_SPECS)
                spec = BACKBONE_SPECS[name]
                self.assertEqual((spec.out_channels, spec.feature_stride), (channels, stride))
                args = cli.build_parser().parse_args(["train", "--backbone", name])
                self.assertEqual(args.backbone, name)

    def test_stages_use_bottlenecks_with_all_internal_and_residual_widths_doubled(self):
        for stage in (1, 2):
            name = f"resnet50_layer{stage}"
            with self.subTest(backbone=name):
                self.assertIn(name, BACKBONE_SPECS)
                model = EfficientAdModel(backbone=name).eval()
                self.assertEqual(len(model.teacher.features), 4 + stage)
                self.assertEqual(len(model.student.features), 4 + stage)
                self.assertEqual(model.student.features[0].out_channels, 128)
                for teacher_stage, student_stage, depth in zip(
                    model.teacher.features[4:], model.student.features[4:], (3, 4), strict=False
                ):
                    self.assertEqual(len(teacher_stage), depth)
                    self.assertEqual(len(student_stage), depth)
                    for teacher_block, student_block in zip(teacher_stage, student_stage, strict=True):
                        self.assertIsInstance(student_block, Bottleneck)
                        for conv_name in ("conv1", "conv2", "conv3"):
                            teacher_conv = getattr(teacher_block, conv_name)
                            student_conv = getattr(student_block, conv_name)
                            self.assertEqual(student_conv.in_channels, 2 * teacher_conv.in_channels)
                            self.assertEqual(student_conv.out_channels, 2 * teacher_conv.out_channels)
                            self.assertEqual(student_conv.stride, teacher_conv.stride)
                        if teacher_block.downsample is not None:
                            self.assertEqual(student_block.downsample[0].in_channels,
                                             2 * teacher_block.downsample[0].in_channels)
                            self.assertEqual(student_block.downsample[0].out_channels,
                                             2 * teacher_block.downsample[0].out_channels)
                channels = 256 * 2 ** (stage - 1)
                self.assertEqual(tuple(model.student.head.weight.shape), (2 * channels, 2 * channels, 3, 3))
                self.assertFalse(model.pad_maps)
                del model

    def test_teacher_student_and_ae_align_at_native_square_and_odd_rectangular_grids(self):
        for stage, channels, hidden, stride in ((1, 256, 128, 4), (2, 512, 256, 8)):
            name = f"resnet50_layer{stage}"
            with self.subTest(backbone=name):
                self.assertIn(name, BACKBONE_SPECS)
                model = EfficientAdModel(backbone=name).eval()
                self.assertEqual(model.ae.encoder.enconv1.out_channels, hidden // 2)
                self.assertEqual(model.ae.encoder.enconv6.out_channels, hidden)
                with torch.no_grad():
                    for h, w in ((256, 256), (257, 289)):
                        image = torch.rand(1, 3, h, w)
                        grid = ((h + stride - 1) // stride, (w + stride - 1) // stride)
                        teacher = model.teacher(image)
                        student = model.student(image)
                        ae = model.autoencoder_features(image, teacher.shape[-2:])
                        self.assertEqual(tuple(teacher.shape), (1, channels, *grid))
                        self.assertEqual(tuple(student.shape), (1, 2 * channels, *grid))
                        self.assertEqual(tuple(ae.shape), (1, channels, *grid))
                        result = model(image)
                        self.assertEqual(tuple(result.anomaly_map.shape), (1, 1, h, w))
                        self.assertTrue(torch.isfinite(result.anomaly_map).all())
                    model.student.head.weight.zero_()
                    model.student.head.bias.fill_(-0.25)
                    model.ae.decoder[-1].weight.zero_()
                    model.ae.decoder[-1].bias.fill_(-0.5)
                    self.assertTrue((model.student(image) == -0.25).all())
                    self.assertTrue((model.autoencoder_features(image, grid) == -0.5).all())
                del model

    def test_all_resnet50_teachers_load_exact_native_pretrained_prefixes(self):
        source = resnet50(weights=None).eval()
        source_features = list(nn.Sequential(source.conv1, source.bn1, source.relu, source.maxpool,
                                             source.layer1, source.layer2, source.layer3))
        image = torch.rand(1, 3, 256, 256)
        normalized = (image - image.new_tensor([.485, .456, .406])[None, :, None, None]) / image.new_tensor(
            [.229, .224, .225])[None, :, None, None]
        for stage in (1, 2, 3):
            name = f"resnet50_layer{stage}"
            with self.subTest(backbone=name):
                self.assertIn(name, BACKBONE_SPECS)
                model = EfficientAdModel(backbone=name).eval()
                with patch("torchvision.models.resnet.ResNet50_Weights.get_state_dict",
                           return_value=source.state_dict()):
                    load_default_teacher_weights(name, model.teacher)
                reference = nn.Sequential(*source_features[:4 + stage])
                self.assertEqual(set(model.teacher.state_dict()),
                                 {f"features.{key}" for key in reference.state_dict()})
                with torch.no_grad():
                    torch.testing.assert_close(model.teacher(image), reference(normalized), rtol=0, atol=0)
                del model

    def test_feature_ae_adapts_to_all_larger_cli_image_sizes(self):
        for stage, channels, stride in ((1, 256, 4), (2, 512, 8)):
            model = EfficientAdModel(backbone=f"resnet50_layer{stage}").eval()
            with torch.no_grad():
                for size in (384, 512, 768):
                    with self.subTest(stage=stage, image_size=size):
                        image = torch.rand(1, 3, size, size)
                        teacher = model.teacher(image)
                        expected_shape = (1, channels, size // stride, size // stride)
                        self.assertEqual(tuple(teacher.shape), expected_shape)
                        ae = model.autoencoder_features(image, teacher.shape[-2:])
                        self.assertEqual(tuple(ae.shape), expected_shape)
                        self.assertTrue(torch.isfinite(ae).all())
                        del image, teacher, ae
            del model

    def test_real_batch_two_loss_updates_student_and_ae_and_keeps_teacher_frozen(self):
        for stage in (1, 2):
            name = f"resnet50_layer{stage}"
            with self.subTest(backbone=name):
                self.assertIn(name, BACKBONE_SPECS)
                torch.manual_seed(26)
                model = EfficientAdModel(backbone=name, hard_loss_mode="per_image").train()
                teacher_before = {k: v.clone() for k, v in model.teacher.state_dict().items()}
                before = [model.student.features[0].weight.detach().clone(),
                          model.ae.encoder.enconv1.weight.detach().clone()]
                losses = model(torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256))
                self.assertTrue(all(torch.isfinite(loss) for loss in losses))
                sum(losses).backward()
                for module in (model.student, model.ae):
                    for key, parameter in module.named_parameters():
                        self.assertIsNotNone(parameter.grad, key)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                for half in model.student.head.weight.grad.chunk(2):
                    self.assertGreater(half.abs().sum().item(), 0)
                torch.optim.SGD([*model.student.parameters(), *model.ae.parameters()], lr=1e-4).step()
                for previous, updated in zip(before, (model.student.features[0].weight,
                                                     model.ae.encoder.enconv1.weight), strict=True):
                    self.assertFalse(torch.equal(previous, updated))
                self.assertFalse(model.teacher.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.teacher.parameters()))
                for key, value in model.teacher.state_dict().items():
                    torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
                del model, losses

    def test_both_checkpoint_formats_restore_new_stages_without_downloading(self):
        cli.load_runtime()
        for stage in (1, 2):
            name = f"resnet50_layer{stage}"
            with self.subTest(backbone=name), TemporaryDirectory() as directory:
                self.assertIn(name, BACKBONE_SPECS)
                config = dict(imagenette_dir="unused", model_size="small", lr=1e-4,
                              weight_decay=1e-5, device="cpu", backbone=name)
                original = cli.new_model(config).eval()
                image = torch.rand(1, 3, 256, 256)
                with torch.no_grad():
                    expected = original.model(image).anomaly_map
                path = Path(directory) / "model.pt"
                cli.save_checkpoint(path, original, config, {}, 0)
                saved = cli.read_checkpoint(path)
                self.assertEqual(saved["config"]["resnet_architecture_version"], 2)
                restored = cli.new_model(saved["config"]).eval()
                restored.model.load_state_dict(saved["model_state"])
                with torch.no_grad():
                    torch.testing.assert_close(restored.model(image).anomaly_map, expected, rtol=0, atol=0)
                del restored, saved
                path = Path(directory) / "model.ckpt"
                hparams = dict(original.hparams, teacher_pretrained=True)
                torch.save(dict(state_dict=original.state_dict(), hyper_parameters=hparams,
                                **{"pytorch-lightning_version": lightning.__version__}), path)
                with patch("torchvision.models.resnet.ResNet50_Weights.get_state_dict",
                           side_effect=AssertionError("unexpected teacher download")):
                    restored = EfficientAd.load_from_checkpoint(path).eval()
                with torch.no_grad():
                    torch.testing.assert_close(restored.model(image).anomaly_map, expected, rtol=0, atol=0)
                del original, restored

    def test_normalized_losses_and_individual_gradient_routes_at_batch_one_and_two(self):
        for stage in (1, 2):
            for batch_size in (1, 2):
                with self.subTest(stage=stage, batch_size=batch_size):
                    torch.manual_seed(91)
                    mode = "global" if batch_size == 1 else "per_image"
                    model = EfficientAdModel(backbone=f"resnet50_layer{stage}", hard_loss_mode=mode).train()
                    model.mean_std["mean"].data.fill_(0.2)
                    model.mean_std["std"].data.fill_(0.7)
                    # Capture the actual augmented forward passes without replacing
                    # augmentation, BatchNorm, dropout, or the real network.
                    outputs = {"teacher": [], "student": [], "ae": []}
                    handles = []
                    for name in outputs:
                        def record(_module, _inputs, output, key=name):
                            outputs[key].append(output.detach())
                        handles.append(getattr(model, name).register_forward_hook(record))
                    try:
                        losses = model(torch.rand(batch_size, 3, 256, 256),
                                       torch.rand(batch_size, 3, 256, 256))
                    finally:
                        for handle in handles:
                            handle.remove()
                    channels = model.teacher_out_channels
                    teacher, augmented_teacher = [(x - 0.2) / 0.7 for x in outputs["teacher"]]
                    student, auxiliary_student, augmented_student = outputs["student"]
                    ae, = outputs["ae"]
                    errors = (teacher - student[:, :channels]).square().flatten(1)
                    hard = torch.stack([row[row >= torch.quantile(row, 0.999)].mean() for row in errors]).mean()
                    expected = (hard + auxiliary_student[:, :channels].square().mean(),
                                (augmented_teacher - ae).square().mean(),
                                (ae - augmented_student[:, channels:]).square().mean())
                    for loss, reference in zip(losses, expected, strict=True):
                        torch.testing.assert_close(loss.detach(), reference)
                    parameters = (model.student.head.weight, model.student.features[0].weight,
                                  model.ae.encoder.enconv1.weight)
                    for index, loss in enumerate(losses):
                        head_grad, stem_grad, ae_grad = torch.autograd.grad(
                            loss, parameters, retain_graph=index < 2, allow_unused=True)
                        if index == 1:
                            self.assertIsNone(head_grad)
                            self.assertIsNone(stem_grad)
                        else:
                            active, inactive = (0, 1) if index == 0 else (1, 0)
                            halves = head_grad.chunk(2)
                            self.assertGreater(halves[active].abs().sum().item(), 0)
                            self.assertEqual(halves[inactive].abs().sum().item(), 0)
                            self.assertGreater(stem_grad.abs().sum().item(), 0)
                        if index == 0:
                            self.assertIsNone(ae_grad)
                        else:
                            self.assertGreater(ae_grad.abs().sum().item(), 0)
                    self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
                    del model, losses, outputs, parameters

    def test_new_stages_reject_legacy_version_and_wrong_teacher_width(self):
        for stage in (1, 2):
            name = f"resnet50_layer{stage}"
            with self.subTest(backbone=name):
                self.assertIn(name, BACKBONE_SPECS)
                with self.assertRaisesRegex(ValueError, "resnet_architecture_version=2"):
                    EfficientAdModel(backbone=name, resnet_architecture_version=1)
                with self.assertRaisesRegex(ValueError, "teacher_out_channels"):
                    EfficientAdModel(backbone=name, teacher_out_channels=384)


if __name__ == "__main__":
    unittest.main()
