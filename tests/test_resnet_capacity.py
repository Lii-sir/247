"""Behavioral regression tests for the versioned ResNet student and feature AE."""

import gc
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import lightning
import torch
from torch import nn

import efficientad_ccd as cli
from self_efficientad import EfficientAd
from self_efficientad.backbones import BACKBONE_SPECS, load_default_teacher_weights
from self_efficientad.torch_model import EfficientAdModel


class ResNetCapacityTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_resnet50_is_available_in_cli_and_model(self):
        self.assertIn("resnet50_layer3", BACKBONE_SPECS)
        args = cli.build_parser().parse_args(["train", "--backbone", "resnet50_layer3"])
        self.assertEqual(args.backbone, "resnet50_layer3")
        model = EfficientAdModel(backbone=args.backbone).eval()
        with torch.no_grad():
            x = torch.rand(1, 3, 256, 256)
            self.assertEqual(tuple(model.teacher(x).shape), (1, 1024, 16, 16))
            self.assertEqual(tuple(model.student(x).shape), (1, 2048, 16, 16))

    def test_public_pair_factories_also_default_to_whole_student_widening(self):
        from self_efficientad.backbones import build_resnet18_layer2_pair, build_resnet18_layer3_pair
        for factory in (build_resnet18_layer2_pair, build_resnet18_layer3_pair):
            with self.subTest(factory=factory.__name__):
                teacher, student = factory()
                self.assertEqual(teacher.features[0].out_channels, 64)
                self.assertEqual(student.features[0].out_channels, 128)

    def test_entire_student_is_widened_and_predictions_can_be_negative(self):
        for backbone, expected in (
            ("resnet18_layer2", [(128, 128), (128, 64), (256, 32)]),
            ("resnet18_layer3", [(128, 128), (128, 64), (256, 32), (512, 16)]),
            ("resnet50_layer3", [(128, 128), (512, 64), (1024, 32), (2048, 16)]),
        ):
            with self.subTest(backbone=backbone):
                self.assertIn(backbone, BACKBONE_SPECS)
                model = EfficientAdModel(backbone=backbone).eval()
                actual = []
                def record(_module, _inputs, output):
                    actual.append((output.shape[1], output.shape[-1]))
                handles = [model.student.features[i].register_forward_hook(record)
                           for i in [0, *range(4, len(model.student.features))]]
                with torch.no_grad():
                    model.student.head.weight.zero_()
                    model.student.head.bias.fill_(-0.25)
                    output = model.student(torch.rand(1, 3, 256, 256))
                for handle in handles:
                    handle.remove()
                self.assertEqual(actual, expected)
                torch.testing.assert_close(output, torch.full_like(output, -0.25))
                del model

    def test_feature_ae_width_and_spatial_output_follow_teacher(self):
        for backbone, widths, channels, size in (
            ("resnet18_layer2", (32, 64), 128, (32, 32)),
            ("resnet18_layer3", (64, 128), 256, (16, 16)),
            ("resnet50_layer3", (128, 256), 1024, (16, 16)),
        ):
            with self.subTest(backbone=backbone):
                self.assertIn(backbone, BACKBONE_SPECS)
                model = EfficientAdModel(backbone=backbone).eval()
                self.assertEqual(model.ae.encoder.enconv1.out_channels, widths[0])
                self.assertEqual(model.ae.encoder.enconv6.out_channels, widths[1])
                with torch.no_grad():
                    output = model.ae(torch.rand(1, 3, 256, 256), size)
                self.assertEqual(tuple(output.shape), (1, channels, *size))
                del model

    def test_new_resnet50_real_losses_update_both_branches_and_ae_only(self):
        self.assertIn("resnet50_layer3", BACKBONE_SPECS)
        torch.manual_seed(3)
        model = EfficientAdModel(backbone="resnet50_layer3", hard_loss_mode="per_image").train()
        teacher_before = {key: value.clone() for key, value in model.teacher.state_dict().items()}
        image = torch.rand(1, 3, 256, 256)
        losses = model(image, torch.rand_like(image))
        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
        sum(losses).backward()
        for module in (model.student, model.ae):
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for grad in model.student.head.weight.grad.chunk(2):
            self.assertGreater(grad.abs().sum().item(), 0)
        self.assertGreater(model.ae.encoder.enconv1.weight.grad.abs().sum().item(), 0)
        before = model.student.features[0].weight.detach().clone()
        torch.optim.SGD([*model.student.parameters(), *model.ae.parameters()], lr=1e-4).step()
        self.assertFalse(torch.equal(before, model.student.features[0].weight))
        self.assertFalse(model.teacher.training)
        for key, value in model.teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.teacher.parameters()))

    def test_feature_ae_supports_all_cli_sizes_and_signed_rectangular_outputs(self):
        from self_efficientad.resnet_autoencoder import ResNetFeatureAutoEncoder
        ae = ResNetFeatureAutoEncoder(256).eval()
        for image_size, feature_size in (
            ((256, 256), (16, 16)), ((384, 384), (24, 24)),
            ((512, 512), (32, 32)), ((768, 768), (48, 48)),
            ((256, 384), (16, 24)),
        ):
            with self.subTest(image_size=image_size), torch.no_grad():
                output = ae(torch.rand(1, 3, *image_size), feature_size)
                self.assertEqual(tuple(output.shape), (1, 256, *feature_size))
                self.assertTrue(torch.isfinite(output).all())
        with torch.no_grad():
            ae.decoder[-1].weight.zero_()
            ae.decoder[-1].bias.fill_(-0.5)
            output = ae(torch.rand(1, 3, 256, 256), (16, 16))
        torch.testing.assert_close(output, torch.full_like(output, -0.5))

    def test_resnet50_statistics_handle_inactive_channels_in_double_precision(self):
        from types import SimpleNamespace
        model = EfficientAd(backbone="resnet50_layer3", pre_processor=False,
                            post_processor=False, evaluator=False, visualizer=False)
        features = torch.tensor([0., 0., 0., 0., 1000., 1000.125, 1000.25, 1000.375]).reshape(1, 2, 2, 2)
        with patch.object(model.model.teacher, "forward", return_value=features):
            stats = model.teacher_channel_mean_std([SimpleNamespace(image=torch.rand(1, 3, 256, 256))])
        self.assertEqual(stats["std"][0, 0, 0, 0].item(), 1)
        torch.testing.assert_close(stats["std"][0, 1, 0, 0], features[:, 1].double().std(correction=0).float())

    def test_invalid_architecture_versions_fail_clearly(self):
        for version in (0, 3, True, "2"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "resnet_architecture_version"):
                EfficientAdModel(backbone="resnet18_layer3", resnet_architecture_version=version)
        with self.assertRaisesRegex(ValueError, "resnet50_layer3"):
            EfficientAdModel(backbone="resnet50_layer3", resnet_architecture_version=1)

    def test_teacher_loading_copies_all_resnet50_pretrained_features(self):
        self.assertIn("resnet50_layer3", BACKBONE_SPECS)
        from torchvision.models import resnet50
        source = resnet50(weights=None).eval()
        model = EfficientAdModel(backbone="resnet50_layer3").eval()
        with patch("torchvision.models.resnet.ResNet50_Weights.get_state_dict",
                   return_value=source.state_dict()):
            load_default_teacher_weights("resnet50_layer3", model.teacher)
        reference = nn.Sequential(source.conv1, source.bn1, source.relu, source.maxpool,
                                  source.layer1, source.layer2, source.layer3)
        x = torch.rand(1, 3, 256, 256)
        normalized = (x - x.new_tensor([.485, .456, .406])[None, :, None, None]) / x.new_tensor(
            [.229, .224, .225])[None, :, None, None]
        with torch.no_grad():
            torch.testing.assert_close(model.teacher(x), reference(normalized), rtol=0, atol=0)

    def test_cli_roundtrip_records_version_and_restores_legacy_without_version(self):
        cli.load_runtime()
        for version in (1, 2):
            with self.subTest(version=version), TemporaryDirectory() as directory:
                config = dict(imagenette_dir="unused", model_size="small", lr=1e-4,
                              weight_decay=1e-5, device="cpu", backbone="resnet18_layer3",
                              resnet_architecture_version=version)
                original = cli.new_model(config).eval()
                x = torch.rand(1, 3, 256, 256)
                with torch.no_grad():
                    expected = original.model(x).anomaly_map
                path = Path(directory) / "model.pt"
                # Writer must persist the actual architecture even for API callers
                # whose config did not explicitly specify it.
                config.pop("resnet_architecture_version")
                cli.save_checkpoint(path, original, config, {}, 0)
                saved = cli.read_checkpoint(path)
                self.assertEqual(saved["config"]["resnet_architecture_version"], version)
                if version == 1:
                    saved["config"].pop("resnet_architecture_version")
                    torch.save(saved, path)
                    saved = cli.read_checkpoint(path)
                restored = cli.new_model(saved["config"]).eval()
                restored.model.load_state_dict(saved["model_state"])
                with torch.no_grad():
                    torch.testing.assert_close(restored.model(x).anomaly_map, expected, rtol=0, atol=0)
                self.assertEqual(restored.model.resnet_architecture_version, version)

    def test_ddp_bootstrap_preserves_architecture_for_api_config_without_version(self):
        from types import SimpleNamespace
        from ccd_distributed import train_distributed
        cli.load_runtime()
        config = dict(imagenette_dir="unused", model_size="small", lr=1e-4,
                      weight_decay=1e-5, device="cpu", backbone="resnet18_layer2",
                      devices=["cpu", "cpu"], global_batch_size=2, batch_size=1)
        model = cli.new_model(config)
        model.imagenet_loader = SimpleNamespace(dataset=[0, 1])
        model.imagenet_iterator = None
        with TemporaryDirectory() as directory:
            root = Path(directory)
            def inspect_worker_handoff(_worker, *, args, nprocs, join):
                saved = cli.read_checkpoint(Path(args[1]))
                self.assertEqual(saved["config"]["resnet_architecture_version"], 2)
                (root / "checkpoints").mkdir()
                torch.save(saved, root / "checkpoints/last.pt")
                return SimpleNamespace(join=lambda timeout: True, processes=[])
            with patch("torch.multiprocessing.spawn", side_effect=inspect_worker_handoff):
                train_distributed(model, config, {"train": [0, 1]}, root, None, 1)

    def test_legacy_lightning_checkpoint_rebuilds_original_architecture(self):
        model = EfficientAd(backbone="resnet18_layer3", resnet_architecture_version=1,
                            pre_processor=False, post_processor=False, evaluator=False,
                            visualizer=False).eval()
        hparams = dict(model.hparams)
        hparams.pop("resnet_architecture_version", None)
        hparams["teacher_pretrained"] = True
        x = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            expected = model.model(x).anomaly_map
        with TemporaryDirectory() as directory:
            path = Path(directory) / "old.ckpt"
            torch.save(dict(state_dict=model.state_dict(), hyper_parameters=hparams,
                            **{"pytorch-lightning_version": lightning.__version__}), path)
            with patch("torchvision.models.resnet.ResNet18_Weights.get_state_dict",
                       side_effect=AssertionError("unexpected download")):
                restored = EfficientAd.load_from_checkpoint(path).eval()
            with torch.no_grad():
                torch.testing.assert_close(restored.model(x).anomaly_map, expected, rtol=0, atol=0)
        self.assertEqual(restored.hparams["resnet_architecture_version"], 1)

    def test_checkpoint_architecture_rebuild_preserves_dtype_and_eval_mode(self):
        model = EfficientAd(backbone="resnet18_layer2", pre_processor=False,
                            post_processor=False, evaluator=False, visualizer=False).double().eval()
        model.on_load_checkpoint({"hyper_parameters": {}})
        self.assertEqual(model.model.resnet_architecture_version, 1)
        self.assertEqual(next(model.model.student.parameters()).dtype, torch.float64)
        self.assertFalse(model.model.training)
        self.assertFalse(model.model.student.training)
        self.assertFalse(model.model.ae.training)
        self.assertFalse(model.model.teacher.training)
        with torch.no_grad():
            result = model.model(torch.rand(1, 3, 256, 256, dtype=torch.float64))
        self.assertEqual(result.anomaly_map.dtype, torch.float64)

    def test_lightning_fit_resumes_legacy_optimizer_on_rebuilt_parameters(self):
        from collections import namedtuple
        from itertools import repeat
        from torch.utils.data import DataLoader

        class OfflineEfficientAd(EfficientAd):
            def on_train_start(self):
                # Isolate restore/optimization from pretrained downloads and data setup.
                self.imagenet_iterator = repeat((torch.rand(1, 3, 256, 256),))

        batch_type = namedtuple("TrainingBatch", ["image"])
        loader = DataLoader([batch_type(torch.rand(3, 256, 256)) for _ in range(2)], batch_size=1)
        options = dict(backbone="resnet18_layer2", pre_processor=False,
                       post_processor=False, evaluator=False, visualizer=False)
        trainer_options = dict(accelerator="cpu", devices=1, max_epochs=-1,
                               logger=False, enable_checkpointing=False,
                               enable_progress_bar=False, enable_model_summary=False,
                               limit_val_batches=0, num_sanity_val_steps=0)
        with TemporaryDirectory() as directory:
            original = OfflineEfficientAd(**options, resnet_architecture_version=1)
            trainer = lightning.Trainer(**trainer_options, max_steps=1, default_root_dir=directory)
            trainer.fit(original, train_dataloaders=loader)
            before = original.model.student.features[0].weight.detach().clone()
            teacher_before = {key: value.clone() for key, value in original.model.teacher.state_dict().items()}
            path = Path(directory) / "legacy.ckpt"
            trainer.save_checkpoint(path)
            payload = torch.load(path, weights_only=False)
            payload["hyper_parameters"].pop("resnet_architecture_version")
            torch.save(payload, path)

            restored = OfflineEfficientAd(**options)
            resumed = lightning.Trainer(**trainer_options, max_steps=2, default_root_dir=directory)
            resumed.fit(restored, train_dataloaders=loader, ckpt_path=path)
            self.assertEqual(resumed.global_step, 2)
            self.assertEqual(restored.model.resnet_architecture_version, 1)
            optimizer = resumed.optimizers[0]
            expected = {id(p) for module in (restored.model.student, restored.model.ae)
                        for p in module.parameters()}
            self.assertEqual({id(p) for group in optimizer.param_groups for p in group["params"]}, expected)
            self.assertTrue(all(int(state["step"]) == 2 for state in optimizer.state.values()))
            self.assertFalse(torch.equal(before, restored.model.student.features[0].weight))
            for key, value in restored.model.teacher.state_dict().items():
                torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)

    def test_each_loss_reaches_only_its_intended_trainable_branches(self):
        torch.manual_seed(7)
        model = EfficientAdModel(backbone="resnet18_layer3", hard_loss_mode="per_image").train()
        losses = model(torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256))
        parameters = (model.student.head.weight, model.student.features[0].weight,
                      model.ae.encoder.enconv1.weight)
        for index, loss in enumerate(losses):
            with self.subTest(loss=("student_teacher", "autoencoder", "student_autoencoder")[index]):
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
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
