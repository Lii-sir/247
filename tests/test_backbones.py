"""验证 EfficientAD 可替换 backbone 的结构和兼容性。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import torch
import lightning

import efficientad_ccd as cli
from self_efficientad import EfficientAd
from self_efficientad.torch_model import EfficientAdModel, SmallPatchDescriptionNetwork


class BackboneTests(unittest.TestCase):
    def test_lightning_checkpoint_with_pretrained_flag_loads_offline(self) -> None:
        model = EfficientAd(backbone="resnet18_layer2", pre_processor=False,
                            post_processor=False, evaluator=False, visualizer=False).eval()
        image = torch.rand(1, 3, 256, 256)
        with torch.inference_mode():
            expected = model.model(image).anomaly_map
        # 模拟曾使用 teacher_pretrained=True 训练的 checkpoint；加载必须只用内嵌权重。
        hparams = dict(model.hparams)
        hparams["teacher_pretrained"] = True
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.ckpt"
            torch.save({"state_dict": model.state_dict(), "hyper_parameters": hparams,
                        "pytorch-lightning_version": lightning.__version__}, checkpoint)
            with patch("torchvision.models.resnet.ResNet18_Weights.get_state_dict",
                       side_effect=AssertionError("checkpoint restore attempted pretrained download")):
                restored = EfficientAd.load_from_checkpoint(checkpoint, map_location="cpu").eval()
            with torch.inference_mode():
                actual = restored.model(image).anomaly_map
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(restored._teacher_loaded_from_checkpoint)

    def test_lightning_learning_rate_decays_at_optimizer_step(self) -> None:
        model = EfficientAd(backbone="resnet18_layer2", pre_processor=False,
                            post_processor=False, evaluator=False, visualizer=False)
        model._trainer = SimpleNamespace(max_epochs=-1, max_steps=20)
        config = model.configure_optimizers()
        scheduler_config = config["lr_scheduler"]
        self.assertIsInstance(scheduler_config, dict)
        self.assertEqual(scheduler_config["interval"], "step")
        optimizer = config["optimizer"]
        scheduler = scheduler_config["scheduler"]
        for step in range(1, 21):
            optimizer.step()
            scheduler.step()
            expected = model.lr if step < 19 else model.lr * 0.1
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected)

    def test_resnet_statistics_allow_inactive_channels(self) -> None:
        model = EfficientAd(backbone="resnet18_layer2", pre_processor=False,
                            post_processor=False, evaluator=False, visualizer=False)
        features = torch.tensor([0.0, 0.0, 0.0, 0.0, 1000.0, 1000.125, 1000.25, 1000.375]).reshape(1, 2, 2, 2)
        with patch.object(model.model.teacher, "forward", return_value=features):
            statistics = model.teacher_channel_mean_std([SimpleNamespace(image=torch.zeros(1, 3, 256, 256))])
        self.assertEqual(statistics["std"][0, 0, 0, 0].item(), 1.0)
        torch.testing.assert_close(statistics["std"][0, 1, 0, 0], features[:, 1].double().std(correction=0).float())
        cli.load_runtime()
        cli.check_statistics(statistics)

    def test_resnet_statistics_reject_fully_constant_features(self) -> None:
        model = EfficientAd(backbone="resnet18_layer2", pre_processor=False,
                            post_processor=False, evaluator=False, visualizer=False)
        with patch.object(model.model.teacher, "forward", return_value=torch.zeros(1, 128, 2, 2)):
            with self.assertRaisesRegex(ValueError, "全部.*恒定"):
                model.teacher_channel_mean_std([SimpleNamespace(image=torch.zeros(1, 3, 256, 256))])

    def test_legacy_torch_positional_arguments(self) -> None:
        model = EfficientAdModel(16, "small", True, False, "global")
        with torch.no_grad():
            self.assertEqual(model.teacher(torch.rand(1, 3, 256, 256)).shape, (1, 16, 64, 64))
        self.assertFalse(model.pad_maps)

    def test_legacy_lightning_positional_arguments(self) -> None:
        model = EfficientAd("unused", 16, "small", 0.003, 0.004, True, False,
                            1, "global", False, False, False, False)
        self.assertEqual(model.lr, 0.003)
        self.assertEqual(model.weight_decay, 0.004)
        self.assertEqual(model.model.teacher.conv1.padding, (3, 3))

    def test_invalid_model_configuration_is_rejected(self) -> None:
        for kwargs in ({"model_size": "typo"}, {"teacher_out_channels": 0},
                       {"backbone": "resnet18_layer2", "teacher_out_channels": 384}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EfficientAdModel(**kwargs)

    def test_lightning_medium_loader_uses_matching_real_weights(self) -> None:
        model = EfficientAd(backbone="pdn_medium", pre_processor=False,
                            post_processor=False, evaluator=False, visualizer=False)
        with TemporaryDirectory() as temporary:
            weights = Path(temporary) / "efficientad_pretrained_weights"
            weights.mkdir()
            source = {name: value.detach().clone() for name, value in model.model.teacher.state_dict().items()}
            source["conv1.bias"].fill_(0.125)
            torch.save(source, weights / "pretrained_teacher_medium.pth")
            with patch("self_efficientad.lightning_model.Path", return_value=Path(temporary)):
                model.prepare_pretrained_model()
        torch.testing.assert_close(model.model.teacher.conv1.bias, source["conv1.bias"], rtol=0, atol=0)

    def test_resume_rejects_explicit_architecture_change_before_output(self) -> None:
        cli.load_runtime()
        for saved_config, arguments in (
            ({"model_size": "small"}, ["--backbone", "resnet18_layer2"]),
            ({"model_size": "small", "backbone": "resnet18_layer2"}, ["--model-size", "small"]),
        ):
            with self.subTest(arguments=arguments):
                args = cli.build_parser().parse_args(["train", "--resume", "unused.pt", *arguments])
                saved = {"optimizer_state": {}, "config": saved_config, "manifest": {}}
                with patch.object(cli, "read_checkpoint", return_value=saved), \
                        patch.object(cli, "new_output") as create_output:
                    with self.assertRaisesRegex(ValueError, "backbone"):
                        cli.train_one(args, "CCD1")
                    create_output.assert_not_called()

    def test_lightning_resume_does_not_reload_default_teacher(self) -> None:
        for restored in (False, True):
            with self.subTest(restored=restored):
                model = EfficientAd(backbone="resnet18_layer2", pre_processor=False,
                                    post_processor=False, evaluator=False, visualizer=False)
                model.model.mean_std["std"].data.fill_(1)
                model._trainer = SimpleNamespace(
                    datamodule=SimpleNamespace(train_batch_size=1, eval_batch_size=1, num_workers=0),
                    train_dataloader=[SimpleNamespace(image=torch.zeros(1, 3, 256, 256))],
                )
                if restored:
                    model.on_load_checkpoint({"state_dict": model.state_dict()})
                with patch.object(model, "prepare_pretrained_model") as load_teacher, \
                        patch.object(model, "prepare_imagenette_data"):
                    model.on_train_start()
                self.assertEqual(load_teacher.call_count, 0 if restored else 1)

    def test_teacher_is_frozen_during_real_optimizer_step(self) -> None:
        model = EfficientAdModel(backbone="resnet18_layer2", hard_loss_mode="per_image").train()
        self.assertTrue(all(not parameter.requires_grad for parameter in model.teacher.parameters()))
        before = {key: value.clone() for key, value in model.teacher.state_dict().items()}
        optimizer = torch.optim.Adam(list(model.student.parameters()) + list(model.ae.parameters()), lr=1e-4)
        losses = model(torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256))
        sum(losses).backward()
        for module in (model.student, model.ae):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
            self.assertTrue(any(torch.count_nonzero(grad) for grad in grads))
        optimizer.step()
        for key, value in model.teacher.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_cli_accepts_resnet18_layer2(self) -> None:
        args = cli.build_parser().parse_args([
            "train", "--backbone", "resnet18_layer2",
        ])
        self.assertEqual(args.backbone, "resnet18_layer2")

    def test_cli_rejects_model_size_with_backbone(self) -> None:
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args([
                "train", "--model-size", "medium",
                "--backbone", "resnet18_layer2",
            ])

    def test_old_config_still_builds_pdn_small(self) -> None:
        cli.load_runtime()
        model = cli.new_model({
            "imagenette_dir": "unused",
            "model_size": "small",
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "device": "cpu",
        })
        self.assertEqual(model.backbone, "pdn_small")
        self.assertIsInstance(model.model.teacher, SmallPatchDescriptionNetwork)

    def test_resnet18_layer2_training_and_inference_shapes(self) -> None:
        model = EfficientAdModel(backbone="resnet18_layer2")
        image = torch.rand(1, 3, 256, 256)
        teacher = model.teacher(image)
        student = model.student(image)
        self.assertEqual(teacher.shape, (1, 128, 32, 32))
        self.assertEqual(student.shape, (1, 256, 32, 32))

        model.train()
        losses = model(image, torch.rand_like(image))
        self.assertEqual(len(losses), 3)
        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
        self.assertFalse(model.teacher.training)

        model.eval()
        prediction = model(image)
        self.assertEqual(prediction.anomaly_map.shape, (1, 1, 256, 256))
        self.assertTrue(torch.isfinite(prediction.anomaly_map).all())

    def test_explicit_pdn_medium_loads_medium_teacher(self) -> None:
        cli.load_runtime()
        model = cli.new_model({
            "imagenette_dir": "unused",
            "model_size": "small",
            "backbone": "pdn_medium",
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "device": "cpu",
        })
        config = {
            "backbone": "pdn_medium",
            "model_size": "small",
            "assets_dir": "assets",
            "imagenette_dir": "unused",
            "num_workers": 0,
            "image_size": 256,
            "device": "cpu",
            "teacher_weights": None,
        }
        def prepare_data(*_args, **_kwargs) -> None:
            model.imagenet_loader = SimpleNamespace(dataset=[object()])

        with patch("efficientad_ccd.Path.is_file", return_value=True), \
                patch("efficientad_ccd.torch.load", return_value=model.model.teacher.state_dict()) as load, \
                patch.object(model, "prepare_imagenette_data", side_effect=prepare_data):
            cli.prepare_assets(model, config, load_teacher=True)
        self.assertIn("pretrained_teacher_medium.pth", str(load.call_args.args[0]))

    def test_resnet_checkpoint_rebuilds_identical_model(self) -> None:
        cli.load_runtime()
        config = {
            "imagenette_dir": "unused",
            "model_size": "small",
            "backbone": "resnet18_layer2",
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "device": "cpu",
        }
        original = cli.new_model(config).eval()
        image = torch.rand(1, 3, 256, 256)
        with torch.inference_mode():
            expected = original.model(image).anomaly_map
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            torch.save(original.model.state_dict(), checkpoint)
            restored = cli.new_model(config)
            restored.model.load_state_dict(torch.load(checkpoint, weights_only=True))
            restored.eval()
            with torch.inference_mode():
                actual = restored.model(image).anomaly_map
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
