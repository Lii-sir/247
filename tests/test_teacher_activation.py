"""Teacher activation semantics, checkpoint compatibility and training gradients."""

import gc
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import lightning
import torch

from ccd_efficientad import cli
from ccd_efficientad.models import EfficientAd
from ccd_efficientad.models.backbones import ResNet50Features, load_default_teacher_weights
from ccd_efficientad.models.torch_model import EfficientAdModel


KEY = "resnet_teacher_output_activation"
BACKBONE = "resnet50_layer1v2"


def configuration(**overrides):
    return dict(imagenette_dir="unused", model_size="small", lr=1e-4,
                weight_decay=1e-5, device="cpu", backbone=BACKBONE, **overrides)


class TeacherActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cli.load_runtime()

    def tearDown(self):
        gc.collect()

    @torch.no_grad()
    def test_only_last_relu_is_removed_with_identical_weights_and_internal_activations(self):
        torch.manual_seed(413)
        teacher = ResNet50Features(layer=1).eval()
        image = torch.rand(1, 3, 64, 80)
        blocks = teacher.features[-1]
        calls = []
        handles = [block.relu.register_forward_hook(
            lambda _m, _i, o, index=i: calls.append((index, o.clone())))
            for i, block in enumerate(blocks)]
        try:
            reference = teacher(image)
            reference_calls = calls.copy()
            calls.clear()
            teacher.remove_final_activation = True
            actual = teacher(image)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual([i for i, _ in reference_calls], [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual([i for i, _ in calls], [0, 0, 0, 1, 1, 1, 2, 2])
        for (_, before), (_, after) in zip(reference_calls, calls):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertLess(actual.min().item(), 0)
        torch.testing.assert_close(actual.relu(), reference, rtol=0, atol=0)

    def test_explicit_modes_and_invalid_modes(self):
        for activation in (None, "none", "relu"):
            model = EfficientAdModel(backbone=BACKBONE, **{KEY: activation})
            self.assertEqual(model.teacher.remove_final_activation, activation != "relu")
            del model
        for name, mode in ((BACKBONE, "typo"), (BACKBONE, False),
                           ("resnet50_layer1", "none"), ("pdn_small", "none")):
            with self.subTest(backbone=name, mode=mode), self.assertRaises(ValueError):
                EfficientAdModel(backbone=name, **{KEY: mode})

    @torch.no_grad()
    def test_loading_default_weights_preserves_both_output_modes(self):
        from torchvision.models import resnet50
        source = resnet50(weights=None).eval()
        for activation in ("relu", "none"):
            model = EfficientAdModel(backbone=BACKBONE, **{KEY: activation}).eval()
            with patch("torchvision.models.resnet.ResNet50_Weights.get_state_dict",
                       return_value=source.state_dict()) as fetch:
                load_default_teacher_weights(BACKBONE, model.teacher)
                fetch.assert_called_once()
            self.assertEqual(model.teacher.remove_final_activation, activation == "none")
            torch.testing.assert_close(model.teacher.features[4][-1].conv3.weight,
                                       source.layer1[-1].conv3.weight, rtol=0, atol=0)
            del model

    @torch.no_grad()
    def test_cli_and_lightning_roundtrip_both_modes_and_unmarked_old_relu(self):
        for activation, legacy in (("none", False), ("relu", False), ("relu", True)):
            with self.subTest(activation=activation, legacy=legacy), TemporaryDirectory() as folder:
                config = configuration(**{KEY: activation})
                original = cli.new_model(config).eval()
                original.model.mean_std["mean"].fill_(0.2)
                original.model.mean_std["std"].fill_(0.7)
                for key, value in dict(qa_st=0.1, qb_st=1., qa_ae=0.2, qb_ae=2.).items():
                    original.model.quantiles[key].fill_(value)
                image = torch.rand(1, 3, 256, 256)
                expected = original.model(image).anomaly_map
                config.pop(KEY)  # Saver must use actual model, not caller assumptions.
                path = Path(folder) / "model.pt"
                cli.save_checkpoint(path, original, config, {}, 4, calibration={"threshold": 1.23})
                payload = cli.read_checkpoint(path)
                self.assertEqual(payload["config"][KEY], activation)
                if legacy:
                    payload["config"].pop(KEY)
                    torch.save(payload, path)
                    payload = cli.read_checkpoint(path)
                self.assertEqual(payload["config"][KEY], activation)
                restored = cli.new_model(payload["config"]).eval()
                restored.model.load_state_dict(payload["model_state"], strict=True)
                torch.testing.assert_close(restored.model(image).anomaly_map, expected, rtol=0, atol=0)
                self.assertEqual(payload["calibration"]["threshold"], 1.23)
                del restored, payload
                hparams = dict(original.hparams, teacher_pretrained=True)
                self.assertEqual(hparams[KEY], activation)
                if legacy:
                    hparams.pop(KEY)
                path = Path(folder) / "model.ckpt"
                torch.save(dict(state_dict=original.state_dict(), hyper_parameters=hparams,
                                **{"pytorch-lightning_version": lightning.__version__}), path)
                with patch("torchvision.models.resnet.ResNet50_Weights.get_state_dict",
                           side_effect=AssertionError("unexpected download")):
                    restored = EfficientAd.load_from_checkpoint(path).eval()
                self.assertEqual(restored.model.resnet_teacher_output_activation, activation)
                self.assertEqual(restored.hparams[KEY], activation)
                torch.testing.assert_close(restored.model(image).anomaly_map, expected, rtol=0, atol=0)
                del original, restored

    def test_resume_rejects_activation_change_before_training(self):
        args = cli.build_parser().parse_args([
            "train", "--resume", "unused.pt", "--device", "cpu",
            "--resnet-teacher-output-activation", "none"])
        payload = dict(config=configuration(**{KEY: "relu"}), manifest={}, optimizer_state={})
        with patch.object(cli, "read_checkpoint", return_value=payload):
            with self.assertRaisesRegex(ValueError, "续训不能更改 Teacher"):
                cli.train_one(args, "CCD1")

    @torch.no_grad()
    def test_signed_teacher_statistics_use_actual_cropped_features(self):
        torch.manual_seed(529)
        model = cli.new_model(configuration()).eval()
        # Include an incomplete final batch, and retain the signed values.
        images = torch.rand(3, 3, 64, 80)
        batches = [SimpleNamespace(image=images[:2]), SimpleNamespace(image=images[2:])]
        values = torch.cat([model.model.teacher(batch.image) for batch in batches]).double()
        self.assertLess(values.min().item(), 0)
        self.assertEqual(values.shape, (3, 256, 14, 18))
        stats = model.teacher_channel_mean_std(batches)
        expected_mean = values.mean((0, 2, 3), keepdim=True).float()
        variance = values.var((0, 2, 3), correction=0, keepdim=True)
        expected_std = variance.where(variance != 0, 1).sqrt().float()
        torch.testing.assert_close(stats["mean"], expected_mean)
        torch.testing.assert_close(stats["std"], expected_std)
        self.assertFalse(torch.allclose(stats["mean"], values.clamp_min(0).mean((0, 2, 3), keepdim=True).float()))
        model.model.mean_std.update(stats)
        standardized = (values.float() - stats["mean"]) / stats["std"]
        torch.testing.assert_close(standardized.mean((0, 2, 3)), torch.zeros(256), atol=1e-6, rtol=0)
        torch.testing.assert_close(standardized.var((0, 2, 3), correction=0), torch.ones(256))

    def test_lightning_real_resume_keeps_activation_statistics_and_optimizer_parameters(self):
        from collections import namedtuple
        from itertools import repeat
        from torch.utils.data import DataLoader

        class OfflineEfficientAd(EfficientAd):
            def on_train_start(self):
                # Exercise real Trainer resume without external datasets/downloads.
                self.imagenet_iterator = repeat((torch.rand(1, 3, 256, 256),))

        batch_type = namedtuple("TrainingBatch", ["image"])
        options = dict(backbone=BACKBONE, pre_processor=False, post_processor=False,
                       evaluator=False, visualizer=False)
        trainer_options = dict(accelerator="cpu", devices=1, max_epochs=-1, logger=False,
                               enable_checkpointing=False, enable_progress_bar=False,
                               enable_model_summary=False, limit_val_batches=0, num_sanity_val_steps=0)
        for activation in ("relu", "none"):
            with self.subTest(activation=activation), TemporaryDirectory() as directory:
                loader = DataLoader([batch_type(torch.rand(3, 256, 256)) for _ in range(2)], batch_size=1)
                original = OfflineEfficientAd(**options, **{KEY: activation})
                original.model.mean_std["mean"].data.fill_(0.2)
                original.model.mean_std["std"].data.fill_(0.7)
                trainer = lightning.Trainer(**trainer_options, max_steps=1, default_root_dir=directory)
                trainer.fit(original, train_dataloaders=loader)
                before = [original.model.student.head.weight.detach().clone(),
                          original.model.ae.encoder.enconv1.weight.detach().clone()]
                frozen = {key: value.clone() for key, value in original.model.teacher.state_dict().items()}
                path = Path(directory) / "resume.ckpt"
                trainer.save_checkpoint(path)
                if activation == "relu":
                    payload = torch.load(path, weights_only=False)
                    payload["hyper_parameters"].pop(KEY)  # Genuine pre-change checkpoint semantics.
                    torch.save(payload, path)
                    del payload
                restored = OfflineEfficientAd(**options, **{KEY: "none" if activation == "relu" else "relu"})
                resumed = lightning.Trainer(**trainer_options, max_steps=2, default_root_dir=directory)
                resumed.fit(restored, train_dataloaders=loader, ckpt_path=path)
                self.assertEqual(resumed.global_step, 2)
                self.assertEqual(restored.model.resnet_teacher_output_activation, activation)
                self.assertEqual(restored.hparams[KEY], activation)
                optimizer = resumed.optimizers[0]
                expected_parameters = {id(p) for module in (restored.model.student, restored.model.ae)
                                       for p in module.parameters()}
                self.assertEqual({id(p) for group in optimizer.param_groups for p in group["params"]}, expected_parameters)
                self.assertTrue(all(int(state["step"]) == 2 for state in optimizer.state.values()))
                for old, current in zip(before, (restored.model.student.head.weight,
                                                restored.model.ae.encoder.enconv1.weight)):
                    self.assertFalse(torch.equal(old, current))
                self.assertFalse(restored.model.teacher.training)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in restored.model.teacher.parameters()))
                for key, value in restored.model.teacher.state_dict().items():
                    torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
                for key in ("mean", "std"):
                    torch.testing.assert_close(restored.model.mean_std[key], original.model.mean_std[key], rtol=0, atol=0)
                del original, restored, trainer, resumed, optimizer

    def test_ddp_handoff_keeps_activation_and_geometry_without_caller_metadata(self):
        from ccd_efficientad.distributed import train_distributed
        config = configuration(devices=["cpu", "cpu"], global_batch_size=2, batch_size=1)
        model = cli.new_model(config)
        model.imagenet_loader = SimpleNamespace(dataset=[0, 1])
        model.imagenet_iterator = None
        with TemporaryDirectory() as folder:
            root = Path(folder)
            def inspect_worker_handoff(_worker, *, args, nprocs, join):
                saved = cli.read_checkpoint(Path(args[1]))
                self.assertEqual(saved["config"][KEY], "none")
                self.assertEqual(saved["config"]["resnet_feature_mode"], "valid")
                worker_model = cli.new_model(saved["config"])
                worker_model.model.load_state_dict(saved["model_state"], strict=True)
                self.assertTrue(worker_model.model.teacher.remove_final_activation)
                (root / "checkpoints").mkdir()
                torch.save(saved, root / "checkpoints/last.pt")
                return SimpleNamespace(join=lambda timeout: True, processes=[])
            with patch("torch.multiprocessing.spawn", side_effect=inspect_worker_handoff):
                train_distributed(model, config, {"train": [0, 1]}, root, None, 1)

    def test_signed_teacher_real_losses_and_gradients_at_batch_one_and_two(self):
        for batch in (1, 2):
            with self.subTest(batch=batch):
                torch.manual_seed(2026)
                model = EfficientAdModel(backbone=BACKBONE,
                                         hard_loss_mode="per_image" if batch > 1 else "global").train()
                model.mean_std["mean"].data.fill_(0.2)
                model.mean_std["std"].data.fill_(0.7)
                frozen = {key: value.clone() for key, value in model.teacher.state_dict().items()}
                outputs = {name: [] for name in ("teacher", "student", "ae")}
                handles = [getattr(model, name).register_forward_hook(
                    lambda _m, _i, o, key=name: outputs[key].append(o.detach())) for name in outputs]
                try:
                    losses = model(torch.rand(batch, 3, 256, 256), torch.rand(batch, 3, 256, 256))
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertLess(outputs["teacher"][0].min().item(), 0)
                teacher, augmented = [(value - 0.2) / 0.7 for value in outputs["teacher"]]
                student, auxiliary, augmented_student = outputs["student"]
                ae, = outputs["ae"]
                errors = (teacher - student[:, :256]).square().flatten(1)
                hard = torch.stack([row[row >= torch.quantile(row, .999)].mean() for row in errors]).mean()
                expected = (hard + auxiliary[:, :256].square().mean(),
                            (augmented - ae).square().mean(), (ae - augmented_student[:, 256:]).square().mean())
                parameters = (model.student.head.weight, model.ae.encoder.enconv1.weight)
                before = [p.detach().clone() for p in parameters]
                for index, (loss, reference) in enumerate(zip(losses, expected)):
                    torch.testing.assert_close(loss.detach(), reference)
                    head, ae_grad = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                    if index == 1:
                        self.assertIsNone(head)
                    else:
                        active = 0 if index == 0 else 1
                        self.assertGreater(head.chunk(2)[active].abs().sum().item(), 0)
                        self.assertEqual(head.chunk(2)[1 - active].abs().sum().item(), 0)
                    if index == 0:
                        self.assertIsNone(ae_grad)
                    else:
                        self.assertGreater(ae_grad.abs().sum().item(), 0)
                sum(losses).backward()
                for module in (model.student, model.ae):
                    for name, p in module.named_parameters():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                torch.optim.SGD([*model.student.parameters(), *model.ae.parameters()], lr=1e-4).step()
                for p, old in zip(parameters, before):
                    self.assertFalse(torch.equal(p, old))
                self.assertFalse(model.teacher.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.teacher.parameters()))
                for key, value in model.teacher.state_dict().items():
                    torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
                del model, losses, outputs, parameters


if __name__ == "__main__":
    unittest.main()
