"""Device selection, global budgets and distributed training regression tests."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import efficientad_ccd as cli


class ToyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = nn.Linear(1, 1, bias=False)
        self.ae = nn.Linear(1, 1, bias=False)
        self.mean_std = nn.ParameterDict({"mean": nn.Parameter(torch.zeros(1))})
        self.quantiles = nn.ParameterDict({"q": nn.Parameter(torch.zeros(1))})

    def forward(self, batch, batch_imagenet):
        student = self.student(batch)
        ae = self.ae(batch)
        return ((student - batch).square().mean() + self.student(batch_imagenet).square().mean(),
                (ae - batch).square().mean(), (student - ae).square().mean())


class ToyModel(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.model = ToyNetwork()
        self.imagenet_loader = DataLoader(TensorDataset(torch.arange(1, 5).float().reshape(-1, 1) / 10),
                                          batch_size=batch_size)
        self.imagenet_iterator = iter(self.imagenet_loader)


def toy_collate(items):
    return cli.Batch(torch.stack(items), None, None)


def optimization_worker(rank, root, distributed, invalid):
    from ccd_distributed import optimize

    cli.load_runtime()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    world_size = 2 if distributed else 1
    if distributed:
        dist.init_process_group("gloo", init_method=(Path(root) / "store").as_uri(),
                                rank=rank, world_size=world_size)
    try:
        batch_size = 4 // world_size
        model = ToyModel(batch_size)
        images = torch.arange(1, 5).float().reshape(-1, 1)
        if invalid:
            images[0] = float("nan")
        loader = DataLoader(images, batch_size=batch_size, collate_fn=toy_collate)
        config = {"batch_size": batch_size, "device": "cpu", "seed": 42,
                  "lr": 0.01, "weight_decay": 0.0, "max_steps": 3}
        with patch.object(cli, "make_loader", return_value=loader):
            try:
                optimize(model, config, {"train": [0, 1, 2, 3], "category": "toy"},
                         Path(root), None, 1, rank=rank, world_size=world_size)
                result = {"state": model.model.state_dict(), "failed": False}
            except RuntimeError as error:
                if "有限数值" not in str(error):
                    raise
                result = {"failed": True}
        torch.save(result, Path(root) / f"rank{rank}.pt")
    finally:
        if distributed:
            dist.destroy_process_group()


def mid_epoch_resume_worker(rank, root, num_workers=0):
    from ccd_distributed import optimize

    cli.load_runtime()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    root = Path(root)
    dist.init_process_group("gloo", init_method=(root / "store").as_uri(), rank=rank, world_size=2)
    try:
        config = {"batch_size": 2, "device": "cpu", "seed": 42,
                  "lr": 0.01, "weight_decay": 0.0, "max_steps": 4}
        images = torch.arange(1, 13).float().reshape(-1, 1)
        auxiliary = torch.arange(1, 9).float().reshape(-1, 1) / 10
        manifest = {"train": list(range(12)), "category": "toy"}
        loader = DataLoader(images, batch_size=2, collate_fn=toy_collate,
                            num_workers=num_workers, persistent_workers=num_workers > 0)
        original_save = cli.save_checkpoint

        def capture_first_step(path, model, config, manifest, step, *args, **kwargs):
            original_save(path, model, config, manifest, step, *args, **kwargs)
            if step == 1:
                original_save(root / "first.pt", model, config, manifest, step, *args, **kwargs)

        for phase in ("continuous", "resumed"):
            model = ToyModel(2)
            model.imagenet_loader = DataLoader(TensorDataset(auxiliary), batch_size=2,
                                               num_workers=num_workers, persistent_workers=num_workers > 0)
            model.imagenet_iterator = None
            previous = cli.read_checkpoint(root / "first.pt") if phase == "resumed" else None
            if previous:
                model.model.load_state_dict(previous["model_state"])
            with patch.object(cli, "make_loader", return_value=loader), \
                    patch.object(cli, "save_checkpoint", side_effect=capture_first_step):
                optimize(model, config, manifest, root / phase, previous, 1, rank=rank, world_size=2)
            dist.barrier()
    finally:
        dist.destroy_process_group()


class DistributedConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cli.load_runtime()

    def test_gpu_list_and_legacy_devices(self):
        args = cli.build_parser().parse_args(["train", "--device", "0", "1", "--batch-size", "4"])
        with patch.object(cli.torch.cuda, "is_available", return_value=True), \
                patch.object(cli.torch.cuda, "device_count", return_value=2):
            self.assertEqual(cli.choose_devices(args.device), ["cuda:0", "cuda:1"])
            self.assertEqual(cli.choose_device("cuda"), "cuda:0")
            self.assertEqual(cli.choose_device(["1"]), "cuda:1")
            self.assertEqual(cli.choose_device("cpu"), "cpu")
            with self.assertRaisesRegex(ValueError, "单.*设备"):
                cli.choose_device(["0", "1"])

    def test_invalid_devices_fail_before_training(self):
        with patch.object(cli.torch.cuda, "is_available", return_value=True), \
                patch.object(cli.torch.cuda, "device_count", return_value=2):
            for devices in (["0", "0"], ["-1"], ["2"], ["cpu", "0"], ["auto", "1"], ["typo"]):
                with self.subTest(devices=devices), self.assertRaises(ValueError):
                    cli.choose_devices(devices)

    def test_global_image_budget_and_resume_world_size(self):
        from ccd_distributed import configure_training_world
        config = {"batch_size": 4, "max_steps": 99, "max_images_requested": 101}
        configure_training_world(config, ["cuda:0", "cuda:1"], resume=False)
        self.assertEqual(config["global_batch_size"], 8)
        self.assertEqual(config["max_steps"], 13)
        self.assertEqual(config["max_images"], 104)
        configure_training_world(config, ["cuda:2", "cuda:3"], resume=True)
        with self.assertRaisesRegex(ValueError, "world_size"):
            configure_training_world(config, ["cpu"], resume=True)
        legacy = {"batch_size": 1, "max_steps": 10}
        with self.assertRaisesRegex(ValueError, "world_size"):
            configure_training_world(legacy, ["cuda:0", "cuda:1"], resume=True)

    def test_samplers_are_disjoint_complete_batches_and_change_epoch(self):
        from torch.utils.data import DataLoader, TensorDataset
        from ccd_distributed import distributed_loader
        source = DataLoader(TensorDataset(cli.torch.arange(11)), batch_size=2)
        loaders = [distributed_loader(source, rank=rank, world_size=2, seed=42) for rank in range(2)]
        shards = [[int(value) for batch in loader for value in batch[0]] for loader in loaders]
        self.assertEqual([len(shard) for shard in shards], [4, 4])
        self.assertFalse(set(shards[0]) & set(shards[1]))
        loaders[0].sampler.set_epoch(1)
        next_epoch = [int(value) for batch in loaders[0] for value in batch[0]]
        self.assertNotEqual(next_epoch, shards[0])
        with self.assertRaisesRegex(ValueError, "全局 batch"):
            distributed_loader(DataLoader([1, 2, 3], batch_size=2), rank=0, world_size=2, seed=42)

    def test_auxiliary_preparation_can_defer_worker_start(self):
        from PIL import Image

        with TemporaryDirectory() as temporary:
            folder = Path(temporary) / "normal"
            folder.mkdir()
            Image.new("RGB", (256, 256)).save(folder / "sample.png")
            model = cli.new_model({"imagenette_dir": temporary, "model_size": "small",
                                   "lr": 0.001, "weight_decay": 0.0, "device": "cpu"})
            model.prepare_imagenette_data((256, 256), num_workers=1, start_iterator=False)
            self.assertEqual(len(model.imagenet_loader.dataset), 1)
            self.assertIsNone(model.imagenet_iterator)
            self.assertIsNone(model.imagenet_loader._iterator)


class DistributedOptimizationTests(unittest.TestCase):
    def test_mid_epoch_resume_matches_uninterrupted_optimization(self):
        self._check_mid_epoch_resume(num_workers=0)

    def test_mid_epoch_resume_with_loader_workers(self):
        self._check_mid_epoch_resume(num_workers=1)

    def _check_mid_epoch_resume(self, num_workers):
        import csv

        cli.load_runtime()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for phase in ("continuous", "resumed"):
                (root / phase).mkdir()
            torch.multiprocessing.spawn(mid_epoch_resume_worker, args=(temporary, num_workers), nprocs=2)
            expected = cli.read_checkpoint(root / "continuous/checkpoints/last.pt")
            actual = cli.read_checkpoint(root / "resumed/checkpoints/last.pt")
            for name in expected["model_state"]:
                torch.testing.assert_close(actual["model_state"][name], expected["model_state"][name])
            self.assertEqual(actual["scheduler_state"], expected["scheduler_state"])
            for key, state in expected["optimizer_state"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(actual["optimizer_state"]["state"][key][name], value)
            with (root / "continuous/loss.csv").open() as left, (root / "resumed/loss.csv").open() as right:
                continuous = list(csv.DictReader(left))[1:]
                resumed = list(csv.DictReader(right))
                self.assertEqual([row["step"] for row in resumed], ["2", "3", "4"])
                for expected_row, actual_row in zip(continuous, resumed, strict=True):
                    self.assertAlmostEqual(float(actual_row["loss"]), float(expected_row["loss"]), places=5)

    def test_two_ranks_match_global_batch_optimizer_and_loss(self):
        import csv

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            single, parallel = root / "single", root / "parallel"
            single.mkdir()
            parallel.mkdir()
            optimization_worker(0, str(single), False, False)
            torch.multiprocessing.spawn(optimization_worker, args=(str(parallel), True, False), nprocs=2)
            expected = torch.load(single / "rank0.pt", weights_only=True)["state"]
            for rank in (0, 1):
                actual = torch.load(parallel / f"rank{rank}.pt", weights_only=True)["state"]
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name])
            with (single / "loss.csv").open() as left, (parallel / "loss.csv").open() as right:
                for expected_row, actual_row in zip(csv.DictReader(left), csv.DictReader(right), strict=True):
                    for field in ("loss", "loss_st", "loss_ae", "loss_stae"):
                        self.assertAlmostEqual(float(actual_row[field]), float(expected_row[field]), places=5)
            self.assertEqual(torch.load(parallel / "checkpoints/last.pt", weights_only=True)["step"], 3)

    def test_nan_on_one_rank_stops_both_without_checkpoint(self):
        with TemporaryDirectory() as temporary:
            torch.multiprocessing.spawn(optimization_worker, args=(temporary, True, True), nprocs=2)
            for rank in (0, 1):
                self.assertTrue(torch.load(Path(temporary) / f"rank{rank}.pt", weights_only=True)["failed"])
            self.assertFalse((Path(temporary) / "checkpoints/last.pt").exists())


if __name__ == "__main__":
    unittest.main()
