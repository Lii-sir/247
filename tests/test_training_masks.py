"""训练、续训和评估的无 mask/缺失 mask 回退，使用合成图片，不修改生产资源。"""

from __future__ import annotations

import copy
import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from ccd_efficientad import cli
from ccd_efficientad.mask import default_mask_record, read_rgb


class TrainingMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cli.load_runtime()

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = {"category": "CCD1", "summary": {}}
        for index, split in enumerate(("train", "val", "threshold_val", "test")):
            image = self.root / f"{split}.png"
            Image.new("RGB", (24, 32), (index * 30, 70, 110)).save(image)
            stat = image.stat()
            self.manifest[split] = [{"path": str(image), "label": int(split == "test"),
                                     "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}]
        self.mask = self.root / "mask.png"
        Image.new("L", (24, 32), 0).save(self.mask)
        self.config = {"image_size": 32, "device": "cpu", "num_workers": 0}

    def attach_mask(self, path=None):
        path = self.mask if path is None else path
        for split in ("train", "val", "threshold_val", "test"):
            record = self.manifest[split][0]
            record["circle"] = default_mask_record(read_rgb(Path(record["path"])), {"default_mask": str(path)})

    def assert_full_image(self, config=None, manifest=None):
        config = self.config if config is None else config
        manifest = self.manifest if manifest is None else manifest
        self.assertEqual(config["mask_mode"], "none")
        self.assertIsNone(config.get("circle_config"))
        self.assertIsNone(config.get("circle_params"))
        for split in ("train", "val", "threshold_val", "test"):
            for record in manifest.get(split, []):
                self.assertNotIn("circle", record)
                _, _, mask = cli.SnapshotDataset([record], 32)[0]
                self.assertFalse(mask.any())

    def test_parser_train_and_evaluate_accept_no_mask(self):
        for arguments in (["train"], ["evaluate", "--checkpoint", "model.pt"]):
            self.assertFalse(cli.build_parser().parse_args(arguments).no_mask)
            self.assertTrue(cli.build_parser().parse_args([*arguments, "--no-mask"]).no_mask)

    def test_no_configuration_is_full_image_without_warning(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            cli.prepare_circle_records(self.manifest, self.config)
        self.assertFalse(captured)
        self.assert_full_image()

    def test_explicit_no_mask_clears_saved_records_without_reading_missing_json(self):
        self.attach_mask()
        self.config.update(no_mask=True, circle_config=str(self.root / "missing.json"))
        self.manifest["default_mask_summary"] = {"count": 4}
        with self.assertWarnsRegex(RuntimeWarning, "--no-mask"):
            cli.prepare_circle_records(self.manifest, self.config)
        self.assert_full_image()
        self.assertNotIn("default_mask_summary", self.manifest)

    def test_missing_configured_mask_warns_and_clears_every_split(self):
        self.attach_mask()
        # 第一张仍可用，后续旧记录的 mask 被移除：回退应同时清除已处理的记录。
        removed = self.root / "removed.png"
        Image.new("L", (24, 32), 0).save(removed)
        self.manifest["test"][0]["circle"] = default_mask_record(
            read_rgb(Path(self.manifest["test"][0]["path"])), {"default_mask": str(removed)})
        removed.unlink()
        self.config["circle_params"] = {"default_mask": str(self.mask)}
        with self.assertWarnsRegex(RuntimeWarning, "无 mask"):
            cli.prepare_circle_records(self.manifest, self.config)
        self.assert_full_image()
        self.assertEqual(self.config["mask_fallback"]["reason"], "mask_file_not_found")
        self.assertEqual(self.manifest["mask_fallback"], self.config["mask_fallback"])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            cli.prepare_circle_records(self.manifest, self.config)
        self.assertFalse(captured)

    def test_relative_missing_mask_warns_but_missing_config_still_fails(self):
        path = self.root / "config.json"
        path.write_text(json.dumps({"CCD1": {"default_mask": "absent.png"}}), encoding="utf-8")
        self.config["circle_config"] = str(path)
        with self.assertWarnsRegex(RuntimeWarning, "absent.png"):
            cli.prepare_circle_records(self.manifest, self.config)
        self.assert_full_image()
        self.config["circle_config"] = str(self.root / "absent.json")
        with self.assertRaises(FileNotFoundError):
            cli.prepare_circle_records(self.manifest, self.config)

    def test_embedded_mask_does_not_require_old_configuration_json(self):
        self.attach_mask()
        self.config["circle_config"] = str(self.root / "old-machine.json")
        with patch("sys.stdout"):
            cli.prepare_circle_records(self.manifest, self.config)
        self.assertEqual(self.config["mask_mode"], "mask")
        self.assertEqual(self.manifest["default_mask_summary"]["count"], 4)

    def test_corrupt_white_mask_and_missing_original_image_do_not_fall_back(self):
        for value in ("corrupt", "white"):
            with self.subTest(value=value):
                if value == "corrupt":
                    self.mask.write_bytes(b"invalid image")
                else:
                    Image.new("L", (24, 32), 255).save(self.mask)
                config = {**self.config, "circle_params": {"default_mask": str(self.mask)}}
                with self.assertRaises(ValueError):
                    cli.prepare_circle_records(self.manifest, config)
                self.assertNotIn("mask_fallback", config)
        Image.new("L", (24, 32), 0).save(self.mask)
        self.config["circle_params"] = {"default_mask": str(self.mask)}
        Path(self.manifest["train"][0]["path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            cli.prepare_circle_records(self.manifest, self.config)

    def test_train_without_mask_prepares_full_image_before_model_creation(self):
        args = cli.build_parser().parse_args(["train", "--device", "cpu", "--output-dir", str(self.root / "train-out")])
        with patch.object(cli, "snapshot", return_value=copy.deepcopy(self.manifest)), \
                patch.object(cli, "new_model", side_effect=RuntimeError("stop-before-model")), patch("sys.stdout"), \
                self.assertRaisesRegex(RuntimeError, "stop-before-model"):
            cli.train_one(args, "CCD1")
        path, = (self.root / "train-out").glob("CCD1/*/config.json")
        config = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
        self.assert_full_image(config, manifest)

    def test_resume_with_missing_mask_prepares_full_image_before_model_creation(self):
        self.attach_mask()
        self.mask.unlink()
        config = {**self.config, "max_steps": 2, "batch_size": 1, "seed": 42, "step": 1,
                  "imagenette_dir": "unused", "backbone": "pdn_small", "model_size": "small"}
        saved = {"config": config, "manifest": self.manifest, "optimizer_state": {}, "step": 1}
        args = cli.build_parser().parse_args(["train", "--device", "cpu", "--resume", "last.pt",
                                             "--output-dir", str(self.root / "resume-out")])
        with patch.object(cli, "read_checkpoint", return_value=saved), patch("sys.stdout"), \
                patch.object(cli, "new_model", side_effect=RuntimeError("stop-before-model")), \
                self.assertWarnsRegex(RuntimeWarning, "无 mask"), self.assertRaisesRegex(RuntimeError, "stop-before-model"):
            cli.train_one(args, "CCD1")
        path, = (self.root / "resume-out").glob("CCD1/*/config.json")
        config = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
        self.assert_full_image(config, manifest)

    def test_evaluate_no_mask_or_missing_mask_does_not_mutate_saved_manifest(self):
        for mode in ("explicit", "missing"):
            with self.subTest(mode=mode):
                Image.new("L", (24, 32), 0).save(self.mask)
                self.attach_mask()
                original = copy.deepcopy(self.manifest)
                config = {**self.config, "circle_params": {"default_mask": str(self.mask)}}
                if mode == "missing":
                    self.mask.unlink()
                calibration = {"threshold": 0.75, "display_max": 1.0}
                saved = {"config": config, "manifest": self.manifest, "calibration": calibration}
                arguments = ["run.py", "evaluate", "--checkpoint", "model.pt", "--output-dir", str(self.root / mode)]
                if mode == "explicit":
                    arguments.append("--no-mask")
                with patch("sys.argv", arguments), patch.object(cli, "load_runtime"), patch("sys.stdout"), \
                        patch.object(cli, "restore_for_inference", return_value=(object(), dict(config), saved)), \
                        patch.object(cli, "evaluate_records", return_value={}) as evaluate, \
                        patch.object(cli, "calibrate") as calibrate, self.assertWarns(RuntimeWarning):
                    cli.main()
                calibrate.assert_not_called()
                self.assertIs(evaluate.call_args.args[3], calibration)
                self.assertEqual(self.manifest, original)
                path, = (self.root / mode).glob("evaluation/CCD1/*/config.json")
                config = json.loads(path.read_text(encoding="utf-8"))
                manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
                self.assert_full_image(config, manifest)

    def test_checkpoint_evaluation_does_not_require_old_training_or_validation_images(self):
        self.attach_mask()
        for split in ("train", "val", "threshold_val"):
            Path(self.manifest[split][0]["path"]).unlink()
        config = {**self.config, "circle_params": {"default_mask": str(self.mask)}}
        with patch("sys.stdout"):
            cli.prepare_circle_records(self.manifest, config, splits=("test",))
        self.assertEqual(config["mask_mode"], "mask")
        self.assertIn("circle", self.manifest["test"][0])

    def test_explicit_score_recalibrates_using_no_mask_records(self):
        self.attach_mask()
        defect = dict(self.manifest["test"][0])
        self.manifest["threshold_val"].append(defect)
        config = {**self.config, "circle_params": {"default_mask": str(self.mask)}}
        saved = {"config": config, "manifest": self.manifest,
                 "calibration": {"threshold": 0.75, "display_max": 1.0}}
        calibration = {"threshold": 0.5, "display_max": 1.0,
                       "score_method": {"name": cli.TOP_SCORE_METHOD}}
        arguments = ["run.py", "evaluate", "--checkpoint", "model.pt", "--no-mask", "--score-mode", "top",
                     "--output-dir", str(self.root / "recalibrate")]
        with patch("sys.argv", arguments), patch.object(cli, "load_runtime"), patch("sys.stdout"), \
                patch.object(cli, "restore_for_inference", return_value=(object(), dict(config), saved)), \
                patch.object(cli, "calibrate", return_value=calibration) as calibrate, \
                patch.object(cli, "save_checkpoint") as save, patch.object(cli, "evaluate_records", return_value={}), \
                self.assertWarnsRegex(RuntimeWarning, "--no-mask"):
            cli.main()
        self.assert_full_image(calibrate.call_args.args[2], calibrate.call_args.args[1])
        self.assert_full_image(save.call_args.args[2], save.call_args.args[3])


if __name__ == "__main__":
    unittest.main()
