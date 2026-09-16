"""验证命令入口的数据隔离和检查点约束；不下载模型或运行神经网络。"""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

import efficientad_ccd as cli


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.calibration = {"threshold": 0.75, "display_max": 1.0}
        self.saved = {
            "manifest": {
                "category": "CCD1",
                "train": [{"sha256": "trained-content"}],
                "val": [{"sha256": "calibration-content"}],
                "test": [],
            },
            "calibration": self.calibration,
        }

    def evaluate_manifest(self, manifest: dict):
        """运行真实参数解析和评估分支，仅替换耗时的模型加载与前向计算。"""
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        arguments = ["efficientad_ccd.py", "evaluate", "--checkpoint", "unused.pt",
                     "--manifest", str(path), "--output-dir", str(self.root / "results")]
        with patch("sys.argv", arguments), patch.object(cli, "load_runtime"), \
                patch.object(cli, "restore_for_inference", return_value=(object(), {}, self.saved)), \
                patch.object(cli, "evaluate_records", return_value={}) as evaluate:
            cli.main()
            return evaluate

    def test_evaluation_rejects_wrong_camera(self) -> None:
        with self.assertRaisesRegex(ValueError, "类别.*不一致"):
            self.evaluate_manifest({"category": "CCD2", "test": []})
        self.assertFalse((self.root / "results").exists())

    def test_evaluation_keeps_renamed_training_and_calibration_images(self) -> None:
        # 评估不再因内容与训练/校准样本重复而拒绝。
        for digest in ("trained-content", "calibration-content"):
            with self.subTest(digest=digest):
                evaluate = self.evaluate_manifest({
                    "category": "CCD1",
                    "test": [{"path": "new-folder/renamed.png", "sha256": digest}],
                })
                self.assertEqual(evaluate.call_args.args[1][0]["sha256"], digest)
        self.assertTrue((self.root / "results").exists())

    def test_new_test_manifest_keeps_original_threshold(self) -> None:
        records = [{"path": "new-test.png", "sha256": "previously-unseen", "label": 1}]
        evaluate = self.evaluate_manifest({"category": "CCD1", "test": records})
        self.assertEqual(evaluate.call_args.args[1], records)
        self.assertIs(evaluate.call_args.args[3], self.calibration)
        manifests = list((self.root / "results").glob("evaluation/CCD1/*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        self.assertEqual(json.loads(manifests[0].read_text(encoding="utf-8"))["test"], records)

    def test_inference_rejects_checkpoint_without_calibration(self) -> None:
        args = SimpleNamespace(checkpoint=self.root / "last.pt")
        with patch.object(cli, "read_checkpoint", return_value={"calibration": None}), \
                patch.object(cli, "new_model") as constructor:
            with self.assertRaisesRegex(ValueError, "尚未校准"):
                cli.restore_for_inference(args)
        constructor.assert_not_called()

    def test_resume_rejects_inference_only_model(self) -> None:
        args = cli.build_parser().parse_args(["train", "--resume", "model.pt"])
        with patch.object(cli, "read_checkpoint", return_value={"calibration": self.calibration}), \
                patch.object(cli, "new_output") as output:
            with self.assertRaisesRegex(ValueError, "last.pt"):
                cli.train_one(args, "CCD1")
        output.assert_not_called()

    def test_training_reports_all_duplicates_before_starting(self) -> None:
        report = {
            "hash_type": "sha256_file_bytes",
            "group_count": 1,
            "file_count": 3,
            "extra_copy_count": 2,
            "cross_train_test_group_count": 1,
            "label_conflict_group_count": 1,
            "groups": [{
                "sha256": "same-content",
                "file_count": 3,
                "extra_copy_count": 2,
                "splits": ["test", "train"],
                "cross_train_test": True,
                "label_conflict": True,
                "files": [
                    {"path": "train-a.png", "split": "train", "label": 0, "defect_type": "good"},
                    {"path": "test-a.png", "split": "test", "label": 1, "defect_type": "defect"},
                    {"path": "test-b.png", "split": "test", "label": 1, "defect_type": "defect"},
                ],
            }],
        }
        manifest = {"category": "CCD1", "duplicate_report": report, "summary": {}}
        args = SimpleNamespace(resume=None, output_dir=self.root / "training")
        with patch.object(cli, "snapshot", return_value=manifest), \
                self.assertRaisesRegex(ValueError, "训练未启动"):
            cli.train_one(args, "CCD1")
        reports = list((self.root / "training").glob("CCD1/*/duplicate_report.json"))
        manifests = list((self.root / "training").glob("CCD1/*/manifest.json"))
        self.assertEqual(len(reports), 1)
        self.assertEqual(len(manifests), 1)
        self.assertEqual(json.loads(reports[0].read_text(encoding="utf-8"))["groups"], report["groups"])

    def test_inspect_all_keeps_successful_camera_when_another_is_incomplete(self) -> None:
        data = self.root / "dataset"
        complete = data / "CCD1/train/good"
        complete.mkdir(parents=True)
        for index in range(8):
            Image.new("RGB", (8, 8), (index, 0, 0)).save(complete / f"{index}.png")
        (data / "CCD2/train/good").mkdir(parents=True)
        output = self.root / "inspections"
        arguments = ["efficientad_ccd.py", "inspect", "--data-root", str(data),
                     "--category", "all", "--min-age-seconds", "0", "--output-dir", str(output)]
        with patch("sys.argv", arguments), self.assertRaisesRegex(ValueError, "CCD2"):
            cli.main()
        manifests = list(output.glob("inspection/*/*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertEqual(manifest["category"], "CCD1")
        self.assertEqual(manifest["summary"]["train"] + manifest["summary"]["val"], 8)

    def test_all_requires_at_least_one_camera(self) -> None:
        # 下载只建立了数据根目录时，不能以成功退出造成已经检查或训练的错觉。
        data = self.root / "empty_dataset"
        data.mkdir()
        for command in ("inspect", "train"):
            arguments = ["efficientad_ccd.py", command, "--data-root", str(data), "--category", "all"]
            with self.subTest(command=command), patch("sys.argv", arguments), \
                    patch.object(cli, "load_runtime"), self.assertRaises(ValueError):
                cli.main()

    def test_heatmap_directory_component_preserves_normal_names(self) -> None:
        self.assertEqual(cli._safe_output_component("good"), "good")
        self.assertEqual(cli._safe_output_component("严重缺陷"), "严重缺陷")
        self.assertEqual(cli._safe_output_component("bad/type"), "bad_type")

    def test_prediction_uses_saved_threshold_with_strict_greater_than(self) -> None:
        image = self.root / "input.png"
        Image.new("RGB", (8, 8), (10, 20, 30)).save(image)
        for score, expected in ((0.75, "OK"), (0.7501, "NG")):
            with self.subTest(score=score):
                model = MagicMock()
                model.model.return_value.pred_score.flatten.return_value = [score]
                fake_report = SimpleNamespace(save_heatmap=MagicMock())
                output = self.root / f"prediction-{score}"
                args = cli.build_parser().parse_args([
                    "predict", "--checkpoint", "model.pt", "--image", str(image),
                    "--output-dir", str(output),
                ])
                with patch.object(cli, "restore_for_inference", return_value=(model, {"device": "cpu"}, self.saved)), \
                        patch.object(cli, "make_loader", return_value=[SimpleNamespace(image=MagicMock())]), \
                        patch.object(cli, "torch", SimpleNamespace(inference_mode=nullcontext), create=True), \
                        patch.dict("sys.modules", {"ccd_report": fake_report}):
                    cli.predict(args)
                paths = list(output.glob("CCD1/*/prediction.json"))
                self.assertEqual(len(paths), 1)
                result = json.loads(paths[0].read_text(encoding="utf-8"))
                self.assertEqual(result["prediction"], expected)
                self.assertEqual(result["threshold"], 0.75)
                fake_report.save_heatmap.assert_called_once()


if __name__ == "__main__":
    unittest.main()
