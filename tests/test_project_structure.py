"""目录迁移、统一入口和旧入口兼容回归；不改动生产数据或权重。"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from ccd_efficientad import cli, inference
from ccd_efficientad.__main__ import main
from ccd_efficientad.paths import PROJECT_DIR


class ProjectStructureTests(unittest.TestCase):
    def test_default_paths_remain_at_project_root(self):
        self.assertEqual(PROJECT_DIR, Path(__file__).resolve().parents[1])
        args = cli.build_parser().parse_args(["train"])
        self.assertEqual(args.output_dir, PROJECT_DIR / "outputs")
        self.assertEqual(args.assets_dir, PROJECT_DIR / "assets")
        args = inference.build_parser().parse_args(["--checkpoint", "unused.pt", "--image", "unused.png"])
        self.assertEqual(args.output_dir, PROJECT_DIR / "outputs" / "inference")

    def test_legacy_imports_alias_real_module_instead_of_copying_globals(self):
        self.assertIs(importlib.import_module("efficientad_ccd"), cli)
        self.assertIs(importlib.import_module("infer_efficientad"), inference)

    def test_help_does_not_import_pytorch_and_unknown_command_fails(self):
        with patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["--help"]), 0)
        self.assertIn("make-dataset", output.getvalue())
        with patch("sys.stderr", new_callable=StringIO):
            self.assertEqual(main(["unknown-command"]), 2)
        script = (
            f"import sys; sys.path.insert(0, {str(PROJECT_DIR)!r}); "
            "from ccd_efficientad.__main__ import main; main(['--help']); "
            "assert 'torch' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-I", "-c", script],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_router_preserves_cli_subcommand_and_restores_argv(self):
        old = sys.argv
        with patch.object(cli, "main") as action:
            action.side_effect = lambda: self.assertEqual(sys.argv[1:], ["inspect", "--category", "CCD2"])
            self.assertEqual(main(["inspect", "--category", "CCD2"]), 0)
        self.assertIs(sys.argv, old)
        with patch.object(inference, "main") as action:
            action.side_effect = lambda: self.assertEqual(sys.argv[1:], ["--help"])
            self.assertEqual(main(["infer", "--help"]), 0)
        self.assertIs(sys.argv, old)

    def test_router_preserves_return_codes_and_does_not_swallow_exceptions(self):
        from tools.data import make_anomaly_dataset
        with patch.object(make_anomaly_dataset, "main", return_value=2):
            self.assertEqual(main(["make-dataset"]), 2)
        previous = sys.argv
        with patch.object(inference, "main", side_effect=RuntimeError("expected")), \
                self.assertRaisesRegex(RuntimeError, "expected"):
            main(["infer"])
        self.assertIs(sys.argv, previous)

    def test_absolute_entry_from_other_cwd_and_legacy_cli_help(self):
        with TemporaryDirectory() as directory:
            env = {**os.environ, "PYTHONUTF8": "1"}
            for script, arguments in (
                ("run.py", ["--help"]),
                ("efficientad_ccd.py", ["train", "--help"]),
                ("infer_efficientad.py", ["--help"]),
                ("tools/masks/generate_circle_mask.py", ["--help"]),
                ("tools/data/make_anomaly_dataset.py", ["--help"]),
            ):
                with self.subTest(script=script):
                    result = subprocess.run([sys.executable, str(PROJECT_DIR / script), *arguments],
                                            cwd=directory, env=env, text=True, encoding="utf-8",
                                            capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("--help", result.stdout)

    def test_data_tool_via_unified_entry_still_needs_no_torch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "good"
            source.mkdir()
            for index in range(10):
                Image.new("RGB", (32, 32), (index * 10, 90, 160)).save(source / f"{index}.png")
            result = subprocess.run([
                sys.executable, "-I", str(PROJECT_DIR / "run.py"), "make-dataset",
                "--input-dir", str(source), "--output-dir", str(root / "dataset"),
                "--train-ratio", "0.6", "--test-good-ratio", "0.2", "--test-ng-ratio", "0.2",
            ], cwd=directory, env={**os.environ, "PYTHONUTF8": "1"}, text=True,
                encoding="utf-8", capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((root / "dataset" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"], {"train/good": 6, "test/good": 2, "test/ng": 2})


if __name__ == "__main__":
    unittest.main()
