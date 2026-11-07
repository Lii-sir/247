"""Synthetic-only checks of benchmark pairing, isolation and CLI reports."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmark_affine.py"
spec = importlib.util.spec_from_file_location("affine_benchmark", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
np, cv2 = bench.np, bench.cv2


class AffineBenchmarkTests(unittest.TestCase):
    def test_variants_are_isolated(self):
        original = dict(score_mode="full", ftol=.01, calibration=dict(enabled=True, profile={}),
                        multiscale=dict(enabled=True), maxfev=33)
        before = json.dumps(original)
        baseline = bench.parameters(original, "baseline")
        cache = bench.parameters(original, "cache")
        multi = bench.parameters(original, "multiscale")
        budget = bench.parameters(original, "budget")
        self.assertFalse(baseline["cache_enabled"])
        self.assertTrue(cache["cache_enabled"])
        self.assertFalse(cache["multiscale"]["enabled"])
        self.assertTrue(multi["multiscale"]["enabled"])
        self.assertIsNone(multi["total_maxfev"])
        self.assertEqual(150, budget["total_maxfev"])
        self.assertEqual(.01, baseline["ftol"])
        self.assertEqual(before, json.dumps(original))
        with self.assertRaises(ValueError):
            bench.parameters(original, "calibration")
        with self.assertRaises(ValueError):
            bench.parameters(dict(score_mode="roi"), "current")

    def test_pairing_rejects_duplicates_and_board_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            depth = folder / "2.png"
            depth.touch()
            images = folder / "images"
            images.mkdir()
            (images / "2_1.png").touch()
            self.assertEqual(1, bench.select_inputs(images, depth, [1], 2)[0][0])
            with self.assertRaises(ValueError):
                bench.select_inputs(images, depth, None, 2)
            other = folder / "3.png"
            other.touch()
            with self.assertRaises(ValueError):
                bench.select_inputs(images, other, [1], 2)
            (images / "001.png").touch()
            with self.assertRaises(ValueError):
                bench.select_inputs(images, depth, [1], 2)

    def test_displacement_requires_real_matrices(self):
        identity = [[1, 0, 0], [0, 1, 0]]
        moved = [[1, 0, 3], [0, 1, 4]]
        self.assertEqual(5, bench.displacement(identity, moved, [128, 96]))
        self.assertIsNone(bench.displacement(None, identity, [128, 96]))

    def test_synthetic_cli_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            rng = np.random.default_rng(123)
            mask = (rng.random((96, 128)) > .5).astype(np.uint8)
            image = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
            for pid in (1, 2):
                cv2.imwrite(str(images / f"2_{pid}.png"), image)
            depth = root / "2.png"
            cv2.imwrite(str(depth), np.tile(mask.astype(np.uint16)*33000, (2, 1)))
            config = {
                "segmentation_3d": {"parameters": dict(expected_piece_count=2, mode="fixed",
                    fixed_rois=[[0, 0, 128, 96], [0, 96, 128, 192]])},
                "registration_3d": {"parameters": dict(scale=1, roi_3d=[2, 2, 30, 40],
                    search_2d_scaled=[0, 0, 128, 96], morphology_kernel_size=1)},
                "dynamic_affine_3d": {"parameters": dict(preview_scale=1, score_mode="full",
                    morphology_kernel_size=1, rois=[[0, 0, 60, 96], [60, 0, 128, 96]],
                    maxiter=1, maxfev=12, translation_delta=[2, 2])}}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            before = config_path.read_bytes()
            output = root / "output"
            args = [sys.executable, "-B", str(SCRIPT), "--images-dir", str(images),
                    "--depth", str(depth), "--config", str(config_path), "--workers", "1", "2",
                    "--variants", "current", "baseline", "cache", "--repeats", "2", "--warmup", "1",
                    "--output-dir", str(output)]
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run(args, capture_output=True, text=True, timeout=90, env=env)
            self.assertEqual(0, result.returncode, result.stdout+result.stderr)
            report_path = next(output.glob("run-*/results.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["complete"])
            self.assertFalse(report["environment"]["torch_loaded"])
            self.assertEqual(6, len(report["groups"]))
            self.assertEqual(before, config_path.read_bytes())
            self.assertEqual(6, len(report["summary"]))
            for group in report["groups"]:
                self.assertEqual(2, len(group["rounds"]))
                for repeat in group["rounds"]:
                    self.assertEqual([1, 2], [r["piece_id"] for r in repeat["pieces"]])
                    self.assertTrue(all("affine_error" not in row["registration"] for row in repeat["pieces"]))
            for summary in report["summary"]:
                self.assertEqual(0., summary["max_reference_displacement_px"])
                self.assertEqual(0., summary["max_reference_nfev_difference"])
            self.assertTrue(report_path.with_name("pieces.csv").exists())
            self.assertTrue(report_path.with_name("analysis.md").exists())
            dry = subprocess.run(args+["--dry-run"], capture_output=True, text=True, timeout=90, env=env)
            self.assertEqual(0, dry.returncode, dry.stdout+dry.stderr)
            reports = [json.loads(p.read_text(encoding="utf-8")) for p in output.glob("run-*/results.json")]
            dry_report = next(r for r in reports if r.get("dry_run"))
            self.assertEqual([], dry_report["groups"])


if __name__ == "__main__":
    unittest.main()
