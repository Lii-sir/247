"""用合成图和随机教师权重验证程序链路，不代表真实数据上的检测效果。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from efficientad_ccd import load_runtime, new_model


def main() -> None:
    """通过真正的命令行验证训练、恢复、评估和预测，不访问外网。"""
    load_runtime()
    rng = np.random.default_rng(2026)
    with TemporaryDirectory(prefix="efficientad_smoke_") as temporary:
        root = Path(temporary)
        data = root / "data"
        for folder, count in [("train/good", 8), ("test/good", 2), ("test/defect", 2)]:
            destination = data / "CCD1" / folder
            destination.mkdir(parents=True)
            for index in range(count):
                pixels = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(destination / f"{index}.png")
        auxiliary = root / "auxiliary" / "dummy_class"
        auxiliary.mkdir(parents=True)
        Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(auxiliary / "0.png")
        config = {"imagenette_dir": str(auxiliary.parent), "model_size": "small", "lr": 1e-4,
                  "weight_decay": 1e-5, "device": "cpu"}
        teacher = new_model(config)
        teacher_path = root / "random_teacher_for_smoke_only.pth"
        torch.save(teacher.model.teacher.state_dict(), teacher_path)
        del teacher
        output = root / "outputs"
        entry = [sys.executable, str(PROJECT_DIR / "efficientad_ccd.py")]

        def run(*arguments: str) -> None:
            subprocess.run(entry + list(arguments), cwd=PROJECT_DIR, check=True)

        run("train", "--data-root", str(data), "--max-steps", "2", "--min-age-seconds", "0",
            "--teacher-weights", str(teacher_path), "--imagenette-dir", str(auxiliary.parent),
            "--output-dir", str(output), "--save-every", "1", "--heatmaps", "1")
        model_path = next(output.glob("CCD1/*/model.pt"))
        run_dir = model_path.parent
        for name in ("manifest.json", "config.json", "loss.csv", "calibration.json", "metrics.json",
                     "predictions.csv", "score_distribution.png", "checkpoints/last.pt"):
            assert (run_dir / name).is_file(), name
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["counts"] == {"total": 4, "normal": 2, "anomaly": 2}
        assert len(list((run_dir / "heatmaps").rglob("*.png"))) == 1
        run("evaluate", "--checkpoint", str(model_path), "--output-dir", str(output), "--heatmaps", "-1")
        evaluated_path = next(output.glob("evaluation/CCD1/*/metrics.json"))
        # 全部测试图应按真实标签归档，每类两张，误判也不改变存放类别。
        for label_directory in ("good", "defect"):
            assert len(list((evaluated_path.parent / "heatmaps" / label_directory).glob("*.png"))) == 2
        evaluated = json.loads(evaluated_path.read_text(encoding="utf-8"))
        assert abs(evaluated["threshold"] - metrics["threshold"]) < 1e-8
        assert evaluated["confusion_matrix"] == metrics["confusion_matrix"]
        run("predict", "--checkpoint", str(model_path), "--image", str(data / "CCD1/test/good/0.png"),
            "--output-dir", str(root / "predictions"))
        assert len(list((root / "predictions").glob("CCD1/*/prediction.png"))) == 1
        # 仅在合成验证中将目标步数延长一步，确认恢复后确实执行一次优化。
        resume_payload = torch.load(run_dir / "checkpoints/last.pt", map_location="cpu", weights_only=True)
        resume_payload["config"]["max_steps"] = 3
        resume_path = root / "resume_for_smoke_only.pt"
        torch.save(resume_payload, resume_path)
        run("train", "--resume", str(resume_path),
            "--output-dir", str(root / "resumed"), "--heatmaps", "0")
        resumed_path = next((root / "resumed").glob("CCD1/*/model.pt"))
        assert torch.load(resumed_path, map_location="cpu", weights_only=True)["step"] == 3
        print("通过：合成数据的训练、校准、保存/加载、独立评估、热图、单图预测和断点恢复。")
        print("注意：该验证使用随机教师，仅检查程序链路，不用于判断 EfficientAD 效果。")


if __name__ == "__main__":
    main()
