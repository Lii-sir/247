"""用合成图验证不同 backbone 的程序链路，不代表真实数据上的检测效果。"""

from __future__ import annotations

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("pdn_small", "pdn_medium", "resnet18_layer2", "resnet18_layer3",
                                               "resnet50_layer1", "resnet50_layer2", "resnet50_layer3"), default="pdn_small")
    parser.add_argument("--ddp-test-device", choices=("cpu", "cuda:0"),
                        help="测试专用：在同一设备启动两个真实 DDP 进程，不代表两张物理卡的性能")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2,
                        help="每个 rank 的合成测试 batch；大模型 DDP 可用 1 降低显存占用")
    args = parser.parse_args()
    load_runtime()
    torch.manual_seed(2026)
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
        for index in range(4 if args.ddp_test_device else 2):
            Image.fromarray(
                rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
            ).save(auxiliary / f"{index}.png")
        config = {"imagenette_dir": str(auxiliary.parent), "model_size": "small", "lr": 1e-4,
                  "weight_decay": 1e-5, "device": "cpu", "backbone": args.backbone,
                  "resnet_architecture_version": 2}
        teacher = new_model(config)
        if args.backbone.startswith("resnet"):
            from torchvision.models import ResNet18_Weights, ResNet50_Weights
            from self_efficientad.backbones import load_default_teacher_weights

            weights = (ResNet50_Weights.IMAGENET1K_V2 if args.backbone.startswith("resnet50_")
                       else ResNet18_Weights.IMAGENET1K_V1)
            cached = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
            if not cached.is_file():
                raise FileNotFoundError(f"ResNet smoke 需要已缓存的 torchvision 教师权重：{cached}")
            load_default_teacher_weights(args.backbone, teacher.model.teacher)
        teacher_path = root / "teacher_for_smoke_only.pth"
        torch.save(teacher.model.teacher.state_dict(), teacher_path)
        del teacher
        default_mask = root / "default_mask.png"
        Image.new("L", (256, 256), 0).save(default_mask)
        circle_config = root / "circle_config.json"
        circle_config.write_text(json.dumps({
            "CCD1": {"default_mask": str(default_mask)},
        }), encoding="utf-8")
        output = root / "outputs"
        entry = [sys.executable, str(PROJECT_DIR / "efficientad_ccd.py")]
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        if args.ddp_test_device:
            entry = [sys.executable, str(PROJECT_DIR / "tests/distributed_cli_entry.py")]
            environment["CCD_DDP_TEST_DEVICE"] = args.ddp_test_device

        def run(*arguments: str) -> None:
            subprocess.run(entry + list(arguments), cwd=PROJECT_DIR, check=True, env=environment)

        global_batch_size = args.batch_size * (2 if args.ddp_test_device else 1)
        run("train", "--backbone", args.backbone, "--data-root", str(data), "--batch-size", str(args.batch_size),
            "--max-images", str(2 * global_batch_size),
            "--num-workers", str(args.num_workers),
            "--min-age-seconds", "0",
            "--teacher-weights", str(teacher_path), "--imagenette-dir", str(auxiliary.parent),
            "--circle-config", str(circle_config), "--output-dir", str(output),
            "--save-every", "1", "--heatmaps", "1")
        model_path = next(output.glob("CCD1/*/model.pt"))
        run_dir = model_path.parent
        assert json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["backbone"] == args.backbone
        training_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert training_config["world_size"] == (2 if args.ddp_test_device else 1)
        assert training_config["global_batch_size"] == global_batch_size
        assert training_config["max_steps"] == 2
        assert training_config["resnet_architecture_version"] == 2
        for name in ("manifest.json", "config.json", "loss.csv", "calibration.json", "metrics.json",
                     "predictions.csv", "score_distribution.png", "checkpoints/last.pt"):
            assert (run_dir / name).is_file(), name
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        # test/good 与 test/defect 各有一张划入 threshold_val，剩余各一张用于最终测试。
        assert metrics["counts"] == {"total": 2, "normal": 1, "anomaly": 1}
        assert metrics["calibration"]["score_method"]["pool_kernels"] == [1, 7, 21]
        assert len(list((run_dir / "heatmaps").rglob("*.png"))) == 1
        assert len(list((run_dir / "heatmap_scales").rglob("*.png"))) == 1
        run("evaluate", "--checkpoint", str(model_path), "--output-dir", str(output), "--heatmaps", "-1")
        evaluated_path = next(output.glob("evaluation/CCD1/*/metrics.json"))
        # 最终测试图应按真实标签和预测结果两级归档。
        saved_heatmaps = list((evaluated_path.parent / "heatmaps").rglob("*.png"))
        assert len(saved_heatmaps) == 2, saved_heatmaps
        assert len(list((evaluated_path.parent / "heatmap_scales").rglob("*.png"))) == 2
        for label_directory in ("good", "defect"):
            paths = [path for path in saved_heatmaps if path.relative_to(evaluated_path.parent / "heatmaps").parts[0] == label_directory]
            assert len(paths) == 1, (label_directory, saved_heatmaps)
        evaluated = json.loads(evaluated_path.read_text(encoding="utf-8"))
        assert abs(evaluated["threshold"] - metrics["threshold"]) < 1e-8
        assert evaluated["confusion_matrix"] == metrics["confusion_matrix"]

        # 同一模型显式切换三种整图 score；每种方式都应重新校准、完成评估，
        # 并将与 threshold 匹配的 score 定义保存到新的 model.pt。
        score_cases = [
            ("top", [], "masked_pixel_max"),
            (
                "pool+top",
                ["--score-pool-kernel", "3", "--score-topk-ratio", "0.01"],
                "masked_local_average_topk_mean",
            ),
            (
                "multiscale_pool",
                ["--score-pool-kernels", "1,3", "--score-topk-ratio", "0.01"],
                "masked_multiscale_normalized_topk_max",
            ),
        ]
        known_evaluations = set(output.glob("evaluation/CCD1/*"))
        for score_mode, extra_arguments, expected_method in score_cases:
            run(
                "evaluate", "--checkpoint", str(model_path), "--output-dir", str(output),
                "--score-mode", score_mode, "--heatmaps", "0", *extra_arguments,
            )
            new_evaluations = set(output.glob("evaluation/CCD1/*")) - known_evaluations
            assert len(new_evaluations) == 1, (score_mode, new_evaluations)
            score_output = new_evaluations.pop()
            known_evaluations.add(score_output)
            for name in ("model.pt", "manifest.json", "config.json", "calibration.json", "metrics.json"):
                assert (score_output / name).is_file(), (score_mode, name)
            score_checkpoint = torch.load(
                score_output / "model.pt", map_location="cpu", weights_only=True
            )
            assert score_checkpoint["calibration"]["score_method"]["name"] == expected_method
            assert score_checkpoint["calibration"]["threshold"] == json.loads(
                (score_output / "metrics.json").read_text(encoding="utf-8")
            )["threshold"]
            switched_prediction_output = root / f"prediction_{score_mode.replace('+', '_')}"
            run(
                "predict", "--checkpoint", str(score_output / "model.pt"),
                "--image", str(data / "CCD1/test/good/0.png"),
                "--output-dir", str(switched_prediction_output),
            )
            switched_predictions = list(
                switched_prediction_output.glob("CCD1/*/prediction.json")
            )
            assert len(switched_predictions) == 1, (score_mode, switched_predictions)
            switched_result = json.loads(
                switched_predictions[0].read_text(encoding="utf-8")
            )
            assert switched_result["threshold"] == score_checkpoint["calibration"]["threshold"]
            assert switched_result["localization"]["score_mode"] == score_checkpoint["calibration"]["score_mode"]
            prediction_folder = switched_predictions[0].parent
            assert (prediction_folder / "prediction.png").is_file()
            assert (prediction_folder / "prediction_scales.png").is_file() == (score_mode == "multiscale_pool")

        run("predict", "--checkpoint", str(model_path), "--image", str(data / "CCD1/test/good/0.png"),
            "--output-dir", str(root / "predictions"))
        assert len(list((root / "predictions").glob("CCD1/*/prediction.png"))) == 1
        assert len(list((root / "predictions").glob("CCD1/*/prediction_scales.png"))) == 1
        # 仅在合成验证中将目标步数延长一步，确认恢复后确实执行一次优化。
        resume_payload = torch.load(run_dir / "checkpoints/last.pt", map_location="cpu", weights_only=True)
        resume_payload["config"]["max_steps"] = 3
        resume_path = root / "resume_for_smoke_only.pt"
        torch.save(resume_payload, resume_path)
        run("train", "--resume", str(resume_path),
            "--num-workers", str(args.num_workers),
            "--output-dir", str(root / "resumed"), "--heatmaps", "0")
        resumed_path = next((root / "resumed").glob("CCD1/*/model.pt"))
        resumed = torch.load(resumed_path, map_location="cpu", weights_only=True)
        assert resumed["step"] == 3
        assert resumed["config"]["backbone"] == args.backbone
        assert resumed["config"]["resnet_architecture_version"] == 2
        print(f"通过：{args.backbone} 合成数据的训练、校准、保存/加载、独立评估、热图、单图预测和断点恢复。")
        print("注意：该验证仅检查程序链路，不用于判断真实数据上的检测效果。")


if __name__ == "__main__":
    main()
