"""CCD 数据集的 EfficientAD 训练、正常样本校准、评估与推理入口。"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from circle_mask import (
    category_config,
    default_mask_record,
    load_configs,
    mask_from_circle_record,
    mask_score,
    read_rgb,
)
from ccd_data import list_categories, prepare_manifest, resolve_data_root

PROJECT_DIR = Path(__file__).resolve().parent
Batch = namedtuple("Batch", ["image", "gt_label", "ignore_mask"])


def write_json(path: Path, data: dict) -> None:
    """使用 UTF-8 保存记录，保留中文目录名。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_runtime() -> None:
    """仅训练与推理时加载深度学习库，使数据检查命令启动更快。"""
    global torch, np, EfficientAd, DataLoader, Image, TF, tqdm
    import numpy as np
    import torch
    from anomalib.models import EfficientAd
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision.transforms import functional as TF
    from tqdm import tqdm


class SnapshotDataset:
    """始终读取快照中的文件；下载继续进行时，不自动吸收新图片。"""

    def __init__(self, records: list[dict], image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        # Windows 多进程 DataLoader 会重新导入模块，因此依赖在这里也要可用。
        import numpy as np
        from PIL import Image, ImageOps
        from torchvision.transforms import functional as TF

        record = self.records[index]
        path = Path(record["path"])
        stat = path.stat()
        if stat.st_size != record["size_bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise RuntimeError(f"快照中的图片已改变，请重新 inspect/train：{path}")
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        # 模型输入保持原始图像，不在输入阶段应用圆形白色填充。
        # 圆 mask 只在输出异常图、score 和校准统计时使用。
        circle = record.get("circle")
        if circle:
            mask = mask_from_circle_record((image.height, image.width), circle)
        else:
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
        resized_mask = Image.fromarray(mask).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        # EfficientAD 在网络内部归一化；输入必须是 [0, 1]，不能再用 Normalize。
        return TF.to_tensor(image), record["label"], TF.to_tensor(resized_mask) > 0.5


def collate_batch(samples: list) -> Batch:
    """保持官方统计函数所需的 image / gt_label 属性。"""
    import torch

    images, labels, masks = zip(*samples, strict=True)
    return Batch(torch.stack(images), torch.tensor(labels, dtype=torch.int64), torch.stack(masks).bool())


def make_loader(records: list[dict], config: dict, *, shuffle: bool = False):
    return DataLoader(
        SnapshotDataset(records, config["image_size"]),
        batch_size=1,  # 官方 EfficientAD 的训练批量固定为 1。
        shuffle=shuffle,
        num_workers=config["num_workers"],
        collate_fn=collate_batch,
        pin_memory=config["device"].startswith("cuda"),
        persistent_workers=config["num_workers"] > 0,
    )


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 CUDA，请运行 README 中的 GPU 检查命令。")
    return "cuda:0" if requested == "cuda" else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def new_model(config: dict):
    # 直接调用官方模型与辅助函数，训练循环由本脚本控制，学习率按 step 更新。
    return EfficientAd(
        imagenet_dir=config["imagenette_dir"],
        model_size=config["model_size"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    ).to(config["device"])


def prepare_assets(model, config: dict, *, load_teacher: bool) -> None:
    """复用官方带 SHA256 校验的下载器，也支持手工提供辅助数据。"""
    from anomalib.data.utils import download_and_extract
    from anomalib.models.image.efficient_ad.lightning_model import WEIGHTS_DOWNLOAD_INFO

    if load_teacher:
        teacher_path = config.get("teacher_weights")
        if teacher_path:
            teacher_path = Path(teacher_path)
            if not teacher_path.is_file():
                raise FileNotFoundError(f"找不到教师权重：{teacher_path}")
        else:
            cache = Path(config["assets_dir"]) / "pre_trained"
            teacher_path = cache / "efficientad_pretrained_weights" / (
                f"pretrained_teacher_{config['model_size']}.pth"
            )
            if not teacher_path.is_file():
                cache.mkdir(parents=True, exist_ok=True)
                download_and_extract(cache, WEIGHTS_DOWNLOAD_INFO)
        model.model.teacher.load_state_dict(
            torch.load(teacher_path, map_location=config["device"], weights_only=True)
        )
    imagenette = Path(config["imagenette_dir"])
    # 仅建立父目录，空的叶目录会让官方函数误判为已经下载完成。
    imagenette.parent.mkdir(parents=True, exist_ok=True)
    model.prepare_imagenette_data((config["image_size"], config["image_size"]))
    if len(model.imagenet_loader.dataset) == 0:
        raise ValueError(f"ImageNette 辅助数据为空：{imagenette}")


def save_checkpoint(path: Path, model, config: dict, manifest: dict, step: int,
                    optimizer=None, scheduler=None, calibration: dict | None = None) -> None:
    """先写临时文件再替换，减少中断时留下半个 checkpoint 的可能性。"""
    payload = {
        "format_version": 1,
        "model_state": model.model.state_dict(),
        "config": config,
        "manifest": manifest,
        "step": step,
        "calibration": calibration,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
        payload["scheduler_state"] = scheduler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path: Path) -> dict:
    # checkpoint 只包含张量和普通 Python 数据，使用 weights_only 限制反序列化。
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("不是本脚本生成的 checkpoint，或格式版本不匹配。")
    return payload


def check_statistics(values, *, quantiles: bool = False) -> None:
    if not all(torch.isfinite(v).all() for v in values.values()):
        raise ValueError("模型统计量出现 NaN/Inf，请检查图片内容与训练损失。")
    if quantiles:
        if any((values[f"qb_{key}"] - values[f"qa_{key}"]).item() <= 0 for key in ("st", "ae")):
            raise ValueError("正常验证集的异常图分位数退化，请增加有代表性的正常图片。")
    elif (values["std"] <= 0).any():
        raise ValueError("教师特征的标准差为零，请检查是否存在大量纯色或完全相同的图片。")


def _mask_tensor(mask, device: str):
    """将新旧数据加载器返回的 mask 统一为 device 上的 bool Tensor。

    新版 collate_batch 返回 Tensor；旧版/自定义 DataLoader 可能返回 list，
    因此校准和推理入口都不能直接调用 ``.to``、``.numel`` 等 Tensor 方法。
    """
    import torch

    if mask is None:
        return None
    if not torch.is_tensor(mask):
        try:
            mask = torch.as_tensor(mask, device=device)
        except (TypeError, ValueError):
            mask = torch.stack([torch.as_tensor(item, device=device) for item in mask])
    else:
        mask = mask.to(device)
    return mask.bool()


def _mask_numpy(mask):
    """将 Tensor/list mask 转为用于可视化的 NumPy 数组。"""
    import numpy as np

    if mask is None:
        return None
    if hasattr(mask, "detach") and hasattr(mask, "cpu"):
        return mask.detach().cpu().numpy().squeeze()
    return np.asarray(mask).squeeze()


def masked_map_quantiles(model, loader, device: str) -> dict:
    """只用正常有效区域估计 EfficientAD 异常图归一化分位数。

    有效像素逐张转到 CPU 再汇总。高分辨率验证集可能超过 PyTorch
    quantile 的元素数量限制，因此最终使用 NumPy 在 CPU 上计算精确分位数。
    """
    print("校准分位数后端：NumPy/CPU（支持超过 2^24 个有效像素）。")
    maps_st, maps_ae = [], []
    with torch.inference_mode():
        for batch in loader:
            ignore_mask = _mask_tensor(getattr(batch, "ignore_mask", None), device)
            if ignore_mask is None or ignore_mask.numel() == 0:
                valid = None
            else:
                valid = ~ignore_mask
            map_st, map_ae = model.model.get_maps(batch.image.to(device), normalize=False)
            if valid is not None:
                valid = valid.expand_as(map_st)
                maps_st.append(map_st[valid].detach().cpu())
                maps_ae.append(map_ae[valid].detach().cpu())
            else:
                maps_st.append(map_st.reshape(-1).detach().cpu())
                maps_ae.append(map_ae.reshape(-1).detach().cpu())
    if not maps_st or not maps_ae:
        raise ValueError("mask 后没有可用于异常图校准的有效像素。")
    values_st = torch.cat(maps_st)
    if values_st.numel() == 0:
        raise ValueError("mask 后没有可用于异常图校准的有效像素。")
    dtype_st = values_st.dtype
    # NumPy 默认 method='linear'，与 torch.quantile 默认插值方式一致，
    # 且不受 PyTorch 对超大输入张量的 2^24 元素限制。
    quantiles_st = np.quantile(values_st.numpy(), [0.9, 0.995])
    del values_st, maps_st

    values_ae = torch.cat(maps_ae)
    if values_ae.numel() == 0:
        raise ValueError("mask 后没有可用于异常图校准的有效像素。")
    dtype_ae = values_ae.dtype
    quantiles_ae = np.quantile(values_ae.numpy(), [0.9, 0.995])
    del values_ae, maps_ae
    return {
        "qa_st": torch.tensor(quantiles_st[0], dtype=dtype_st, device=device),
        "qa_ae": torch.tensor(quantiles_ae[0], dtype=dtype_ae, device=device),
        "qb_st": torch.tensor(quantiles_st[1], dtype=dtype_st, device=device),
        "qb_ae": torch.tensor(quantiles_ae[1], dtype=dtype_ae, device=device),
    }


def prediction_score(prediction, batch, device: str) -> float:
    """由有效异常图计算 score；兼容无 mask 的旧测试/旧模型对象。"""
    anomaly_map = getattr(prediction, "anomaly_map", None)
    ignore_mask = getattr(batch, "ignore_mask", None)
    if anomaly_map is not None and ignore_mask is not None:
        return float(mask_score(anomaly_map, _mask_tensor(ignore_mask, device)).flatten()[0])
    return float(prediction.pred_score.flatten()[0])


def calibrate(model, manifest: dict, config: dict, output_dir: Path) -> dict:
    """仅用留出的正常验证图校准异常图和图像阈值，不接触测试标签。"""
    model.eval()  # 关闭自编码器 dropout 后再估计分位数。
    loader = make_loader(manifest["val"], config)
    with torch.inference_mode():
        quantiles = masked_map_quantiles(model, loader, config["device"])
        check_statistics(quantiles, quantiles=True)
        model.model.quantiles.update(quantiles)
        scores = []
        rows = []
        for record, batch in zip(manifest["val"], loader, strict=True):
            prediction = model.model(batch.image.to(config["device"]))
            score = prediction_score(prediction, batch, config["device"])
            if not math.isfinite(score):
                raise ValueError(f"校准分数不是有限数值：{record['path']}")
            scores.append(score)
            rows.append({"path": record["path"], "score": score})
    # 使用上取整分位数；样本很少时通常就是验证正常分数的最大值。
    threshold = float(np.quantile(scores, config["threshold_quantile"], method="higher"))
    calibration = {
        "method": "held_out_normal_score_quantile",
        "quantile": config["threshold_quantile"],
        "quantile_interpolation": "higher",
        "threshold": threshold,
        "decision_rule": "score > threshold",
        "normal_validation_count": len(scores),
        "validation_false_positive_count": sum(s > threshold for s in scores),
        "display_max": max(max(scores), threshold, 0.1),
        "map_quantiles": {key: float(value) for key, value in quantiles.items()},
        "notes": "分数不是概率；该分位数不保证未来数据的误报率。没有使用测试集调阈值。",
    }
    write_json(output_dir / "calibration.json", {**calibration, "validation_scores": rows})
    return calibration


def evaluate_records(model, records: list[dict], config: dict, calibration: dict,
                     output_dir: Path, heatmaps: int) -> dict:
    from ccd_report import save_heatmap, write_report

    if not records:
        print("当前快照没有测试图片，已保存模型和校准结果，跳过测试。")
        return {}
    model.eval()
    # 正式计时前预热一次，避免 CUDA 首次初始化/算子选择明显抬高平均耗时。
    warmup_batch = next(iter(make_loader([records[0]], {**config, "num_workers": 0})))
    with torch.inference_mode():
        model.model(warmup_batch.image.to(config["device"]))
        if config["device"].startswith("cuda"):
            torch.cuda.synchronize()
    rows = []
    loader = make_loader(records, config)
    inference_seconds = 0.0
    evaluation_started = time.perf_counter()
    with torch.inference_mode():
        for record, batch in tqdm(zip(records, loader, strict=True), total=len(records), desc="测试集推理"):
            image = batch.image.to(config["device"])
            if config["device"].startswith("cuda"):
                torch.cuda.synchronize()
            inference_started = time.perf_counter()
            prediction = model.model(image)
            score = prediction_score(prediction, batch, config["device"])
            if config["device"].startswith("cuda"):
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - inference_started
            if not math.isfinite(score):
                raise ValueError(f"测试分数不是有限数值：{record['path']}")
            rows.append({
                "path": record["path"], "label": record["label"], "score": score,
                "pred_label": int(score > calibration["threshold"]),
                "defect_type": record["defect_type"],
            })
    evaluation_seconds = time.perf_counter() - evaluation_started
    image_count = len(rows)
    speed = {
        "image_count": image_count,
        "warmup_runs_not_timed": 1,
        "device": config["device"],
        "model_and_score_total_seconds": inference_seconds,
        "model_and_score_average_ms": 1000.0 * inference_seconds / image_count,
        "model_and_score_fps": image_count / inference_seconds if inference_seconds > 0 else None,
        "evaluation_total_seconds": evaluation_seconds,
        "evaluation_average_ms": 1000.0 * evaluation_seconds / image_count,
        "evaluation_fps": image_count / evaluation_seconds if evaluation_seconds > 0 else None,
        "notes": (
            "model_and_score 统计设备前向与 mask 后整图 score，不含图片读取、缩放和热力图保存；"
            "evaluation_total 包含测试 DataLoader 读取与上述计算，不含报告和热力图保存。"
        ),
    }
    metrics = write_report(output_dir, rows, calibration["threshold"], calibration, speed=speed)
    write_json(output_dir / "inference_speed.json", speed)
    print(
        "检测速度："
        f"{speed['model_and_score_average_ms']:.3f} ms/张，"
        f"{speed['model_and_score_fps']:.2f} FPS；"
        f"验证端到端 {speed['evaluation_average_ms']:.3f} ms/张，"
        f"{speed['evaluation_fps']:.2f} FPS。"
    )
    # 先保证预测为 anomaly 的图片进入可视化，再在剩余名额中选择
    # 误判/接近阈值的 normal 图片，避免默认 heatmaps=32 时异常目录为空。
    def visualization_key(index: int) -> tuple[bool, float]:
        row = rows[index]
        return row["label"] == row["pred_label"], abs(row["score"] - calibration["threshold"])

    anomaly_indices = sorted((i for i, row in enumerate(rows) if row["pred_label"]), key=visualization_key)
    normal_indices = sorted((i for i, row in enumerate(rows) if not row["pred_label"]), key=visualization_key)
    indices = anomaly_indices + normal_indices
    indices = indices if heatmaps < 0 else indices[:heatmaps]
    with torch.inference_mode():
        for rank, index in enumerate(tqdm(indices, desc="保存热力图")):
            record, row = records[index], rows[index]
            batch = next(iter(make_loader([record], {**config, "num_workers": 0})))
            anomaly_map = model.model(batch.image.to(config["device"])).anomaly_map.squeeze().cpu().numpy()
            # 第一级保持 test 下的真实子目录（good、defect1、defect2 ...），
            # 第二级按整图预测结果区分 normal / anomaly。
            source_directory = _safe_output_component(record.get("defect_type", "unspecified"))
            prediction_directory = "anomaly" if row["pred_label"] else "normal"
            save_heatmap(
                Path(record["path"]), anomaly_map,
                output_dir / "heatmaps" / source_directory / prediction_directory / f"{rank:04d}_{Path(record['path']).stem}.png",
                display_max=calibration["display_max"], score=row["score"], threshold=calibration["threshold"],
                ignore_mask=_mask_numpy(getattr(batch, "ignore_mask", None)),
            )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    return metrics


def _safe_output_component(value: str) -> str:
    """将数据集子目录转换为安全的输出目录名。"""
    value = str(value).strip()
    if not value or value in {".", ".."}:
        return "unspecified"
    # defect_type 来自数据集目录名；仍防御路径分隔符和 Windows 保留字符。
    invalid = '<>:/\\|?*"'
    value = "_".join("_" if char in invalid or ord(char) < 32 else char for char in value)
    return value.rstrip(" .") or "unspecified"


def snapshot(args, category: str) -> dict:
    return prepare_manifest(
        resolve_data_root(args.data_root), category=category,
        val_ratio=args.val_ratio, seed=args.seed,
        min_age_seconds=args.min_age_seconds, verify_images=True,
    )


def report_duplicate_content(manifest: dict, output_dir: Path) -> Path | None:
    """保存并在终端完整列出重复图片组；返回报告路径。"""
    report = manifest.get("duplicate_report") or {}
    groups = report.get("groups") or []
    if not groups:
        print("重复内容检查：未发现 SHA256 相同的图片。")
        return None

    report_path = output_dir / "duplicate_report.json"
    write_json(report_path, report)
    print(
        "重复内容检查："
        f"发现 {report['group_count']} 组，涉及 {report['file_count']} 张图片，"
        f"其中额外副本 {report['extra_copy_count']} 张；完整报告：{report_path}"
    )
    for index, group in enumerate(groups, start=1):
        flags = []
        if group.get("cross_train_test"):
            flags.append("train/test 重复")
        if group.get("label_conflict"):
            flags.append("标签冲突")
        description = "、".join(flags) if flags else "同一集合内重复"
        print(f"[{index}/{len(groups)}] {description} | sha256={group['sha256']}")
        for item in group["files"]:
            label = "anomaly" if item["label"] else "normal"
            print(f"  - {item['split']}/{item['defect_type']} | {label} | {item['path']}")
    return report_path


def prepare_circle_records(manifest: dict, config: dict) -> None:
    """按类别检测每张图的目标圆，并把结果固化到 manifest。"""
    circle_path = config.get("circle_config")
    configs = load_configs(Path(circle_path)) if circle_path else {}
    params = config.get("circle_params") or category_config(configs, manifest["category"])
    if not params:
        raise ValueError("当前模式要求所有图片使用默认 mask，请配置 --circle-config 和 default_mask。")
    # 将本次训练实际使用的参数写入 checkpoint/config，避免外部 JSON
    # 后续被修改后，单图预测与校准时的 mask 规则发生漂移。
    config["circle_params"] = params
    count = 0
    for split in ("train", "val", "test"):
        for record in manifest.get(split, []):
            image = read_rgb(Path(record["path"]))
            try:
                record["circle"] = default_mask_record(
                    image, params, base_dir=Path(circle_path).parent if circle_path else None
                )
            except ValueError as exc:
                raise ValueError(f"默认 mask 配置/读取失败（{split}）：{record['path']}；{exc}") from exc
            count += 1
    if count:
        manifest["default_mask_summary"] = {"count": count}
        print(f"默认 mask 已应用：{count} 张；当前不执行圆检测。")


def new_output(base: Path, category: str) -> Path:
    output = base.resolve() / category / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    return output


def train_one(args, category: str) -> None:
    if args.resume:
        previous = read_checkpoint(args.resume)
        if "optimizer_state" not in previous:
            raise ValueError("续训请使用 checkpoints/last.pt；model.pt 是推理用文件。")
        config, manifest = previous["config"].copy(), previous["manifest"]
        config.update(device=choose_device(args.device), num_workers=args.num_workers)
        output = new_output(args.output_dir, manifest["category"])
        print(f"从第 {previous['step']} 步续训；总步数/分辨率/划分沿用 checkpoint。")
    else:
        previous = None
        manifest = snapshot(args, category)
        output = new_output(args.output_dir, category)
        duplicate_report_path = report_duplicate_content(manifest, output)
        if duplicate_report_path is not None:
            # 先保存扫描清单和全部重复组，再停止训练，避免数据泄漏。
            write_json(output / "manifest.json", manifest)
            raise ValueError(
                "数据集中存在内容完全相同的图片，训练未启动。"
                f"请处理后重试；完整报告：{duplicate_report_path}"
            )
        assets = args.assets_dir.resolve()
        config = {
            "device": choose_device(args.device), "num_workers": args.num_workers,
            "image_size": args.image_size, "model_size": args.model_size,
            "max_steps": args.max_steps, "lr": args.lr, "weight_decay": args.weight_decay,
            "seed": args.seed, "threshold_quantile": args.threshold_quantile,
            "circle_config": str(args.circle_config.resolve()) if args.circle_config else None,
            "assets_dir": str(assets),
            "imagenette_dir": str(args.imagenette_dir.resolve() if args.imagenette_dir else assets / "imagenette"),
            "teacher_weights": str(args.teacher_weights.resolve()) if args.teacher_weights else None,
        }
        prepare_circle_records(manifest, config)
    seed_everything(config["seed"])
    write_json(output / "manifest.json", manifest)
    write_json(output / "config.json", config)
    print(f"运行目录：{output}\n数据统计：{manifest['summary']}\n设备：{config['device']}")
    print("首次训练会下载官方教师权重和 ImageNette（完整辅助集约 1.5 GB）；已有缓存会复用。")
    model = new_model(config)
    if previous:
        model.model.load_state_dict(previous["model_state"])
    prepare_assets(model, config, load_teacher=previous is None)
    loader = make_loader(manifest["train"], config, shuffle=True)
    if previous is None:
        model.eval()
        statistics = model.teacher_channel_mean_std(loader)
        check_statistics(statistics)
        model.model.mean_std.update(statistics)
    optimizer = torch.optim.Adam(
        list(model.model.student.parameters()) + list(model.model.ae.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, int(0.95 * config["max_steps"])), gamma=0.1,
    )
    step = previous["step"] if previous else 0
    if previous:
        optimizer.load_state_dict(previous["optimizer_state"])
        scheduler.load_state_dict(previous["scheduler_state"])
    model.train()
    iterator = iter(loader)
    log_path = output / "loss.csv"
    with log_path.open("w", encoding="utf-8", newline="") as log:
        log.write("step,loss,loss_st,loss_ae,loss_stae,lr\n")
        progress = tqdm(total=config["max_steps"], initial=step, desc=f"训练 {manifest['category']}")
        try:
            while step < config["max_steps"]:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                try:
                    auxiliary = next(model.imagenet_iterator)[0]
                except StopIteration:
                    model.imagenet_iterator = iter(model.imagenet_loader)
                    auxiliary = next(model.imagenet_iterator)[0]
                optimizer.zero_grad(set_to_none=True)
                losses = model.model(batch=batch.image.to(config["device"]),
                                     batch_imagenet=auxiliary.to(config["device"]))
                loss = sum(losses)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"第 {step + 1} 步损失不是有限数值，训练停止。")
                loss.backward()
                optimizer.step()
                scheduler.step()  # 每步调度，在总步数的 95% 处将学习率降为原来的 0.1。
                step += 1
                numbers = [float(loss.detach())] + [float(x.detach()) for x in losses]
                log.write(f"{step}," + ",".join(str(x) for x in numbers) + f",{scheduler.get_last_lr()[0]}\n")
                progress.update(1)
                if step % 20 == 0 or step == 1:
                    progress.set_postfix(loss=f"{numbers[0]:.5f}")
                    log.flush()
                if step % args.save_every == 0:
                    save_checkpoint(output / "checkpoints" / "last.pt", model, config, manifest, step, optimizer, scheduler)
        except KeyboardInterrupt:
            save_checkpoint(output / "checkpoints" / "last.pt", model, config, manifest, step, optimizer, scheduler)
            print(f"\n已中断并保存续训文件：{output / 'checkpoints' / 'last.pt'}")
            raise
        finally:
            progress.close()
    save_checkpoint(output / "checkpoints" / "last.pt", model, config, manifest, step, optimizer, scheduler)
    calibration = calibrate(model, manifest, config, output)
    save_checkpoint(output / "model.pt", model, config, manifest, step, calibration=calibration)
    evaluate_records(model, manifest["test"], config, calibration, output, args.heatmaps)
    print(f"完成。模型：{output / 'model.pt'}\n报告：{output}")


def restore_for_inference(args):
    saved = read_checkpoint(args.checkpoint)
    if not saved.get("calibration"):
        raise ValueError("此 checkpoint 尚未校准，请使用训练结束生成的 model.pt。")
    config = saved["config"].copy()
    config.update(device=choose_device(args.device), num_workers=args.num_workers)
    model = new_model(config)
    model.model.load_state_dict(saved["model_state"])
    model.eval()
    return model, config, saved


def predict(args) -> None:
    from ccd_report import save_heatmap

    model, config, saved = restore_for_inference(args)
    image_path = args.image.resolve()
    stat = image_path.stat()
    record = {"path": str(image_path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "label": 0}
    circle_path = config.get("circle_config")
    if circle_path:
        params = config.get("circle_params") or category_config(load_configs(Path(circle_path)), saved["manifest"]["category"])
        record["circle"] = default_mask_record(
            read_rgb(image_path), params, base_dir=Path(circle_path).parent
        )
    batch = next(iter(make_loader([record], config)))
    with torch.inference_mode():
        prediction = model.model(batch.image.to(config["device"]))
    score = prediction_score(prediction, batch, config["device"])
    if not math.isfinite(score):
        raise ValueError("预测分数不是有限数值。")
    calibration = saved["calibration"]
    output = new_output(args.output_dir, saved["manifest"]["category"])
    result = {"image": str(image_path), "score": score, "threshold": calibration["threshold"],
              "prediction": "NG" if score > calibration["threshold"] else "OK"}
    write_json(output / "prediction.json", result)
    ignore_mask = getattr(batch, "ignore_mask", None)
    save_heatmap(image_path, prediction.anomaly_map.squeeze().cpu().numpy(), output / "prediction.png",
                 display_max=calibration["display_max"], score=score, threshold=calibration["threshold"],
                 ignore_mask=_mask_numpy(ignore_mask))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"预测结果：{output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "train", "evaluate", "predict"):
        command = sub.add_parser(name, help={"inspect": "检查数据与划分", "train": "训练并评估",
                                            "evaluate": "用已保存的阈值评估", "predict": "预测一张图片"}[name])
        command.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
        if name in ("inspect", "train"):
            command.add_argument("--data-root", type=Path, help="包含 CCD1 等目录的数据根目录")
            command.add_argument("--category", default="CCD1", help="CCD1、CCD2 等；all 表示依次处理全部 CCD")
            command.add_argument("--val-ratio", type=float, default=0.2)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--min-age-seconds", type=float, default=60, help="跳过最近写入的文件")
        if name in ("train",):
            command.add_argument("--circle-config", type=Path, help="按工件类型配置圆检测与 mask 的 JSON 文件")
        if name != "inspect":
            command.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
            command.add_argument("--num-workers", type=int, default=0, help="Windows 首次建议用 0")
        if name in ("train", "evaluate"):
            command.add_argument("--heatmaps", type=int, default=32, help="热图数量；0 不保存，-1 保存全部")
        if name in ("evaluate", "predict"):
            command.add_argument("--checkpoint", type=Path, required=True, help="本脚本生成的 model.pt")
        if name == "train":
            command.add_argument("--max-steps", type=int, default=10000, help="快速试跑可设为 1000，正式对照可设为 70000")
            command.add_argument("--image-size", type=int, choices=[256, 384, 512, 768], default=256)
            command.add_argument("--model-size", choices=["small", "medium"], default="small")
            command.add_argument("--lr", type=float, default=1e-4)
            command.add_argument("--weight-decay", type=float, default=1e-5)
            command.add_argument("--threshold-quantile", type=float, default=0.99)
            command.add_argument("--save-every", type=int, default=1000)
            command.add_argument("--assets-dir", type=Path, default=PROJECT_DIR / "assets")
            command.add_argument("--imagenette-dir", type=Path, help="已有 ImageNette/ImageNet 图片目录，按 ImageFolder 格式")
            command.add_argument("--teacher-weights", type=Path, help="已有 pretrained_teacher_small.pth 或 medium 权重")
            command.add_argument("--resume", type=Path, help="从 checkpoints/last.pt 续训，沿用该次超参数与数据快照")
        elif name == "evaluate":
            command.add_argument("--manifest", type=Path, help="可选：inspect 生成的新快照；不会重新选择阈值")
        elif name == "predict":
            command.add_argument("--image", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "num_workers") and args.num_workers < 0:
        raise ValueError("num-workers 不能为负数。")
    if hasattr(args, "heatmaps") and args.heatmaps < -1:
        raise ValueError("heatmaps 必须是 -1 或非负数。")
    if args.command == "inspect":
        root = resolve_data_root(args.data_root)
        categories = list_categories(root) if args.category.lower() == "all" else [args.category]
        if not categories:
            raise ValueError(f"数据目录下还没有包含 train/test 的 CCD 类别：{root}")
        failed = []
        for category in categories:
            try:
                manifest = snapshot(args, category)
                output = new_output(args.output_dir / "inspection", category)
                write_json(output / "manifest.json", manifest)
                report_duplicate_content(manifest, output)
                print(f"{category}: {manifest['summary']}\n快照：{output / 'manifest.json'}")
            except (ValueError, FileNotFoundError) as error:
                if args.category.lower() != "all":
                    raise
                failed.append(category)
                print(f"{category} 检查失败（可能尚未下载完整）：{error}", file=sys.stderr)
        if failed:
            raise ValueError(f"以下类别未通过检查：{', '.join(failed)}；其他类别的快照已保存。")
        return
    load_runtime()
    if args.command == "train":
        if (args.max_steps < 1 or args.save_every < 1 or not math.isfinite(args.lr) or args.lr <= 0
                or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
            raise ValueError("max-steps、save-every 和 lr 必须为正数，weight-decay 不能为负数。")
        if not 0 < args.threshold_quantile <= 1:
            raise ValueError("threshold-quantile 必须在 (0, 1] 内。")
        if args.resume and args.category.lower() == "all":
            raise ValueError("resume 一次只能恢复一个模型，不能与 category all 合用。")
        categories = list_categories(resolve_data_root(args.data_root)) if args.category.lower() == "all" else [args.category]
        if not categories:
            raise ValueError("数据目录下还没有包含 train/test 的 CCD 类别。")
        for category in categories:
            train_one(args, category)
    elif args.command == "evaluate":
        model, config, saved = restore_for_inference(args)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else saved["manifest"]
        if manifest["category"] != saved["manifest"]["category"]:
            raise ValueError("评估快照的 CCD 类别与 checkpoint 不一致。")
        saved_config = saved.get("config", {})
        if saved_config.get("circle_config") and any("circle" not in record for record in manifest["test"]):
            prepare_circle_records(manifest, saved_config)
        output = new_output(args.output_dir / "evaluation", manifest["category"])
        write_json(output / "manifest.json", manifest)
        evaluate_records(model, manifest["test"], config, saved["calibration"], output, args.heatmaps)
        print(f"评估结果：{output}")
    else:
        predict(args)


if __name__ == "__main__":
    # Windows 的 DataLoader 多进程需要入口保护。
    # 重定向日志时也采用 UTF-8，避免中文路径和提示在 PowerShell 中乱码。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
