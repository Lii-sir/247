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
    multiscale_topk_scores,
    pooled_topk_score,
    read_rgb,
)
from ccd_localization import (
    build_localization,
    localization_summary,
    normalize_localization_params,
)
from ccd_data import list_categories, prepare_manifest, resolve_data_root, split_threshold_validation

PROJECT_DIR = Path(__file__).resolve().parent
Batch = namedtuple("Batch", ["image", "gt_label", "ignore_mask"])


def write_json(path: Path, data: dict) -> None:
    """使用 UTF-8 保存记录，保留中文目录名。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


LOCALIZATION_ARGUMENT_FIELDS = {
    "box_min_area_ratio": "min_area_ratio",
    "box_morph_kernel": "morph_kernel",
    "box_open_iterations": "open_iterations",
    "box_close_iterations": "close_iterations",
    "box_merge_iou": "merge_iou",
    "box_merge_containment": "merge_containment",
    "box_merge_distance_ratio": "merge_distance_ratio",
    "box_padding_ratio": "padding_ratio",
    "box_fallback_size_ratio": "fallback_size_ratio",
    "box_normalized_display_max": "normalized_display_max",
}


def apply_localization_config(config: dict, args) -> None:
    """将 checkpoint/默认定位参数与本次命令行覆盖合并到统一配置。"""
    params = dict(config.get("localization", {}))
    for argument, field in LOCALIZATION_ARGUMENT_FIELDS.items():
        value = getattr(args, argument, None)
        if value is not None:
            params[field] = value
    config["localization"] = normalize_localization_params(params)


def add_localization_arguments(parser: argparse.ArgumentParser) -> None:
    """为训练、评估和单图预测添加同一组可复用的画框参数。"""
    parser.add_argument("--box-min-area-ratio", type=float, help="最小连通域面积/有效图面积，默认 0.00002")
    parser.add_argument("--box-morph-kernel", type=int, help="开闭运算核，必须为正奇数，默认 3")
    parser.add_argument("--box-open-iterations", type=int, help="形态学开运算次数，默认 1")
    parser.add_argument("--box-close-iterations", type=int, help="形态学闭运算次数，默认 1")
    parser.add_argument("--box-merge-iou", type=float, help="框重叠合并 IoU，默认 0.15")
    parser.add_argument(
        "--box-merge-containment", type=float,
        help="小框被大框覆盖到该比例时合并，默认 0.80",
    )
    parser.add_argument("--box-merge-distance-ratio", type=float, help="临近框合并距离/图像对角线，默认 0.005")
    parser.add_argument("--box-padding-ratio", type=float, help="框外扩像素/最大边长，默认 0.003")
    parser.add_argument("--box-fallback-size-ratio", type=float, help="异常但无连通域时兜底框尺寸比例，默认 0.02")
    parser.add_argument("--box-normalized-display-max", type=float, help="多尺度归一化融合图固定色阶下限，默认 3.0")


def load_runtime() -> None:
    """仅训练与推理时加载深度学习库，使数据检查命令启动更快。"""
    global torch, np, EfficientAd, DataLoader, Image, TF, tqdm
    import numpy as np
    import torch
    from self_efficientad import EfficientAd
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


def make_loader(records: list[dict], config: dict, *, shuffle: bool = False, training: bool = False):
    """Create a loader with explicit training/evaluation batch semantics.

    Training uses the configured full batch and drops the final incomplete batch
    so the normal-image/ImageNette ratio stays fixed. All evaluation paths keep
    one image per batch because their reporting and heatmap code is per-image.
    """
    batch_size = config.get("batch_size", 1) if training else 1
    if training and len(records) < batch_size:
        raise ValueError(
            "训练图片数量必须不少于 batch-size，"
            f"当前为 {len(records)} < {batch_size}；请减小 batch-size 或增加训练图片。"
        )
    return DataLoader(
        SnapshotDataset(records, config["image_size"]),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=training,
        num_workers=config["num_workers"],
        collate_fn=collate_batch,
        pin_memory=config["device"].startswith("cuda"),
        persistent_workers=config["num_workers"] > 0,
    )


def choose_devices(requested: str | list[str]) -> list[str]:
    values = [requested] if isinstance(requested, str) else requested
    if values == ["cpu"] or (values == ["auto"] and not torch.cuda.is_available()):
        return ["cpu"]
    if values in (["auto"], ["cuda"]):
        values = ["0"]
    if not values or any(not value.isdecimal() for value in values):
        raise ValueError("device 应为 auto、cuda、cpu 或 GPU 编号列表，例如 --device 0 1。")
    indices = [int(value) for value in values]
    if len(set(indices)) != len(indices):
        raise ValueError("device 中不能重复指定同一张 GPU。")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 CUDA，请运行 README 中的 GPU 检查命令。")
    count = torch.cuda.device_count()
    if any(index >= count for index in indices):
        raise ValueError(f"GPU 编号越界：当前可见 {count} 张卡，请使用 0 到 {count - 1}。")
    return [f"cuda:{index}" for index in indices]


def choose_device(requested: str | list[str]) -> str:
    devices = choose_devices(requested)
    if len(devices) != 1:
        raise ValueError("评估和预测仅支持单个设备；多卡列表用于 train。")
    return devices[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def new_model(config: dict):
    # 使用项目内副本；训练循环由本脚本控制，学习率按 step 更新。
    backbone = config.get("backbone") or f"pdn_{config['model_size']}"
    return EfficientAd(
        imagenet_dir=config["imagenette_dir"],
        model_size=config["model_size"],
        backbone=backbone,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        batch_size=config.get("batch_size", 1),
        hard_loss_mode=config.get("hard_loss_mode", "global"),
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    ).to(config["device"])


def prepare_assets(model, config: dict, *, load_teacher: bool) -> None:
    """复用官方带 SHA256 校验的下载器，也支持手工提供辅助数据。"""
    from anomalib.data.utils import download_and_extract
    from self_efficientad.lightning_model import WEIGHTS_DOWNLOAD_INFO

    if load_teacher:
        teacher_path = config.get("teacher_weights")
        backbone = config.get("backbone", f"pdn_{config['model_size']}")
        if teacher_path:
            teacher_path = Path(teacher_path)
            if not teacher_path.is_file():
                raise FileNotFoundError(f"找不到教师权重：{teacher_path}")
            model.model.teacher.load_state_dict(
                torch.load(teacher_path, map_location=config["device"], weights_only=True)
            )
        elif backbone.startswith("pdn_"):
            cache = Path(config["assets_dir"]) / "pre_trained"
            teacher_size = "medium" if backbone == "pdn_medium" else "small"
            teacher_path = cache / "efficientad_pretrained_weights" / (
                f"pretrained_teacher_{teacher_size}.pth"
            )
            if not teacher_path.is_file():
                cache.mkdir(parents=True, exist_ok=True)
                download_and_extract(cache, WEIGHTS_DOWNLOAD_INFO)
            model.model.teacher.load_state_dict(
                torch.load(teacher_path, map_location=config["device"], weights_only=True)
            )
        else:
            from self_efficientad.backbones import load_default_teacher_weights

            load_default_teacher_weights(backbone, model.model.teacher)
    imagenette = Path(config["imagenette_dir"])
    # 仅建立父目录，空的叶目录会让官方函数误判为已经下载完成。
    imagenette.parent.mkdir(parents=True, exist_ok=True)
    model.prepare_imagenette_data(
        (config["image_size"], config["image_size"]),
        num_workers=config["num_workers"],
        start_iterator=config.get("world_size", 1) == 1,
    )
    if len(model.imagenet_loader.dataset) == 0:
        raise ValueError(f"ImageNette 辅助数据为空：{imagenette}")


def save_checkpoint(path: Path, model, config: dict, manifest: dict, step: int,
                    optimizer=None, scheduler=None, calibration: dict | None = None) -> None:
    """先写临时文件再替换，减少中断时留下半个 checkpoint 的可能性。"""
    if "torch" not in globals():
        load_runtime()
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
    if "torch" not in globals():
        load_runtime()
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


TOP_SCORE_METHOD = "masked_pixel_max"
SINGLE_SCALE_SCORE_METHOD = "masked_local_average_topk_mean"
MULTISCALE_SCORE_METHOD = "masked_multiscale_normalized_topk_max"
SCORE_MODE_TOP = "top"
SCORE_MODE_POOL_TOPK = "pool_topk"
SCORE_MODE_MULTISCALE = "multiscale_pool"
SCORE_MODE_CHECKPOINT = "checkpoint"


def normalize_score_mode(value: str, *, allow_checkpoint: bool = False) -> str:
    """规范化命令行 score 模式，同时接受用户常用写法。"""
    aliases = {
        "top": SCORE_MODE_TOP,
        "max": SCORE_MODE_TOP,
        "pool+top": SCORE_MODE_POOL_TOPK,
        "pool_topk": SCORE_MODE_POOL_TOPK,
        "pool-topk": SCORE_MODE_POOL_TOPK,
        "multiscale_pool": SCORE_MODE_MULTISCALE,
        "multiscale-pool": SCORE_MODE_MULTISCALE,
        "multi_pool": SCORE_MODE_MULTISCALE,
        "multi-pool": SCORE_MODE_MULTISCALE,
    }
    if allow_checkpoint:
        aliases["checkpoint"] = SCORE_MODE_CHECKPOINT
        aliases["saved"] = SCORE_MODE_CHECKPOINT
    normalized = aliases.get(str(value).strip().lower())
    if normalized is None:
        options = "top、pool+top、multiscale_pool"
        if allow_checkpoint:
            options += "、checkpoint"
        raise ValueError(f"score-mode 必须是 {options} 之一。")
    return normalized


def calibration_score_mode(calibration: dict) -> str:
    """识别 checkpoint 的 score 方式；无 score_method 的旧模型属于 top 版本。"""
    method = (calibration.get("score_method") or {}).get("name")
    if method == MULTISCALE_SCORE_METHOD:
        return SCORE_MODE_MULTISCALE
    if method == SINGLE_SCALE_SCORE_METHOD:
        return SCORE_MODE_POOL_TOPK
    if method in (None, TOP_SCORE_METHOD):
        return SCORE_MODE_TOP
    raise ValueError(f"checkpoint 包含未知的 score_method：{method}")


def prediction_localization(
    prediction,
    batch,
    config: dict,
    calibration: dict,
    score_values: dict,
) -> dict:
    """使用与当前整图 Score 相同的响应定义生成画框数据。"""
    return build_localization(
        prediction.anomaly_map,
        getattr(batch, "ignore_mask", None),
        score_mode=calibration_score_mode(calibration),
        score_method=calibration.get("score_method") or {},
        score_values=score_values,
        threshold=calibration["threshold"],
        raw_display_max=calibration["display_max"],
        params=config.get("localization"),
    )


def score_pool_kernels(config: dict) -> tuple[int, ...]:
    """读取并规范化多尺度池化核，同时兼容旧版单尺度配置。"""
    configured = config.get("score_pool_kernels")
    if configured is None:
        configured = [config.get("score_pool_kernel", 21)]
    kernels = tuple(dict.fromkeys(int(kernel) for kernel in configured))
    if not kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
        raise ValueError("score_pool_kernels 必须包含至少一个正奇数。")
    return kernels


def raw_prediction_scores(
    prediction,
    batch,
    config: dict,
    *,
    pool_kernels: tuple[int, ...] | list[int] | None = None,
    topk_ratio: float | None = None,
) -> dict[str, float]:
    """计算每个池化尺度的原始 Top-K 分数和原始最大值。"""
    anomaly_map = getattr(prediction, "anomaly_map", None)
    ignore_mask = getattr(batch, "ignore_mask", None)
    if anomaly_map is not None and ignore_mask is not None:
        mask = _mask_tensor(ignore_mask, config["device"])
        score_max = float(mask_score(anomaly_map, mask).flatten()[0])
        kernels = score_pool_kernels(config) if pool_kernels is None else tuple(pool_kernels)
        scale_scores = multiscale_topk_scores(
            anomaly_map,
            mask,
            pool_kernels=kernels,
            topk_ratio=(config.get("score_topk_ratio", 0.001) if topk_ratio is None else topk_ratio),
        ) if kernels else {}
        return {
            "score_max": score_max,
            **{
                f"score_kernel_{kernel}": float(value.flatten()[0])
                for kernel, value in scale_scores.items()
            },
        }
    score = float(prediction.pred_score.flatten()[0])
    return {"score_max": score}


def build_score_method(
    normal_rows: list[dict], config: dict, score_mode: str = SCORE_MODE_MULTISCALE
) -> dict:
    """建立所选 score 定义；多尺度方式额外校准每个尺度。"""
    score_mode = normalize_score_mode(score_mode)
    if score_mode == SCORE_MODE_TOP:
        return {"name": TOP_SCORE_METHOD, "mode": SCORE_MODE_TOP}
    if score_mode == SCORE_MODE_POOL_TOPK:
        return {
            "name": SINGLE_SCALE_SCORE_METHOD,
            "mode": SCORE_MODE_POOL_TOPK,
            "pool_kernel": int(config.get("score_pool_kernel", 21)),
            "topk_ratio": float(config.get("score_topk_ratio", 0.001)),
        }

    normalization = {}
    for kernel in score_pool_kernels(config):
        field = f"score_kernel_{kernel}"
        values = np.asarray([row[field] for row in normal_rows], dtype=np.float64)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"尺度 {kernel} 的正常验证分数为空或包含 NaN/Inf。")
        median = float(np.quantile(values, 0.5))
        high = float(np.quantile(values, 0.99))
        denominator = high - median
        epsilon = max(1.0, abs(median), abs(high)) * 1e-12
        if denominator <= epsilon:
            raise ValueError(
                f"尺度 {kernel} 的正常验证分数退化（median={median}, q99={high}）；"
                "请增加有代表性的正常验证图片。"
            )
        normalization[str(kernel)] = {
            "median": median,
            "q99": high,
            "denominator": denominator,
        }
    return {
        "name": MULTISCALE_SCORE_METHOD,
        "mode": SCORE_MODE_MULTISCALE,
        "pool_kernels": list(score_pool_kernels(config)),
        "topk_ratio": config.get("score_topk_ratio", 0.001),
        "normalization": normalization,
        "normalization_formula": "max(0, (raw_score - median) / (q99 - median))",
        "fusion": "maximum_normalized_scale_score",
    }


def fuse_prediction_scores(raw_scores: dict[str, float], score_method: dict) -> dict[str, float]:
    """按正常集尺度基准归一化并取最大值作为最终整图 score。"""
    normalized = {}
    for kernel in score_method["pool_kernels"]:
        key = str(kernel)
        reference = score_method["normalization"][key]
        raw = raw_scores[f"score_kernel_{kernel}"]
        value = max(0.0, (raw - reference["median"]) / reference["denominator"])
        normalized[f"score_normalized_kernel_{kernel}"] = float(value)
    final_score = max(normalized.values())
    return {
        "score": final_score,
        "score_topk": final_score,
        **raw_scores,
        **normalized,
    }


def prediction_scores(prediction, batch, config: dict, calibration: dict | None = None) -> dict[str, float]:
    """按照 calibration 保存的定义计算整图分数。"""
    score_method = (calibration or {}).get("score_method", {})
    method_name = score_method.get("name")
    if method_name == MULTISCALE_SCORE_METHOD:
        return fuse_prediction_scores(
            raw_prediction_scores(
                prediction,
                batch,
                config,
                pool_kernels=score_method["pool_kernels"],
                topk_ratio=score_method["topk_ratio"],
            ),
            score_method,
        )

    anomaly_map = getattr(prediction, "anomaly_map", None)
    ignore_mask = getattr(batch, "ignore_mask", None)
    if anomaly_map is not None and ignore_mask is not None:
        mask = _mask_tensor(ignore_mask, config["device"])
        score_max = float(mask_score(anomaly_map, mask).flatten()[0])
        # f647384 及更早 checkpoint 没有 score_method，其 threshold 对应单像素最大值。
        if method_name in (None, TOP_SCORE_METHOD):
            return {"score": score_max, "score_topk": score_max, "score_max": score_max}
        if method_name != SINGLE_SCALE_SCORE_METHOD:
            raise ValueError(f"calibration 包含未知的 score_method：{method_name}")
        kernel = int(score_method.get("pool_kernel", config.get("score_pool_kernel", 21)))
        score = float(pooled_topk_score(
            anomaly_map, mask, pool_kernel=kernel,
            topk_ratio=score_method.get("topk_ratio", config.get("score_topk_ratio", 0.001)),
        ).flatten()[0])
        return {"score": score, "score_topk": score, "score_max": score_max}
    score = float(prediction.pred_score.flatten()[0])
    return {"score": score, "score_topk": score, "score_max": score}


def prediction_score(prediction, batch, device: str) -> float:
    """兼容旧调用入口；正式流程使用 prediction_scores。"""
    return prediction_scores(prediction, batch, {"device": device})["score"]


def threshold_for_target_recall(rows: list[dict], target_recall: float) -> tuple[float, dict]:
    """选择满足每种异常类型目标召回率的最高阈值，等于阈值仍判为正常。"""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall 必须位于 (0, 1]。")
    normal_scores = np.asarray([row["score"] for row in rows if row["label"] == 0], dtype=np.float64)
    anomaly_scores = np.asarray([row["score"] for row in rows if row["label"] == 1], dtype=np.float64)
    if not len(normal_scores) or not len(anomaly_scores):
        raise ValueError("阈值验证集必须同时包含正常图片和异常图片。")
    anomaly_groups: dict[str, list[float]] = {}
    for row in rows:
        if row["label"] == 1:
            defect_type = row.get("defect_type", "anomaly") or "anomaly"
            anomaly_groups.setdefault(defect_type, []).append(row["score"])
    per_defect = {}
    boundaries = []
    for defect_type, values in sorted(anomaly_groups.items()):
        scores = np.asarray(values, dtype=np.float64)
        required_tp = int(math.ceil(target_recall * len(scores)))
        boundary = float(np.sort(scores)[::-1][required_tp - 1])
        boundaries.append(boundary)
        per_defect[defect_type] = {
            "count": int(len(scores)),
            "required_true_positive": required_tp,
            "boundary": boundary,
        }
    limiting_boundary = min(boundaries)
    threshold = float(np.nextafter(limiting_boundary, -np.inf))
    predictions_normal = normal_scores > threshold
    predictions_anomaly = anomaly_scores > threshold
    for defect_type, detail in per_defect.items():
        scores = np.asarray(anomaly_groups[defect_type], dtype=np.float64)
        detail["true_positive"] = int((scores > threshold).sum())
        detail["false_negative"] = int((scores <= threshold).sum())
        detail["achieved_recall"] = float((scores > threshold).mean())
    stats = {
        "target_recall": target_recall,
        "normal_count": int(len(normal_scores)),
        "anomaly_count": int(len(anomaly_scores)),
        "true_positive": int(predictions_anomaly.sum()),
        "false_negative": int((~predictions_anomaly).sum()),
        "false_positive": int(predictions_normal.sum()),
        "true_negative": int((~predictions_normal).sum()),
        "achieved_recall": float(predictions_anomaly.mean()),
        "false_positive_rate": float(predictions_normal.mean()),
        "limiting_boundary": limiting_boundary,
        "degenerate_zero_boundary": limiting_boundary <= 0,
        "per_defect": per_defect,
    }
    return threshold, stats


def calibrate(
    model,
    manifest: dict,
    config: dict,
    output_dir: Path,
    score_mode: str | None = None,
) -> dict:
    """校准异常图，并为指定 score 方式用独立带标签验证集选择阈值。"""
    score_mode = normalize_score_mode(score_mode or config.get("score_mode", SCORE_MODE_MULTISCALE))
    model.eval()  # 关闭自编码器 dropout 后再估计分位数。
    loader = make_loader(manifest["val"], config)
    with torch.inference_mode():
        quantiles = masked_map_quantiles(model, loader, config["device"])
        check_statistics(quantiles, quantiles=True)
        model.model.quantiles.update(quantiles)
        if score_mode == SCORE_MODE_TOP:
            calibration_kernels: list[int] = []
        elif score_mode == SCORE_MODE_POOL_TOPK:
            calibration_kernels = [int(config.get("score_pool_kernel", 21))]
        else:
            calibration_kernels = list(score_pool_kernels(config))
        normal_score_rows = []
        normal_loader = make_loader(manifest["val"], config)
        for record, batch in zip(manifest["val"], normal_loader, strict=True):
            prediction = model.model(batch.image.to(config["device"]))
            raw_scores = raw_prediction_scores(
                prediction,
                batch,
                config,
                pool_kernels=calibration_kernels,
                topk_ratio=config.get("score_topk_ratio", 0.001),
            )
            normal_score_rows.append({"path": record["path"], "label": 0, **raw_scores})
        score_method = build_score_method(normal_score_rows, config, score_mode)
        threshold_records = manifest.get("threshold_val", [])
        if not threshold_records:
            raise ValueError("没有阈值验证集；请从 test 分层划出 threshold_val 后再校准。")
        threshold_loader = make_loader(threshold_records, config)
        rows = []
        for record, batch in zip(threshold_records, threshold_loader, strict=True):
            prediction = model.model(batch.image.to(config["device"]))
            score_values = prediction_scores(
                prediction, batch, config, {"score_method": score_method}
            )
            if not all(math.isfinite(value) for value in score_values.values()):
                raise ValueError(f"校准分数不是有限数值：{record['path']}")
            rows.append({
                "path": record["path"],
                "label": record["label"],
                "defect_type": record.get("defect_type", "good" if record.get("label", 0) == 0 else "anomaly"),
                **score_values,
            })
    threshold, selection = threshold_for_target_recall(rows, config.get("target_recall", 0.99))
    if selection["degenerate_zero_boundary"]:
        print(
            "警告：至少一种异常类型的目标召回边界为 0；当前 score 无法有效区分该异常，"
            "为满足召回约束，阈值会低于 0 并可能导致极高误报率。"
        )
    normal_map_maxima = [row["score_max"] for row in normal_score_rows]
    calibration = {
        "method": "labeled_validation_highest_threshold_for_target_recall",
        "score_mode": score_mode,
        "threshold": threshold,
        "decision_rule": "score > threshold",
        "score_method": score_method,
        "threshold_selection": selection,
        "normal_map_validation_count": len(manifest["val"]),
        "threshold_validation_count": len(rows),
        "display_max": max(float(np.quantile(normal_map_maxima, 0.99)), 0.1),
        "map_quantiles": {key: float(value) for key, value in quantiles.items()},
        "notes": (
            f"本次使用 {score_mode} 整图分数，分数不是概率；"
            "threshold_val 从原 test 分层划出，最终 test 未参与尺度校准或阈值选择。"
        ),
    }
    write_json(output_dir / "calibration.json", {
        **calibration,
        "normal_score_validation": normal_score_rows,
        "threshold_validation_scores": rows,
    })
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
            score_values = prediction_scores(prediction, batch, config, calibration)
            if config["device"].startswith("cuda"):
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - inference_started
            if not all(math.isfinite(value) for value in score_values.values()):
                raise ValueError(f"测试分数不是有限数值：{record['path']}")
            rows.append({
                "path": record["path"], "label": record["label"], **score_values,
                "pred_label": int(score_values["score"] > calibration["threshold"]),
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
            prediction = model.model(batch.image.to(config["device"]))
            anomaly_map = prediction.anomaly_map.squeeze().cpu().numpy()
            localization = prediction_localization(
                prediction, batch, config, calibration, row
            )
            # 第一级保持 test 下的真实子目录（good、defect1、defect2 ...），
            # 第二级按整图预测结果区分 normal / anomaly。
            source_directory = _safe_output_component(record.get("defect_type", "unspecified"))
            prediction_directory = "anomaly" if row["pred_label"] else "normal"
            filename = f"{rank:04d}_{Path(record['path']).stem}.png"
            save_heatmap(
                Path(record["path"]), anomaly_map,
                output_dir / "heatmaps" / source_directory / prediction_directory / filename,
                display_max=calibration["display_max"], score=row["score"], threshold=calibration["threshold"],
                ignore_mask=_mask_numpy(getattr(batch, "ignore_mask", None)),
                localization=localization,
                multiscale_output_path=(
                    output_dir / "heatmap_scales" / source_directory
                    / prediction_directory / filename
                ),
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
    value = "".join("_" if char in invalid or ord(char) < 32 else char for char in value)
    return value.rstrip(" .") or "unspecified"


def snapshot(args, category: str) -> dict:
    return prepare_manifest(
        resolve_data_root(args.data_root), category=category,
        val_ratio=args.val_ratio, seed=args.seed,
        min_age_seconds=args.min_age_seconds, verify_images=True,
        threshold_val_ratio=args.threshold_val_ratio,
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
    for split in ("train", "val", "threshold_val", "test"):
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
    from ccd_distributed import configure_training_world, optimize, train_distributed

    devices = choose_devices(args.device)
    if args.resume:
        previous = read_checkpoint(args.resume)
        if "optimizer_state" not in previous:
            raise ValueError("续训请使用 checkpoints/last.pt；model.pt 是推理用文件。")
        config, manifest = previous["config"].copy(), previous["manifest"]
        saved_backbone = config.get("backbone") or f"pdn_{config.get('model_size', 'small')}"
        requested_backbone = args.backbone
        if args.model_size is not None:
            requested_backbone = f"pdn_{args.model_size}"
        if requested_backbone is not None and requested_backbone != saved_backbone:
            raise ValueError(
                f"续训不能更改 backbone：checkpoint={saved_backbone}，请求={requested_backbone}；"
                "请移除 --resume 启动新训练。"
            )
        config.update(device=devices[0], num_workers=args.num_workers)
        # Old checkpoints predate batched training and must retain their exact
        # original loss semantics when resumed.
        config.setdefault("batch_size", 1)
        config.setdefault("hard_loss_mode", "global")
        config.setdefault("model_size", "small")
        config.setdefault("backbone", f"pdn_{config['model_size']}")
        config.setdefault("batch_training_version", 0)
        config.setdefault("training_budget_mode", "optimizer_steps")
        config.setdefault("max_images_requested", None)
        # Keep the derived budget consistent with the actual resume target,
        # including checkpoints whose metadata was edited to extend training.
        configure_training_world(config, devices, resume=True)
        config.setdefault("score_mode", SCORE_MODE_MULTISCALE)
        config.setdefault("score_pool_kernels", [1, 7, 21])
        config.setdefault("score_topk_ratio", 0.001)
        config.setdefault("target_recall", 0.99)
        config.setdefault("threshold_val_ratio", 0.2)
        apply_localization_config(config, args)
        if not manifest.get("threshold_val"):
            manifest["threshold_val"], manifest["test"] = split_threshold_validation(
                manifest.get("test", []), config["threshold_val_ratio"], config["seed"]
            )
            print(
                "旧 checkpoint 未包含 threshold_val：已从原 test 分层划出 "
                f"{len(manifest['threshold_val'])} 张，剩余 test {len(manifest['test'])} 张。"
            )
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
        batch_size = args.batch_size
        hard_loss_mode = "per_image" if batch_size > 1 else "global"
        if args.max_images is not None:
            max_steps = math.ceil(args.max_images / batch_size)
            max_images_requested = args.max_images
            budget_mode = "images"
        else:
            max_steps = args.max_steps
            max_images_requested = None
            budget_mode = "optimizer_steps"
        max_images = max_steps * batch_size
        config = {
            "device": devices[0], "num_workers": args.num_workers,
            "image_size": args.image_size,
            "model_size": (args.backbone.removeprefix("pdn_")
                           if args.backbone and args.backbone.startswith("pdn_")
                           else args.model_size or "small"),
            "backbone": args.backbone or f"pdn_{args.model_size or 'small'}",
            "batch_size": batch_size, "hard_loss_mode": hard_loss_mode,
            "batch_training_version": 1 if batch_size > 1 else 0,
            "max_steps": max_steps, "max_images": max_images,
            "max_images_requested": max_images_requested,
            "training_budget_mode": budget_mode,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "seed": args.seed, "threshold_quantile": args.threshold_quantile,
            "threshold_val_ratio": args.threshold_val_ratio,
            "target_recall": args.target_recall,
            "score_mode": SCORE_MODE_MULTISCALE,
            "score_pool_kernels": (
                [args.score_pool_kernel]
                if args.score_pool_kernel is not None
                else list(args.score_pool_kernels or (1, 7, 21))
            ),
            "score_topk_ratio": args.score_topk_ratio,
            "circle_config": str(args.circle_config.resolve()) if args.circle_config else None,
            "assets_dir": str(assets),
            "imagenette_dir": str(args.imagenette_dir.resolve() if args.imagenette_dir else assets / "imagenette"),
            "teacher_weights": str(args.teacher_weights.resolve()) if args.teacher_weights else None,
        }
        configure_training_world(config, devices, resume=False)
        apply_localization_config(config, args)
        prepare_circle_records(manifest, config)
    seed_everything(config["seed"])
    write_json(output / "manifest.json", manifest)
    write_json(output / "config.json", config)
    print(f"运行目录：{output}\n数据统计：{manifest['summary']}\n设备：{config['device']}")
    print(
        f"每卡 batch-size={config['batch_size']}，全局 batch={config['global_batch_size']}，"
        f"hard-loss={config['hard_loss_mode']}，"
        f"optimizer steps={config['max_steps']}，有效图片预算={config['max_images']}"
    )
    print(f"backbone={config['backbone']}；辅助数据：{config['imagenette_dir']}；缺失的默认资源将自动下载。")
    model = new_model(config)
    if previous:
        model.model.load_state_dict(previous["model_state"])
    prepare_assets(model, config, load_teacher=previous is None)
    if previous is None:
        model.eval()
        # Statistics must include every training image, including a final
        # incomplete batch that is intentionally omitted during optimization.
        statistics_loader = make_loader(manifest["train"], config, shuffle=False, training=False)
        statistics = model.teacher_channel_mean_std(statistics_loader)
        check_statistics(statistics)
        model.model.mean_std.update(statistics)
    train = train_distributed if len(devices) > 1 else optimize
    step = train(model, config, manifest, output, previous, args.save_every)
    calibration = calibrate(model, manifest, config, output)
    save_checkpoint(output / "model.pt", model, config, manifest, step, calibration=calibration)
    evaluate_records(model, manifest["test"], config, calibration, output, args.heatmaps)
    print(f"完成。模型：{output / 'model.pt'}\n报告：{output}")


def restore_for_inference(args):
    saved = read_checkpoint(args.checkpoint)
    if not saved.get("calibration"):
        raise ValueError("此 checkpoint 尚未校准，请使用训练结束生成的 model.pt。")
    config = saved["config"].copy()
    config.setdefault("model_size", "small")
    config.setdefault("backbone", f"pdn_{config['model_size']}")
    config.update(device=choose_device(args.device), num_workers=args.num_workers)
    config.setdefault("score_mode", calibration_score_mode(saved["calibration"]))
    config.setdefault("score_pool_kernels", [1, 7, 21])
    config.setdefault("score_topk_ratio", 0.001)
    config.setdefault("target_recall", 0.99)
    config.setdefault("threshold_val_ratio", 0.2)
    apply_localization_config(config, args)
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
    score_values = prediction_scores(prediction, batch, config, saved["calibration"])
    if not all(math.isfinite(value) for value in score_values.values()):
        raise ValueError("预测分数不是有限数值。")
    calibration = saved["calibration"]
    output = new_output(args.output_dir, saved["manifest"]["category"])
    localization = prediction_localization(
        prediction, batch, config, calibration, score_values
    )
    result = {
        "image": str(image_path), **score_values, "threshold": calibration["threshold"],
        "prediction": "NG" if score_values["score"] > calibration["threshold"] else "OK",
        "localization": localization_summary(localization),
    }
    write_json(output / "prediction.json", result)
    ignore_mask = getattr(batch, "ignore_mask", None)
    save_heatmap(image_path, prediction.anomaly_map.squeeze().cpu().numpy(), output / "prediction.png",
                 display_max=calibration["display_max"], score=score_values["score"], threshold=calibration["threshold"],
                 ignore_mask=_mask_numpy(ignore_mask), localization=localization,
                 multiscale_output_path=output / "prediction_scales.png")
    print(json.dumps({
        **{key: value for key, value in result.items() if key != "localization"},
        "box_count": len(localization["boxes"]),
        "active_scales": [
            scale["kernel"] for scale in localization.get("scales", []) if scale["active"]
        ],
    }, ensure_ascii=False, indent=2))
    print(f"预测结果：{output}")


def parse_pool_kernels(value: str) -> tuple[int, ...]:
    """解析 ``1,7,21`` 形式的多尺度池化核。"""
    try:
        kernels = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as error:
        raise argparse.ArgumentTypeError("池化核必须是逗号分隔的整数，例如 1,7,21。") from error
    if not kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
        raise argparse.ArgumentTypeError("池化核必须全部是正奇数，例如 1,7,21。")
    return kernels


def parse_evaluation_score_mode(value: str) -> str:
    """解析评估 score 模式并将别名转换为稳定的内部名称。"""
    try:
        return normalize_score_mode(value, allow_checkpoint=True)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


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
            command.add_argument(
                "--threshold-val-ratio", type=float, default=0.2,
                help="从 test 各子目录分层划出阈值验证集的比例",
            )
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--min-age-seconds", type=float, default=60, help="跳过最近写入的文件")
        if name in ("train",):
            command.add_argument("--circle-config", type=Path, help="按工件类型配置圆检测与 mask 的 JSON 文件")
        if name != "inspect":
            command.add_argument("--device", nargs="+", default="auto",
                                 help="auto/cuda/cpu 或 GPU 编号；train 支持 --device 0 1，评估/预测只接受一张卡")
            command.add_argument("--num-workers", type=int, default=0, help="Windows 首次建议用 0")
            add_localization_arguments(command)
        if name in ("train", "evaluate"):
            command.add_argument("--heatmaps", type=int, default=32, help="热图数量；0 不保存，-1 保存全部")
        if name in ("evaluate", "predict"):
            command.add_argument("--checkpoint", type=Path, required=True, help="本脚本生成的 model.pt")
        if name == "train":
            command.add_argument("--max-steps", type=int, default=10000, help="快速试跑可设为 1000，正式对照可设为 70000")
            command.add_argument(
                "--max-images", type=int,
                help="全局训练图片预算；按 batch-size × 卡数向上换算 optimizer steps，优先于 --max-steps",
            )
            command.add_argument(
                "--batch-size", type=int, default=1,
                help="每张卡的训练 batch，默认 1；多卡或大于 1 时启用逐图片 hard loss。评估始终使用 1",
            )
            command.add_argument("--image-size", type=int, choices=[256, 384, 512, 768], default=256)
            backbone_group = command.add_mutually_exclusive_group()
            backbone_group.add_argument("--model-size", choices=["small", "medium"], default=None,
                                        help="旧版 PDN 大小选项；新训练未指定架构时默认 small")
            backbone_group.add_argument(
                "--backbone",
                choices=["pdn_small", "pdn_medium", "resnet18_layer2"],
                help="特征提取器；默认随 --model-size 选择 pdn_small 或 pdn_medium",
            )
            command.add_argument("--lr", type=float, default=1e-4)
            command.add_argument("--weight-decay", type=float, default=1e-5)
            command.add_argument("--threshold-quantile", type=float, default=0.99)
            command.add_argument(
                "--target-recall", type=float, default=0.99,
                help="带标签阈值验证集要求达到的异常召回率",
            )
            command.add_argument(
                "--score-pool-kernels", type=parse_pool_kernels,
                help="多尺度局部平均池化窗口，逗号分隔，默认 1,7,21",
            )
            command.add_argument(
                "--score-pool-kernel", type=int,
                help="兼容旧命令：仅使用一个池化尺度；新训练建议使用 --score-pool-kernels",
            )
            command.add_argument(
                "--score-topk-ratio", type=float, default=0.001,
                help="池化后最高位置的比例，默认 0.1%%",
            )
            command.add_argument("--save-every", type=int, default=1000)
            command.add_argument("--assets-dir", type=Path, default=PROJECT_DIR / "assets")
            command.add_argument(
                "--imagenette-dir",
                type=Path,
                help="辅助图片目录（兼容旧参数名），按 ImageFolder 格式；可使用 ImageNette 或 VisA",
            )
            command.add_argument(
                "--teacher-weights", type=Path,
                help="与所选 backbone 匹配的教师权重；ResNet 可省略并自动使用 torchvision ImageNet 权重",
            )
            command.add_argument("--resume", type=Path, help="从 checkpoints/last.pt 续训，沿用该次超参数与数据快照")
        elif name == "evaluate":
            command.add_argument(
                "--manifest", type=Path,
                help=(
                    "可选评估快照；checkpoint 模式沿用原阈值，"
                    "显式 score 模式用快照中的 threshold_val 重新校准"
                ),
            )
            command.add_argument(
                "--score-mode", type=parse_evaluation_score_mode, default=SCORE_MODE_CHECKPOINT,
                help=(
                    "整图分数：top=mask 后最大单像素；pool+top=单尺度池化后 Top-K；"
                    "multiscale_pool=多尺度归一化融合；checkpoint=沿用模型（默认）"
                ),
            )
            command.add_argument(
                "--score-pool-kernel", type=int,
                help="pool+top 的池化核，正奇数，默认沿用 checkpoint 或 21",
            )
            command.add_argument(
                "--score-pool-kernels", type=parse_pool_kernels,
                help="multiscale_pool 的池化核，逗号分隔，默认沿用 checkpoint 或 1,7,21",
            )
            command.add_argument(
                "--score-topk-ratio", type=float,
                help="pool+top 和 multiscale_pool 的 Top-K 比例，默认沿用 checkpoint 或 0.001",
            )
        elif name == "predict":
            command.add_argument("--image", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "num_workers") and args.num_workers < 0:
        raise ValueError("num-workers 不能为负数。")
    if hasattr(args, "heatmaps") and args.heatmaps < -1:
        raise ValueError("heatmaps 必须是 -1 或非负数。")
    if hasattr(args, "threshold_val_ratio") and not 0 < args.threshold_val_ratio < 1:
        raise ValueError("threshold-val-ratio 必须位于 (0, 1) 内。")
    if hasattr(args, "target_recall") and not 0 < args.target_recall <= 1:
        raise ValueError("target-recall 必须位于 (0, 1] 内。")
    if hasattr(args, "batch_size") and args.batch_size < 1:
        raise ValueError("batch-size 必须为正整数。")
    if hasattr(args, "max_images") and args.max_images is not None and args.max_images < 1:
        raise ValueError("max-images 必须为正整数。")
    if (hasattr(args, "score_pool_kernel") and args.score_pool_kernel is not None
            and (args.score_pool_kernel < 1 or args.score_pool_kernel % 2 == 0)):
        raise ValueError("score-pool-kernel 必须是正奇数。")
    if (hasattr(args, "score_topk_ratio") and args.score_topk_ratio is not None
            and not 0 < args.score_topk_ratio <= 1):
        raise ValueError("score-topk-ratio 必须位于 (0, 1] 内。")
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
        if ((args.max_steps < 1 and args.max_images is None) or args.save_every < 1
                or not math.isfinite(args.lr) or args.lr <= 0
                or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
            raise ValueError("未指定 max-images 时 max-steps 必须为正数；save-every 和 lr 必须为正数，weight-decay 不能为负数。")
        if not 0 < args.threshold_quantile <= 1:
            raise ValueError("threshold-quantile 必须在 (0, 1] 内。")
        if args.score_pool_kernel is not None and args.score_pool_kernels is not None:
            raise ValueError(
                "训练时 --score-pool-kernel 与 --score-pool-kernels 不能同时使用；"
                "前者是单尺度，后者是多尺度。"
            )
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
        requested_score_mode = args.score_mode
        explicit_score_mode = requested_score_mode != SCORE_MODE_CHECKPOINT
        score_overrides = (
            args.score_pool_kernel, args.score_pool_kernels, args.score_topk_ratio
        )
        if not explicit_score_mode and any(value is not None for value in score_overrides):
            raise ValueError(
                "score 池化参数不能与 --score-mode checkpoint 合用；"
                "请显式指定 top、pool+top 或 multiscale_pool。"
            )
        if requested_score_mode == SCORE_MODE_TOP and any(
            value is not None for value in score_overrides
        ):
            raise ValueError("top 模式不使用池化核或 Top-K 比例，请移除 score 池化参数。")
        if (requested_score_mode == SCORE_MODE_POOL_TOPK
                and args.score_pool_kernels is not None):
            raise ValueError(
                "pool+top 是单尺度模式，请使用 --score-pool-kernel，"
                "不要使用 --score-pool-kernels。"
            )
        if (requested_score_mode == SCORE_MODE_MULTISCALE
                and args.score_pool_kernel is not None):
            raise ValueError(
                "multiscale_pool 是多尺度模式，请使用 --score-pool-kernels，"
                "不要使用 --score-pool-kernel。"
            )
        config.setdefault("score_pool_kernel", 21)
        config.setdefault("score_pool_kernels", [1, 7, 21])
        config.setdefault("score_topk_ratio", 0.001)
        if explicit_score_mode:
            if args.score_pool_kernel is not None:
                config["score_pool_kernel"] = args.score_pool_kernel
            if args.score_pool_kernels is not None:
                config["score_pool_kernels"] = list(args.score_pool_kernels)
            if args.score_topk_ratio is not None:
                config["score_topk_ratio"] = args.score_topk_ratio

        # checkpoint 模式必须完整复现保存时的 score 和 threshold。只有显式选择
        # score 模式时，才允许从 test 划分 threshold_val 并重新选择对应阈值。
        if explicit_score_mode and not manifest.get("threshold_val"):
            ratio = float(saved_config.get("threshold_val_ratio", 0.2))
            threshold_val, test = split_threshold_validation(
                manifest.get("test", []), ratio, int(saved_config.get("seed", 42))
            )
            manifest["threshold_val"], manifest["test"] = threshold_val, test
            print(
                "评估快照未包含 threshold_val：已从 test 分层划出 "
                f"{len(threshold_val)} 张；当前评估仅使用剩余 {len(test)} 张最终测试图。"
            )
        scored_records = [
            record
            for split in ("val", "threshold_val", "test")
            for record in manifest.get(split, [])
        ]
        if saved_config.get("circle_config") and any("circle" not in record for record in scored_records):
            prepare_circle_records(manifest, saved_config)
        if explicit_score_mode:
            threshold_labels = {int(record.get("label", 0)) for record in manifest.get("threshold_val", [])}
            if threshold_labels != {0, 1}:
                raise ValueError(
                    "显式选择 score-mode 时，threshold_val 必须同时包含 good(label=0) "
                    "和异常(label=1) 图片；请增大 threshold-val-ratio 或提供完整 manifest。"
                )
        output = new_output(args.output_dir / "evaluation", manifest["category"])
        write_json(output / "manifest.json", manifest)
        if explicit_score_mode:
            config["score_mode"] = requested_score_mode
            print(f"正在按 {requested_score_mode} 重新计算 score 并选择匹配的 threshold。")
            calibration = calibrate(model, manifest, config, output, requested_score_mode)
            save_checkpoint(
                output / "model.pt", model, config, manifest,
                int(saved.get("step", 0)), calibration=calibration,
            )
            print(f"已保存采用 {requested_score_mode} score 的模型：{output / 'model.pt'}")
        else:
            calibration = saved["calibration"]
            config["score_mode"] = calibration_score_mode(calibration)
            print(f"沿用 checkpoint 中的 {config['score_mode']} score 和 threshold。")
        write_json(output / "config.json", config)
        if not explicit_score_mode:
            write_json(output / "calibration.json", calibration)
        evaluate_records(model, manifest["test"], config, calibration, output, args.heatmaps)
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
