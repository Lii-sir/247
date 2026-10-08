"""唯一接触 Ultralytics 的适配层。模型可复用，返回普通 NumPy 数据。"""

from pathlib import Path
from time import perf_counter

from common.image_io import read_image
from .models import Segment, SegmentationResult, SegmentationSettings


class PartSegmenter:
    """顺序调用 predict；不要在多个线程中同时使用同一个实例。"""

    def __init__(self, weights: str | Path):
        self.weights = Path(weights).expanduser().resolve()
        if not self.weights.is_file():
            raise ValueError(f"权重文件不存在：{self.weights}")
        # 延迟导入：渲染、测试和命令行帮助不需要初始化 Torch。
        from ultralytics import YOLO

        self._model = YOLO(str(self.weights))
        if self._model.task != "segment":
            raise ValueError("请选择实例分割权重；当前权重不是 segment 模型")

    @property
    def class_names(self) -> tuple[str, ...]:
        """Expose model labels without leaking the Ultralytics adapter."""
        names = self._model.names
        return tuple(names.values()) if isinstance(names, dict) else tuple(names)

    def predict(self, image_path: str | Path, settings: SegmentationSettings) -> SegmentationResult:
        start = perf_counter()
        image_path = Path(image_path).resolve()
        image = read_image(image_path)
        self._validate_device(settings.device)
        prediction = self._model.predict(
            source=image,
            conf=settings.confidence,
            iou=settings.iou,
            imgsz=settings.image_size,
            device=settings.device,
            retina_masks=True,  # 返回原图尺寸的掩膜，避免缩放/letterbox 坐标偏移。
            verbose=False,
            save=False,
        )[0]
        segments = []
        if prediction.boxes is not None and len(prediction.boxes):
            if prediction.masks is None:
                raise ValueError("模型返回了检测框但没有分割掩膜")
            masks = prediction.masks.data.cpu().numpy() > 0.5
            boxes = prediction.boxes.xyxy.cpu().numpy()
            classes = prediction.boxes.cls.cpu().numpy().astype(int)
            scores = prediction.boxes.conf.cpu().numpy()
            if masks.shape != (len(boxes), *image.shape[:2]):
                raise ValueError("分割掩膜数量或尺寸与原图不一致")
            for class_id, score, box, mask in zip(classes, scores, boxes, masks, strict=True):
                segments.append(Segment(
                    class_id=int(class_id),
                    class_name=prediction.names[int(class_id)],
                    confidence=float(score),
                    box=tuple(float(value) for value in box),
                    mask=mask,
                ))
        return SegmentationResult(image_path, image, tuple(segments), (perf_counter() - start) * 1000)

    @staticmethod
    def _validate_device(device: str) -> None:
        if device.strip().casefold() == "cpu":
            return
        try:
            import torch
            available = torch.cuda.is_available()
            count = torch.cuda.device_count() if available else 0
        except ImportError as exc:
            raise RuntimeError("CUDA 推理需要安装带 CUDA 的 PyTorch") from exc
        if not available:
            raise RuntimeError("当前 PyTorch 不支持 CUDA；请安装 CUDA 版 PyTorch，或显式使用 --device cpu")
        value = device.strip().casefold().replace("cuda:", "")
        if value.isdigit() and int(value) >= count:
            raise RuntimeError(f"CUDA 设备 {device} 不存在，当前只有 {count} 个 CUDA 设备")

