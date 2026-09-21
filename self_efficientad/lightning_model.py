# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientAd: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.

This module implements the EfficientAd model for fast and accurate anomaly
detection. EfficientAd uses a student-teacher architecture with a pre-trained
PDN backbone (with an experimental ResNet alternative) for
millisecond-level inference times.

The model consists of:
    - A frozen PDN or ResNet teacher network
    - A lightweight student network
    - Knowledge distillation training
    - Anomaly detection via feature comparison

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import EfficientAd
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = EfficientAd()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

Paper:
    "EfficientAd: Accurate Visual Anomaly Detection at
    Millisecond-Level Latencies"
    https://arxiv.org/pdf/2303.14535.pdf

See Also:
    :class:`anomalib.models.image.efficient_ad.torch_model.EfficientAdModel`:
        PyTorch implementation of the EfficientAd model architecture.
"""

import logging
from pathlib import Path
from typing import Any

import torch
import tqdm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, RandomGrayscale, Resize, ToTensor

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.data.transforms.utils import extract_transforms_by_type
from anomalib.data.utils import DownloadInfo, download_and_extract
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import EfficientAdModel, EfficientAdModelSize, reduce_tensor_elems

logger = logging.getLogger(__name__)

IMAGENETTE_DOWNLOAD_INFO = DownloadInfo(
    name="imagenette2.tgz",
    url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    hashsum="6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd",
)

WEIGHTS_DOWNLOAD_INFO = DownloadInfo(
    name="efficientad_pretrained_weights.zip",
    url="https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip",
    hashsum="c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a",
)


class EfficientAd(AnomalibModule):
    """PL Lightning Module for the EfficientAd algorithm.

    The EfficientAd model uses a student-teacher architecture with a pretrained
    PDN or experimental ResNet backbone for anomaly detection.

    Args:
        imagenet_dir (Path | str): Directory path for the Imagenet dataset.
            Defaults to ``"./datasets/imagenette"``.
        teacher_out_channels (int | None): Number of convolution output channels.
            ``None`` selects 384 for PDN or 128 for ResNet.
            Defaults to ``None``.
        model_size (EfficientAdModelSize | str): Size of student and teacher model.
            Defaults to ``EfficientAdModelSize.S``.
        lr (float): Learning rate.
            Defaults to ``0.0001``.
        weight_decay (float): Optimizer weight decay.
            Defaults to ``0.00001``.
        padding (bool): Use padding in convolutional layers.
            Defaults to ``False``.
        pad_maps (bool): Relevant if ``padding=False``. If ``True``, pads the output
            anomaly maps to match size of ``padding=True`` case.
            Defaults to ``True``.
        pre_processor (PreProcessor | bool, optional): Pre-processor used to transform
            input data before passing to model.
            Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor used to process
            model predictions.
            Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator used to compute metrics.
            Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer used to create
            visualizations.
            Defaults to ``True``.

    Example:
        >>> from anomalib.models import EfficientAd
        >>> model = EfficientAd(
        ...     imagenet_dir="./datasets/imagenette",
        ...     model_size="s",
        ...     lr=1e-4
        ... )

    """

    def __init__(
        self,
        imagenet_dir: Path | str = "./datasets/imagenette",
        teacher_out_channels: int | None = None,
        model_size: EfficientAdModelSize | str = EfficientAdModelSize.S,
        lr: float = 0.0001,
        weight_decay: float = 0.00001,
        padding: bool = False,
        pad_maps: bool = True,
        batch_size: int = 1,
        hard_loss_mode: str = "global",
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
        *,
        backbone: str | None = None,
        teacher_pretrained: bool = False,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.imagenet_dir = Path(imagenet_dir)
        if not isinstance(model_size, EfficientAdModelSize):
            model_size = EfficientAdModelSize(model_size)
        self.model_size: EfficientAdModelSize = model_size
        self.backbone = backbone or ("pdn_medium" if model_size == EfficientAdModelSize.M else "pdn_small")
        if self.backbone.startswith("pdn_"):
            self.model_size = EfficientAdModelSize(self.backbone.removeprefix("pdn_"))
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size > 1 and hard_loss_mode != "per_image":
            raise ValueError(
                "batch_size > 1 requires hard_loss_mode='per_image'; "
                "use batch_size=1 for the original global hard loss."
            )
        self.model: EfficientAdModel = EfficientAdModel(
            teacher_out_channels=teacher_out_channels,
            model_size=model_size,
            backbone=self.backbone,
            teacher_pretrained=teacher_pretrained,
            padding=padding,
            pad_maps=pad_maps,
            hard_loss_mode=hard_loss_mode,
        )
        self.batch_size: int = batch_size
        self.hard_loss_mode: str = hard_loss_mode
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self._teacher_loaded_from_checkpoint = False

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Preserve the teacher restored by Lightning instead of loading defaults."""
        super().on_load_checkpoint(checkpoint)
        self._teacher_loaded_from_checkpoint = True

    @classmethod
    def load_from_checkpoint(cls, *args: Any, **kwargs: Any) -> "EfficientAd":
        """Restore embedded teacher weights without fetching initialization weights.

        Lightning constructs the model before calling ``on_load_checkpoint``.
        Override the saved bootstrap flag before construction, including for older
        checkpoints that saved ``teacher_pretrained=True``.
        """
        kwargs["teacher_pretrained"] = False
        return super().load_from_checkpoint(*args, **kwargs)

    def prepare_pretrained_model(self) -> None:
        """Prepare the pretrained teacher model.

        Downloads and loads pretrained weights for the teacher model if not already
        present.
        """
        if not self.backbone.startswith("pdn_"):
            from .backbones import load_default_teacher_weights

            load_default_teacher_weights(self.backbone, self.model.teacher)
            return

        pretrained_models_dir = Path("./pre_trained/")
        if not (pretrained_models_dir / "efficientad_pretrained_weights").is_dir():
            download_and_extract(pretrained_models_dir, WEIGHTS_DOWNLOAD_INFO)
        model_size_str = self.backbone.removeprefix("pdn_")
        teacher_path = (
            pretrained_models_dir / "efficientad_pretrained_weights" / f"pretrained_teacher_{model_size_str}.pth"
        )
        logger.info(f"Load pretrained teacher model from {teacher_path}")
        self.model.teacher.load_state_dict(
            torch.load(teacher_path, map_location=torch.device(self.device), weights_only=True),
        )

    def prepare_imagenette_data(
        self,
        image_size: tuple[int, int] | torch.Size,
        *,
        num_workers: int = 0,
    ) -> None:
        """Prepare ImageNette dataset transformations.

        Sets up data transforms and downloads ImageNette dataset if not present.

        Args:
            image_size (tuple[int, int] | torch.Size): Target image size for
                transforms.
            num_workers (int): Number of workers for the auxiliary ImageNette
                loader. Defaults to ``0``.
        """
        self.data_transforms_imagenet = Compose(
            [
                Resize((image_size[0] * 2, image_size[1] * 2)),
                RandomGrayscale(p=0.3),
                CenterCrop((image_size[0], image_size[1])),
                ToTensor(),
            ],
        )

        if not self.imagenet_dir.is_dir():
            download_and_extract(self.imagenet_dir, IMAGENETTE_DOWNLOAD_INFO)
        imagenet_dataset = ImageFolder(self.imagenet_dir, transform=self.data_transforms_imagenet)
        if len(imagenet_dataset) < self.batch_size:
            raise ValueError(
                "ImageNette 图片数量必须不少于训练 batch_size，"
                f"当前为 {len(imagenet_dataset)} < {self.batch_size}。"
            )
        self.imagenet_loader = DataLoader(
            imagenet_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        self.imagenet_iterator = iter(self.imagenet_loader)

    def _teacher_statistics_dataloader(self) -> DataLoader:
        """Build a non-dropping single-image loader for teacher statistics."""
        datamodule = self.trainer.datamodule
        dataset = getattr(datamodule, "train_data", None)
        if dataset is None:
            raise ValueError(
                "EfficientAd 需要 AnomalibDataModule.train_data 来建立教师统计 loader。"
            )
        num_workers = int(getattr(datamodule, "num_workers", 0))
        collate_fn = getattr(datamodule, "external_collate_fn", None)
        if collate_fn is None:
            collate_fn = getattr(dataset, "collate_fn", None)
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0,
        )

    @torch.no_grad()
    def teacher_channel_mean_std(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Calculate channel-wise mean and std of teacher model activations.

        Computes running mean and standard deviation of teacher model feature maps
        over the full dataset.

        Args:
            dataloader (DataLoader): Dataloader for the dataset.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - ``mean``: Channel-wise means of shape ``(1, C, 1, 1)``
                - ``std``: Channel-wise standard deviations of shape
                  ``(1, C, 1, 1)``

        Raises:
            ValueError: If no data is provided (``n`` remains ``None``).
        """
        arrays_defined = False
        n: torch.Tensor | None = None
        chanel_sum: torch.Tensor | None = None
        chanel_sum_sqr: torch.Tensor | None = None

        for batch in tqdm.tqdm(dataloader, desc="Calculate teacher channel mean & std", position=0, leave=True):
            y = self.model.teacher(batch.image.to(self.device))
            if self.backbone == "resnet18_layer2":
                # ResNet 的稀疏 ReLU 特征可含恒定通道；低方差统计使用双精度。
                y = y.double()
            if not arrays_defined:
                _, num_channels, _, _ = y.shape
                n = torch.zeros((num_channels,), dtype=torch.int64, device=y.device)
                stats_dtype = torch.float64 if self.backbone == "resnet18_layer2" else torch.float32
                chanel_sum = torch.zeros((num_channels,), dtype=stats_dtype, device=y.device)
                chanel_sum_sqr = torch.zeros((num_channels,), dtype=stats_dtype, device=y.device)
                arrays_defined = True

            n += y[:, 0].numel()
            chanel_sum += torch.sum(y, dim=[0, 2, 3])
            chanel_sum_sqr += torch.sum(y**2, dim=[0, 2, 3])

        if n is None:
            msg = "The value of 'n' cannot be None."
            raise ValueError(msg)

        channel_mean = chanel_sum / n

        variance = (chanel_sum_sqr / n) - (channel_mean**2)
        if self.backbone == "resnet18_layer2":
            variance = variance.clamp_min(0)
            inactive = variance == 0
            if inactive.all():
                raise ValueError("教师特征全部为恒定值，请增加有代表性的正常图片。")
            if inactive.any():
                logger.warning("ResNet 教师有 %d 个恒定通道，使用单位标准差归一化。", inactive.sum().item())
            # 恒定特征只减均值，不用极小 epsilon 放大未见过的激活。
            variance = torch.where(inactive, torch.ones_like(variance), variance)
        channel_std = torch.sqrt(variance).float()[None, :, None, None]
        channel_mean = channel_mean.float()[None, :, None, None]

        return {"mean": channel_mean, "std": channel_std}

    @torch.no_grad()
    def map_norm_quantiles(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """Calculate quantiles of student and autoencoder feature maps.

        Computes the 90% and 99.5% quantiles of the feature maps from both the
        student network and autoencoder on normal (good) validation samples.

        Args:
            dataloader (DataLoader): Validation dataloader.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - ``qa_st``: 90% quantile of student maps
                - ``qa_ae``: 90% quantile of autoencoder maps
                - ``qb_st``: 99.5% quantile of student maps
                - ``qb_ae``: 99.5% quantile of autoencoder maps
        """
        maps_st = []
        maps_ae = []
        logger.info("Calculate Validation Dataset Quantiles")
        for batch in tqdm.tqdm(dataloader, desc="Calculate Validation Dataset Quantiles", position=0, leave=True):
            good = batch.gt_label == 0
            if good.any():  # only use good images of validation set!
                map_st, map_ae = self.model.get_maps(
                    batch.image[good].to(self.device),
                    normalize=False,
                )
                maps_st.append(map_st)
                maps_ae.append(map_ae)

        qa_st, qb_st = self._get_quantiles_of_maps(maps_st)
        qa_ae, qb_ae = self._get_quantiles_of_maps(maps_ae)
        return {"qa_st": qa_st, "qa_ae": qa_ae, "qb_st": qb_st, "qb_ae": qb_ae}

    def _get_quantiles_of_maps(self, maps: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate quantiles of anomaly maps.

        Computes the 90% and 99.5% quantiles of the given anomaly maps. If total
        number of elements exceeds 16777216, uses a random subset.

        Args:
            maps (list[torch.Tensor]): List of anomaly maps.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - 90% quantile scalar
                - 99.5% quantile scalar
        """
        maps_flat = reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=0.9).to(self.device)
        qb = torch.quantile(maps_flat, q=0.995).to(self.device)
        return qa, qb

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure default pre-processor for EfficientAd.

        Note that ImageNet normalization is applied in the forward pass, not here.

        Args:
            image_size (tuple[int, int] | None, optional): Target image size.
                Defaults to ``(256, 256)``.

        Returns:
            PreProcessor: Configured pre-processor with resize transform.
        """
        image_size = image_size or (256, 256)
        transform = Compose([Resize(image_size, antialias=True)])
        return PreProcessor(transform=transform)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers for training.

        Sets up Adam optimizer with learning rate scheduler that decays LR by 0.1
        at 95% of training.

        Returns:
            dict: Dictionary containing:
                - ``optimizer``: Adam optimizer
                - ``lr_scheduler``: StepLR scheduler

        Raises:
            ValueError: If neither ``max_epochs`` nor ``max_steps`` is defined.
        """
        optimizer = torch.optim.Adam(
            list(self.model.student.parameters()) + list(self.model.ae.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.trainer.max_epochs < 0 and self.trainer.max_steps < 0:
            msg = "A finite number of steps or epochs must be defined"
            raise ValueError(msg)

        # lightning stops training when either 'max_steps' or 'max_epochs' is reached (earliest),
        # so actual training steps need to be determined here
        if self.trainer.max_epochs < 0:
            # max_epochs not set
            num_steps = self.trainer.max_steps
        elif self.trainer.max_steps < 0:
            # max_steps not set -> determine steps as 'max_epochs' * 'steps in a single training epoch'
            num_steps = self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())
        else:
            num_steps = min(
                self.trainer.max_steps,
                self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader()),
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(0.95 * num_steps)),
            gamma=0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_train_start(self) -> None:
        """Set up model before training begins.

        Performs the following steps:
        1. Validates training parameters (configured batch size, no normalization)
        2. Sets up pretrained teacher model
        3. Prepares ImageNette dataset
        4. Calculates channel statistics

        Raises:
            ValueError: If the data module batch size does not match the
                configured batch size or transforms contain normalization.
        """
        datamodule = self.trainer.datamodule
        if datamodule.train_batch_size != self.batch_size:
            msg = (
                "train_batch_size for EfficientAd must match the model batch_size; "
                f"got {datamodule.train_batch_size} and {self.batch_size}."
            )
            raise ValueError(msg)
        if datamodule.eval_batch_size != 1:
            raise ValueError(
                "eval_batch_size for EfficientAd must be 1 for validation, testing "
                "and heatmap generation; "
                f"got {datamodule.eval_batch_size}."
            )

        if self.pre_processor and extract_transforms_by_type(self.pre_processor.transform, Normalize):
            msg = "Transforms for EfficientAd should not contain Normalize."
            raise ValueError(msg)

        sample = next(iter(self.trainer.train_dataloader))
        image_size = sample.image.shape[-2:]
        if not self._teacher_loaded_from_checkpoint:
            self.prepare_pretrained_model()
        self.prepare_imagenette_data(image_size, num_workers=datamodule.num_workers)
        if not self.model.is_set(self.model.mean_std):
            channel_mean_std = self.teacher_channel_mean_std(self._teacher_statistics_dataloader())
            self.model.mean_std.update(channel_mean_std)

    def __getstate__(self) -> dict:
        """Modifies the python objects __getstate__ method.

        To ensure that the imagenet iterator instance will not get pickled
        when the model is saved, it needs to be removed from the objects dict.
        """
        state = self.__dict__.copy()
        state.pop("imagenet_iterator", None)
        return state

    def training_step(self, batch: Batch, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Perform training step.

        Computes student, autoencoder and combined losses using both the input
        batch and a batch from ImageNette.

        Args:
            batch (Batch): Input batch containing image and labels
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            dict[str, torch.Tensor]: Dictionary containing total loss
        """
        del args, kwargs  # These variables are not used.

        try:
            # infinite dataloader; [0] getting the image not the label
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)
        except StopIteration:
            self.imagenet_iterator = iter(self.imagenet_loader)
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)

        loss_st, loss_ae, loss_stae = self.model(batch=batch.image, batch_imagenet=batch_imagenet)

        loss = loss_st + loss_ae + loss_stae
        self.log("train_st", loss_st.item(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ae", loss_ae.item(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_stae", loss_stae.item(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def on_validation_start(self) -> None:
        """Calculate feature map statistics before validation.

        Computes quantiles of feature maps on validation set and updates model.
        """
        map_norm_quantiles = self.map_norm_quantiles(self.trainer.datamodule.val_dataloader())
        self.model.quantiles.update(map_norm_quantiles)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform validation step.

        Generates anomaly maps for the input batch.

        Args:
            batch (Batch): Input batch
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Batch with added predictions
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get trainer arguments.

        Returns:
            dict[str, Any]: Dictionary with trainer arguments:
                - ``num_sanity_val_steps``: 0
        """
        return {"num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Get model's learning type.

        Returns:
            LearningType: Always ``LearningType.ONE_CLASS``
        """
        return LearningType.ONE_CLASS
