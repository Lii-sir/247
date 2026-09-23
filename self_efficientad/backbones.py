"""Feature extractors used by the local EfficientAD implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    out_channels: int
    feature_stride: int


BACKBONE_SPECS = {
    "pdn_small": BackboneSpec("pdn_small", 384, 4),
    "pdn_medium": BackboneSpec("pdn_medium", 384, 4),
    "resnet18_layer2": BackboneSpec("resnet18_layer2", 128, 8),
    "resnet18_layer3": BackboneSpec("resnet18_layer3", 256, 16),
}


def get_backbone_spec(name: str) -> BackboneSpec:
    try:
        return BACKBONE_SPECS[name]
    except KeyError as error:
        choices = ", ".join(BACKBONE_SPECS)
        raise ValueError(f"未知 backbone {name!r}，可选：{choices}") from error


class ResNet18Layer2(nn.Module):
    """ResNet-18 stem through layer2, preserving a spatial feature map."""

    def __init__(self, *, output_channels: int, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        source = resnet18(weights=weights)
        self.features = nn.Sequential(
            source.conv1,
            source.bn1,
            source.relu,
            source.maxpool,
            source.layer1,
            source.layer2,
        )
        self.projection = (
            nn.Identity()
            if output_channels == 128
            else nn.Conv2d(128, output_channels, kernel_size=1)
        )
        self.output_channels = output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return self.projection(self.features((x - mean) / std))


class ResNet18Layer3(nn.Module):
    """ResNet-18 through layer3; the student adds a spatial output head."""

    def __init__(self, *, output_channels: int, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        source = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(
            source.conv1, source.bn1, source.relu, source.maxpool,
            source.layer1, source.layer2, source.layer3,
        )
        # Keep the teacher's pretrained feature space untouched. The student's
        # trainable head produces both EfficientAD branches directly.
        self.head = (nn.Identity() if output_channels == 256
                     else nn.Conv2d(256, output_channels, kernel_size=3, padding=1))
        self.output_channels = output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return self.head(self.features((x - mean) / std))


def build_resnet18_layer2_pair(*, teacher_pretrained: bool = False) -> tuple[nn.Module, nn.Module]:
    """Build the ResNet teacher and its two-branch student."""

    teacher = ResNet18Layer2(output_channels=128, pretrained=teacher_pretrained).eval()
    student = ResNet18Layer2(output_channels=256, pretrained=False)
    return teacher, student


def build_resnet18_layer3_pair(*, teacher_pretrained: bool = False) -> tuple[nn.Module, nn.Module]:
    teacher = ResNet18Layer3(output_channels=256, pretrained=teacher_pretrained).eval()
    student = ResNet18Layer3(output_channels=512, pretrained=False)
    return teacher, student


def build_backbone_pair(
    name: str,
    *,
    teacher_out_channels: int | None = None,
    padding: bool = False,
    teacher_pretrained: bool = False,
) -> tuple[nn.Module, nn.Module, int]:
    """Build frozen-teacher and two-branch student feature extractors."""

    from .torch_model import MediumPatchDescriptionNetwork, SmallPatchDescriptionNetwork

    spec = get_backbone_spec(name)
    out_channels = spec.out_channels if teacher_out_channels is None else teacher_out_channels
    if not isinstance(out_channels, int) or isinstance(out_channels, bool) or out_channels < 1:
        raise ValueError("teacher_out_channels 必须是正整数。")
    if name == "pdn_small":
        teacher = SmallPatchDescriptionNetwork(out_channels=out_channels, padding=padding).eval()
        student = SmallPatchDescriptionNetwork(out_channels=out_channels * 2, padding=padding)
    elif name == "pdn_medium":
        teacher = MediumPatchDescriptionNetwork(out_channels=out_channels, padding=padding).eval()
        student = MediumPatchDescriptionNetwork(out_channels=out_channels * 2, padding=padding)
    elif name == "resnet18_layer2":
        if teacher_out_channels not in (None, spec.out_channels):
            raise ValueError(
                f"{name} 的 teacher_out_channels 固定为 {spec.out_channels}，"
                f"收到 {teacher_out_channels}。"
            )
        teacher, student = build_resnet18_layer2_pair(teacher_pretrained=teacher_pretrained)
    elif name == "resnet18_layer3":
        if teacher_out_channels not in (None, spec.out_channels):
            raise ValueError(
                f"{name} 的 teacher_out_channels 固定为 {spec.out_channels}，"
                f"收到 {teacher_out_channels}。"
            )
        teacher, student = build_resnet18_layer3_pair(teacher_pretrained=teacher_pretrained)
    else:  # pragma: no cover - get_backbone_spec gives the public error
        raise ValueError(f"Unsupported backbone: {name}")
    return teacher, student, out_channels


def load_default_teacher_weights(name: str, teacher: nn.Module) -> None:
    """Load the framework-provided pretrained teacher for a non-PDN backbone."""

    if name == "resnet18_layer3":
        from torchvision.models import ResNet18_Weights, resnet18

        source = resnet18(weights=ResNet18_Weights.DEFAULT)
        pretrained_features = nn.Sequential(
            source.conv1, source.bn1, source.relu, source.maxpool,
            source.layer1, source.layer2, source.layer3,
        )
        teacher.features.load_state_dict(pretrained_features.state_dict())
        return
    if name != "resnet18_layer2":
        raise ValueError(f"{name} 没有 torchvision 默认教师权重。")
    pretrained = ResNet18Layer2(output_channels=128, pretrained=True)
    teacher.load_state_dict(pretrained.state_dict())
