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
    "resnet50_layer3": BackboneSpec("resnet50_layer3", 1024, 16),
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


class ResNet50Layer3(nn.Module):
    """Native ImageNet feature space; no randomly initialized teacher head."""

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        source = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = nn.Sequential(
            source.conv1, source.bn1, source.relu, source.maxpool,
            source.layer1, source.layer2, source.layer3,
        )
        self.output_channels = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return self.features((x - mean) / std)


class WideResNetStudent(nn.Module):
    """Random student with 2x widths in the stem and every retained stage.

    Stage depths and downsampling match the teacher. A linear spatial head
    permits negative predictions of channel-standardized teacher features.
    """

    def __init__(self, backbone: str) -> None:
        super().__init__()
        from torchvision.models.resnet import BasicBlock, Bottleneck

        spec = get_backbone_spec(backbone)
        block = Bottleneck if backbone == "resnet50_layer3" else BasicBlock
        depths = (3, 4, 6) if block is Bottleneck else (2, 2, 2)
        stage_count = 2 if backbone == "resnet18_layer2" else 3
        modules = [
            nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
        in_channels = 128
        for index, (planes, depth) in enumerate(zip((128, 256, 512), depths)):
            if index >= stage_count:
                break
            stride = 1 if index == 0 else 2
            out_channels = planes * block.expansion
            downsample = None
            if stride != 1 or in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            layers = [block(in_channels, planes, stride=stride, downsample=downsample)]
            layers.extend(block(out_channels, planes) for _ in range(depth - 1))
            modules.append(nn.Sequential(*layers))
            in_channels = out_channels
        self.features = nn.Sequential(*modules)
        for module in self.features.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.output_channels = 2 * spec.out_channels
        self.head = nn.Conv2d(in_channels, self.output_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return self.head(self.features((x - mean) / std))


def build_resnet18_layer2_pair(
    *, teacher_pretrained: bool = False, resnet_architecture_version: int = 2,
) -> tuple[nn.Module, nn.Module]:
    """Build a pair using the same versioned defaults as the training model."""
    teacher, student, _ = build_backbone_pair(
        "resnet18_layer2", teacher_pretrained=teacher_pretrained,
        resnet_architecture_version=resnet_architecture_version,
    )
    return teacher, student


def build_resnet18_layer3_pair(
    *, teacher_pretrained: bool = False, resnet_architecture_version: int = 2,
) -> tuple[nn.Module, nn.Module]:
    teacher, student, _ = build_backbone_pair(
        "resnet18_layer3", teacher_pretrained=teacher_pretrained,
        resnet_architecture_version=resnet_architecture_version,
    )
    return teacher, student


def build_backbone_pair(
    name: str,
    *,
    teacher_out_channels: int | None = None,
    padding: bool = False,
    teacher_pretrained: bool = False,
    resnet_architecture_version: int = 2,
) -> tuple[nn.Module, nn.Module, int]:
    """Build frozen-teacher and two-branch student feature extractors."""

    from .torch_model import MediumPatchDescriptionNetwork, SmallPatchDescriptionNetwork

    spec = get_backbone_spec(name)
    if type(resnet_architecture_version) is not int or resnet_architecture_version not in (1, 2):
        raise ValueError("resnet_architecture_version 必须为 1（旧结构）或 2（整体扩宽）。")
    out_channels = spec.out_channels if teacher_out_channels is None else teacher_out_channels
    if not isinstance(out_channels, int) or isinstance(out_channels, bool) or out_channels < 1:
        raise ValueError("teacher_out_channels 必须是正整数。")
    if name == "pdn_small":
        teacher = SmallPatchDescriptionNetwork(out_channels=out_channels, padding=padding).eval()
        student = SmallPatchDescriptionNetwork(out_channels=out_channels * 2, padding=padding)
    elif name == "pdn_medium":
        teacher = MediumPatchDescriptionNetwork(out_channels=out_channels, padding=padding).eval()
        student = MediumPatchDescriptionNetwork(out_channels=out_channels * 2, padding=padding)
    elif name.startswith("resnet"):
        if teacher_out_channels not in (None, spec.out_channels):
            raise ValueError(
                f"{name} 的 teacher_out_channels 固定为 {spec.out_channels}，"
                f"收到 {teacher_out_channels}。"
            )
        if name == "resnet50_layer3":
            if resnet_architecture_version == 1:
                raise ValueError("resnet50_layer3 只支持 resnet_architecture_version=2。")
            teacher = ResNet50Layer3(pretrained=teacher_pretrained).eval()
        elif name == "resnet18_layer2":
            teacher = ResNet18Layer2(output_channels=128, pretrained=teacher_pretrained).eval()
        else:
            teacher = ResNet18Layer3(output_channels=256, pretrained=teacher_pretrained).eval()
        if resnet_architecture_version == 2:
            student = WideResNetStudent(name)
        elif name == "resnet18_layer2":
            student = ResNet18Layer2(output_channels=256, pretrained=False)
        else:
            student = ResNet18Layer3(output_channels=512, pretrained=False)
    else:  # pragma: no cover - get_backbone_spec gives the public error
        raise ValueError(f"Unsupported backbone: {name}")
    return teacher, student, out_channels


def load_default_teacher_weights(name: str, teacher: nn.Module) -> None:
    """Load the framework-provided pretrained teacher for a non-PDN backbone."""

    if name == "resnet50_layer3":
        pretrained = ResNet50Layer3(pretrained=True)
        teacher.load_state_dict(pretrained.state_dict())
        return
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
