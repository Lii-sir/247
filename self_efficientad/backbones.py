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
    "resnet50_layer1": BackboneSpec("resnet50_layer1", 256, 4),
    "resnet50_layer1v2": BackboneSpec("resnet50_layer1v2", 256, 4),
    "resnet50_layer2": BackboneSpec("resnet50_layer2", 512, 8),
    "resnet50_layer3": BackboneSpec("resnet50_layer3", 1024, 16),
}


def get_backbone_spec(name: str) -> BackboneSpec:
    try:
        return BACKBONE_SPECS[name]
    except KeyError as error:
        choices = ", ".join(BACKBONE_SPECS)
        raise ValueError(f"未知 backbone {name!r}，可选：{choices}") from error


def crop_output_border(module: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Align teacher targets to a valid 3x3 student head without changing the trunk."""
    border = getattr(module, "output_border", 0)
    if not border:
        return features
    if min(features.shape[-2:]) <= 2 * border:
        raise ValueError("ResNet 最终特征图太小，无法裁剪输出边界；请增大输入尺寸。")
    return features[..., border:-border, border:-border]


def resolve_resnet_feature_mode(mode: str | None, version: int) -> str:
    if mode is None:
        return "valid" if version == 2 else "native"
    if mode not in ("valid", "native"):
        raise ValueError("resnet_feature_mode 必须为 valid 或 native。")
    return mode


def resolve_teacher_output_activation(backbone: str, activation: str | None) -> str:
    """New layer1v2 runs omit the final ReLU; checkpoint readers supply legacy relu."""
    if activation is None:
        activation = "none" if backbone == "resnet50_layer1v2" else "relu"
    if activation not in ("relu", "none"):
        raise ValueError("resnet_teacher_output_activation 必须为 relu 或 none。")
    if activation == "none" and backbone != "resnet50_layer1v2":
        raise ValueError("去掉 Teacher 最终 ReLU 目前仅支持 resnet50_layer1v2。")
    return activation


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
        return crop_output_border(self, self.projection(self.features((x - mean) / std)))


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
        return crop_output_border(self, self.head(self.features((x - mean) / std)))


class ResNet50Features(nn.Module):
    """Native ImageNet trunk with optional output-boundary controls."""

    def __init__(
        self,
        *,
        layer: int,
        pretrained: bool = False,
        remove_final_activation: bool = False,
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        if type(layer) is not int or layer not in (1, 2, 3):
            raise ValueError("ResNet-50 feature layer must be 1, 2 or 3.")
        source = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = nn.Sequential(
            source.conv1, source.bn1, source.relu, source.maxpool,
            *(getattr(source, f"layer{index}") for index in range(1, layer + 1)),
        )
        self.output_channels = 256 * 2 ** (layer - 1)
        self.remove_final_activation = remove_final_activation

    @staticmethod
    def _forward_bottleneck_without_output_activation(
        block: nn.Module, x: torch.Tensor,
    ) -> torch.Tensor:
        """Run a torchvision Bottleneck without its final residual ReLU.

        The Bottleneck's ``relu`` module is reused after conv1, conv2 and the
        residual addition, so replacing ``block.relu`` with ``Identity`` would
        incorrectly remove the two internal activations as well.
        """
        identity = x

        out = block.conv1(x)
        out = block.bn1(out)
        out = block.relu(out)

        out = block.conv2(out)
        out = block.bn2(out)
        out = block.relu(out)

        out = block.conv3(out)
        out = block.bn3(out)

        if block.downsample is not None:
            identity = block.downsample(x)

        out += identity
        return out

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Accept ImageNet-normalized input; return features before border cropping."""
        if not self.remove_final_activation:
            features = self.features(x)
        else:
            # Keep every earlier stage/block unchanged.  Only the final
            # Bottleneck of the selected output stage omits its output ReLU.
            x = self.features[:-1](x)
            final_stage = self.features[-1]
            for block in final_stage[:-1]:
                x = block(x)
            features = self._forward_bottleneck_without_output_activation(final_stage[-1], x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        features = self.forward_features((x - mean) / std)
        return crop_output_border(self, features)


class ResNet50Layer3(ResNet50Features):
    """Compatibility constructor; parameter names and shapes remain unchanged."""

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__(layer=3, pretrained=pretrained)


class WideResNetStudent(nn.Module):
    """Random student with 2x widths in the stem and every retained stage.

    Stage depths and downsampling match the teacher. A linear spatial head
    permits negative predictions of channel-standardized teacher features.
    """

    def __init__(self, backbone: str) -> None:
        super().__init__()
        from torchvision.models.resnet import BasicBlock, Bottleneck

        spec = get_backbone_spec(backbone)
        if not backbone.startswith("resnet"):
            raise ValueError("WideResNetStudent requires a ResNet backbone.")
        block = Bottleneck if backbone.startswith("resnet50_") else BasicBlock
        depths = (3, 4, 6) if block is Bottleneck else (2, 2, 2)
        stage_count = int(backbone.rsplit("_layer", 1)[1])
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


class ResNet50Layer1V2Student(nn.Module):
    """Standard ResNet-50 layer1 with a two-branch output projection.

    The retained ResNet trunk keeps the native 64-channel stem and 256-channel
    layer1 widths. Only the final spatial projection expands the output to
    ``2 * 256`` channels, matching the PDN student convention.
    """

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet50

        source = resnet50(weights=None)
        self.features = nn.Sequential(
            source.conv1, source.bn1, source.relu, source.maxpool, source.layer1,
        )
        self.output_channels = 512
        self.head = nn.Conv2d(256, self.output_channels, kernel_size=3, padding=1)

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
    resnet_feature_mode: str | None = None,
    resnet_teacher_output_activation: str | None = None,
) -> tuple[nn.Module, nn.Module, int]:
    """Build frozen-teacher and two-branch student feature extractors."""

    from .torch_model import MediumPatchDescriptionNetwork, SmallPatchDescriptionNetwork

    spec = get_backbone_spec(name)
    if type(resnet_architecture_version) is not int or resnet_architecture_version not in (1, 2):
        raise ValueError("resnet_architecture_version 必须为 1（旧结构）或 2（整体扩宽）。")
    mode = resolve_resnet_feature_mode(resnet_feature_mode, resnet_architecture_version)
    activation = resolve_teacher_output_activation(name, resnet_teacher_output_activation)
    if name.startswith("resnet") and mode == "valid" and resnet_architecture_version == 1:
        raise ValueError("valid 输出模式需要 resnet_architecture_version=2；旧结构请使用 native。")
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
        if name.startswith("resnet50_"):
            if resnet_architecture_version == 1:
                raise ValueError(f"{name} 只支持 resnet_architecture_version=2。")
            if name == "resnet50_layer3":
                teacher = ResNet50Layer3(pretrained=teacher_pretrained).eval()
            else:
                teacher_layer = 1 if name == "resnet50_layer1v2" else int(name.rsplit("_layer", 1)[1])
                teacher = ResNet50Features(
                    layer=teacher_layer,
                    pretrained=teacher_pretrained,
                    remove_final_activation=activation == "none",
                ).eval()
        elif name == "resnet18_layer2":
            teacher = ResNet18Layer2(output_channels=128, pretrained=teacher_pretrained).eval()
        else:
            teacher = ResNet18Layer3(output_channels=256, pretrained=teacher_pretrained).eval()
        if resnet_architecture_version == 2:
            student = (ResNet50Layer1V2Student() if name == "resnet50_layer1v2"
                       else WideResNetStudent(name))
        elif name == "resnet18_layer2":
            student = ResNet18Layer2(output_channels=256, pretrained=False)
        else:
            student = ResNet18Layer3(output_channels=512, pretrained=False)
        if mode == "valid":
            # Align the valid student projection with a one-cell teacher crop.
            # The independently configured activation does not change geometry.
            student.head.padding = (0, 0)
            teacher.output_border = 1
    else:  # pragma: no cover - get_backbone_spec gives the public error
        raise ValueError(f"Unsupported backbone: {name}")
    return teacher, student, out_channels


def load_default_teacher_weights(name: str, teacher: nn.Module) -> None:
    """Load the framework-provided pretrained teacher for a non-PDN backbone."""

    if name in BACKBONE_SPECS and name.startswith("resnet50_"):
        layer = 1 if name == "resnet50_layer1v2" else int(name.rsplit("_layer", 1)[1])
        pretrained = ResNet50Features(
            layer=layer,
            pretrained=True,
            remove_final_activation=teacher.remove_final_activation,
        )
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
