from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpecification:
    name: str
    channels: tuple[int, ...]
    residual: bool
    dropout: float
    augmentation: bool
    noise_std: float
    weight_decay: float
    architecture: str = "compact_cnn"
    pretrained: bool = False


FORMAL_SPECIFICATIONS = (
    ModelSpecification("tiny_baseline", (16, 32), False, 0.0, False, 0.0, 0.0),
    ModelSpecification("deeper_baseline", (16, 32, 64), False, 0.0, False, 0.0, 0.0),
    ModelSpecification("regularised_cnn", (16, 32, 64), False, 0.30, True, 0.02, 1e-4),
    ModelSpecification("compact_residual", (16, 32, 64), True, 0.30, True, 0.02, 1e-4),
    ModelSpecification("wide_residual", (24, 48, 96), True, 0.30, True, 0.02, 1e-4),
    ModelSpecification(
        "pretrained_mobilenet_v3_small",
        (),
        True,
        0.20,
        True,
        0.01,
        1e-4,
        architecture="mobilenet_v3_small",
        pretrained=True,
    ),
)


class ConvStage(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )


class ResidualStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, downsample: bool) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.activation(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.activation(outputs + residual)


class CompactCNN(nn.Module):
    def __init__(self, channels: tuple[int, ...], residual: bool, dropout: float) -> None:
        super().__init__()
        if residual:
            stages: list[nn.Module] = [
                nn.Sequential(
                    nn.Conv2d(1, channels[0], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels[0]),
                    nn.ReLU(inplace=True),
                ),
                ResidualStage(channels[0], channels[0], downsample=False),
            ]
            for input_channels, output_channels in zip(channels[:-1], channels[1:]):
                stages.append(ResidualStage(input_channels, output_channels, downsample=True))
        else:
            stages = []
            input_channels = 1
            for output_channels in channels:
                stages.append(ConvStage(input_channels, output_channels))
                input_channels = output_channels
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels[-1], 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(self.pool(features)).squeeze(1)


class PretrainedMobileNetV3Small(nn.Module):
    """ImageNet-pretrained MobileNetV3-Small adapted to one-channel X-rays."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        try:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except ImportError as error:
            raise RuntimeError(
                "torchvision is required for the pretrained MobileNetV3-Small benchmark."
            ) from error
        self.network = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        input_features = self.network.classifier[-1].in_features
        self.network.classifier[2] = nn.Dropout(p=dropout, inplace=True)
        self.network.classifier[-1] = nn.Linear(input_features, 1)
        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raw_grayscale = (inputs * 0.25 + 0.5).clamp(0.0, 1.0)
        rgb = raw_grayscale.repeat(1, 3, 1, 1)
        normalised_rgb = (rgb - self.imagenet_mean) / self.imagenet_std
        return self.network(normalised_rgb).squeeze(1)


def build_model(specification: ModelSpecification) -> nn.Module:
    if specification.architecture == "mobilenet_v3_small":
        return PretrainedMobileNetV3Small(dropout=specification.dropout)
    if specification.architecture == "compact_cnn":
        return CompactCNN(
            channels=specification.channels,
            residual=specification.residual,
            dropout=specification.dropout,
        )
    raise ValueError(f"Unsupported architecture: {specification.architecture}")


def count_parameters(model: nn.Module) -> int:
    # The assignment concerns model size, so frozen and trainable parameters both count.
    return sum(parameter.numel() for parameter in model.parameters())


def last_convolution(model: nn.Module) -> nn.Conv2d:
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not convolutions:
        raise RuntimeError("The model contains no convolutional layer.")
    return convolutions[-1]
