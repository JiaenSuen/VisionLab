"""MobileNet v1 in PyTorch.

Reference:
    Howard et al., "MobileNets: Efficient Convolutional Neural Networks
    for Mobile Vision Applications", arXiv:1704.04861, 2017.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn


def _make_divisible_channels(channels: int, width_mult: float) -> int:
    """Scale channels while keeping at least one channel."""
    if width_mult <= 0:
        raise ValueError("width_mult must be greater than 0.")
    return max(1, int(channels * width_mult))


class ConvBNReLU(nn.Sequential):
    """Standard convolution followed by batch norm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Sequential):
    """Depthwise 3x3 convolution and pointwise 1x1 convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class MobileNetV1(nn.Module):
    """MobileNet v1 for ImageNet-sized inputs."""

    # Each tuple is (output channels, stride).
    _BLOCK_CONFIG: List[Tuple[int, int]] = [
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1),
    ]

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        stem_stride: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be greater than 0.")
        if stem_stride not in (1, 2):
            raise ValueError("stem_stride must be 1 or 2.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        stem_channels = _make_divisible_channels(32, width_mult)
        layers: List[nn.Module] = [
            ConvBNReLU(3, stem_channels, kernel_size=3, stride=stem_stride)
        ]

        in_channels = stem_channels
        for base_channels, stride in self._BLOCK_CONFIG:
            out_channels = _make_divisible_channels(base_channels, width_mult)
            layers.append(
                DepthwiseSeparableConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                )
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_channels, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize layers with standard CNN defaults."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


class MobileNetV1CIFAR(MobileNetV1):
    """CIFAR adaptation that preserves the paper's block schedule.

    The 32x32 input uses stride 1 in the stem to avoid early information loss.
    The remaining depthwise-separable stages follow the original architecture.
    """

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            width_mult=width_mult,
            stem_stride=1,
            dropout=dropout,
        )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    parameters = (
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )
    return sum(parameter.numel() for parameter in parameters)


@torch.inference_mode()
def _smoke_test(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int],
    expected_classes: int,
) -> None:
    """Run a minimal inference test."""
    model.eval()
    inputs = torch.randn(*input_shape)
    outputs = model(inputs)

    expected_shape = (input_shape[0], expected_classes)
    if tuple(outputs.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {tuple(outputs.shape)}, "
            f"expected {expected_shape}."
        )

    if not torch.isfinite(outputs).all():
        raise RuntimeError("Model output contains NaN or Inf.")

    print(f"{model.__class__.__name__}")
    print(f"  Input shape : {tuple(inputs.shape)}")
    print(f"  Output shape: {tuple(outputs.shape)}")
    print(f"  Parameters  : {count_parameters(model):,}")
    print("  Inference   : OK")


if __name__ == "__main__":
    torch.manual_seed(0)

    imagenet_model = MobileNetV1(num_classes=1000, width_mult=1.0)
    cifar_model = MobileNetV1CIFAR(num_classes=10, width_mult=1.0)

    _smoke_test(
        model=imagenet_model,
        input_shape=(2, 3, 224, 224),
        expected_classes=1000,
    )
    print()
    _smoke_test(
        model=cifar_model,
        input_shape=(2, 3, 32, 32),
        expected_classes=10,
    )
