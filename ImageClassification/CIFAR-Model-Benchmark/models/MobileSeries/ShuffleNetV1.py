import torch
import torch.nn as nn
from typing import List


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Shuffle channels across groups to enable cross-group information flow.

    Args:
        x: Input tensor with shape [N, C, H, W].
        groups: Number of channel groups.

    Returns:
        Tensor with the same shape as the input.
    """
    batch_size, channels, height, width = x.shape

    if channels % groups != 0:
        raise ValueError(
            f"channels ({channels}) must be divisible by groups ({groups})."
        )

    channels_per_group = channels // groups

    # [N, C, H, W] -> [N, groups, channels_per_group, H, W]
    x = x.reshape(
        batch_size,
        groups,
        channels_per_group,
        height,
        width,
    )

    # Exchange group and intra-group channel dimensions.
    x = x.transpose(1, 2).contiguous()

    # Restore [N, C, H, W].
    return x.reshape(batch_size, channels, height, width)


class ShuffleNetUnit(nn.Module):
    """
    ShuffleNet V1 unit.

    For stride=1:
        The transformed branch is added to the identity branch.

    For stride=2:
        The transformed branch is concatenated with an average-pooled shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 3,
        stride: int = 1,
        first_group: bool = False,
        reduction_ratio: int = 4,
    ) -> None:
        super().__init__()

        if stride not in (1, 2):
            raise ValueError("stride must be either 1 or 2.")

        if stride == 1 and in_channels != out_channels:
            raise ValueError(
                "For stride=1, in_channels must equal out_channels."
            )

        self.stride = stride
        self.groups = groups

        # In a stride-2 unit, the shortcut contributes in_channels channels.
        branch_out_channels = (
            out_channels
            if stride == 1
            else out_channels - in_channels
        )

        if branch_out_channels <= 0:
            raise ValueError(
                "For stride=2, out_channels must be greater than in_channels."
            )

        bottleneck_channels = out_channels // reduction_ratio
        first_pw_groups = 1 if first_group else groups

        if in_channels % first_pw_groups != 0:
            raise ValueError(
                "in_channels must be divisible by first_pw_groups."
            )

        if bottleneck_channels % first_pw_groups != 0:
            raise ValueError(
                "bottleneck_channels must be divisible by first_pw_groups."
            )

        if bottleneck_channels % groups != 0:
            raise ValueError(
                "bottleneck_channels must be divisible by groups."
            )

        if branch_out_channels % groups != 0:
            raise ValueError(
                "branch_out_channels must be divisible by groups."
            )

        # Reduce channels with a pointwise group convolution.
        self.group_conv_reduce = nn.Sequential(
            nn.Conv2d(
                in_channels,
                bottleneck_channels,
                kernel_size=1,
                groups=first_pw_groups,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        # Extract spatial features independently for each channel.
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=bottleneck_channels,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
        )

        # Expand channels with another pointwise group convolution.
        self.group_conv_expand = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                branch_out_channels,
                kernel_size=1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(branch_out_channels),
        )

        self.shortcut = (
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            if stride == 2
            else nn.Identity()
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.group_conv_reduce(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv(out)
        out = self.group_conv_expand(out)

        if self.stride == 1:
            out = out + identity
        else:
            out = torch.cat((out, identity), dim=1)

        return self.activation(out)


class ShuffleNetV1CIFAR(nn.Module):
    """
    ShuffleNet V1 adapted for CIFAR-10 and CIFAR-100.

    CIFAR adaptation:
        - Uses a 3x3 stem convolution with stride 1.
        - Removes the initial max-pooling layer.
        - Preserves the original ShuffleNet V1 stage design.
    """

    CHANNEL_CONFIGS = {
        1: [24, 144, 288, 576],
        2: [24, 200, 400, 800],
        3: [24, 240, 480, 960],
        4: [24, 272, 544, 1088],
        8: [24, 384, 768, 1536],
    }

    def __init__(
        self,
        num_classes: int,
        groups: int = 3,
        stage_repeats: List[int] = [4, 8, 4],
    ) -> None:
        super().__init__()

        if groups not in self.CHANNEL_CONFIGS:
            raise ValueError(
                f"Unsupported groups={groups}. "
                f"Choose from {list(self.CHANNEL_CONFIGS.keys())}."
            )

        if len(stage_repeats) != 3:
            raise ValueError("stage_repeats must contain three values.")

        stage_channels = self.CHANNEL_CONFIGS[groups]

        # CIFAR images are only 32x32, so the stem keeps full resolution.
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=stage_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stage_channels[0]),
            nn.ReLU(inplace=True),
        )

        self.stage2 = self._make_stage(
            in_channels=stage_channels[0],
            out_channels=stage_channels[1],
            repeat=stage_repeats[0],
            groups=groups,
            first_stage=True,
        )

        self.stage3 = self._make_stage(
            in_channels=stage_channels[1],
            out_channels=stage_channels[2],
            repeat=stage_repeats[1],
            groups=groups,
            first_stage=False,
        )

        self.stage4 = self._make_stage(
            in_channels=stage_channels[2],
            out_channels=stage_channels[3],
            repeat=stage_repeats[2],
            groups=groups,
            first_stage=False,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(stage_channels[3], num_classes)

        self._initialize_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        repeat: int,
        groups: int,
        first_stage: bool,
    ) -> nn.Sequential:
        if repeat < 1:
            raise ValueError("Each stage must contain at least one unit.")

        layers = [
            # The first unit downsamples the feature map.
            ShuffleNetUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=groups,
                stride=2,
                first_group=first_stage,
            )
        ]

        # Remaining units preserve spatial resolution.
        for _ in range(repeat - 1):
            layers.append(
                ShuffleNetUnit(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    groups=groups,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)      # 32x32
        x = self.stage2(x)    # 16x16
        x = self.stage3(x)    # 8x8
        x = self.stage4(x)    # 4x4

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        return self.classifier(x)


def build_shufflenet_v1(num_classes: int) -> ShuffleNetV1CIFAR:
    """
    Build the default CIFAR ShuffleNet V1 model.

    The default configuration follows the groups=3 variant from
    the original ShuffleNet V1 paper.
    """
    return ShuffleNetV1CIFAR(
        num_classes=num_classes,
        groups=3,
        stage_repeats=[4, 8, 4],
    )


if __name__ == "__main__":
    model = build_shufflenet_v1(num_classes=10)

    sample_input = torch.randn(2, 3, 32, 32)
    logits = model(sample_input)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(model)
    print(f"Input shape:  {sample_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameters:   {parameter_count / 1e6:.2f} M")