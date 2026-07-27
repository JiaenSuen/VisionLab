import torch
import torch.nn as nn
from typing import List


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Shuffle channels across groups to enable cross-group information flow.

    Input shape:
        x: [batch_size, channels, height, width]
    """
    batch_size, channels, height, width = x.shape

    if channels % groups != 0:
        raise ValueError(
            f"Channels ({channels}) must be divisible by groups ({groups})."
        )

    channels_per_group = channels // groups

    # [N, C, H, W] -> [N, groups, channels_per_group, H, W]
    x = x.reshape(
        batch_size,
        groups,
        channels_per_group,
        height,
        width
    )

    # Exchange the group and intra-group channel dimensions
    x = x.transpose(1, 2).contiguous()

    # Restore the original four-dimensional tensor shape
    x = x.reshape(batch_size, channels, height, width)

    return x


class ShuffleNetUnit(nn.Module):
    """
    ShuffleNet V1 building block.

    Stride 1:
        The main branch is added to the identity branch.

    Stride 2:
        The main branch is concatenated with an average-pooled shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 3,
        stride: int = 1,
        first_group: bool = False,
        reduction_ratio: int = 4
    ):
        super().__init__()

        if stride not in (1, 2):
            raise ValueError("Stride must be either 1 or 2.")

        if stride == 1 and in_channels != out_channels:
            raise ValueError(
                "For stride=1, in_channels must equal out_channels "
                "because residual addition is used."
            )

        self.stride = stride
        self.groups = groups

        # In a stride-2 unit, the shortcut contributes in_channels
        # after pooling, so the main branch only generates the remaining
        # output channels.
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

        # The first pointwise convolution in Stage 2 is not grouped
        # in the original ShuffleNet V1 design.
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

        # 1x1 Pointwise Group Convolution:
        # reduce channel dimensions at a lower computational cost
        self.group_conv_reduce = nn.Sequential(
            nn.Conv2d(
                in_channels,
                bottleneck_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=first_pw_groups,
                bias=False
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )

        # 3x3 Depthwise Convolution:
        # independently extracts spatial features for each channel
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=bottleneck_channels,
                bias=False
            ),
            nn.BatchNorm2d(bottleneck_channels)
        )

        # 1x1 Pointwise Group Convolution:
        # expand the channel representation
        self.group_conv_expand = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                branch_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(branch_out_channels)
        )

        if stride == 2:
            # The shortcut is downsampled without introducing
            # additional learnable parameters.
            self.shortcut = nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.group_conv_reduce(x)

        # Shuffle channels after grouped pointwise convolution
        # to prevent isolated group-wise information pathways.
        out = channel_shuffle(out, self.groups)

        out = self.depthwise_conv(out)
        out = self.group_conv_expand(out)

        if self.stride == 1:
            # Residual addition preserves the spatial and channel dimensions.
            out = out + identity
        else:
            # Concatenation simultaneously preserves shortcut information
            # and increases the number of output channels.
            out = torch.cat((out, identity), dim=1)

        return self.activation(out)


class ShuffleNetV1(nn.Module):
    """
    ShuffleNet V1 for image classification.

    Default configuration:
        groups = 3
        output channels = [24, 240, 480, 960]
        stage repeats = [4, 8, 4]
    """

    CHANNEL_CONFIGS = {
        1: [24, 144, 288, 576],
        2: [24, 200, 400, 800],
        3: [24, 240, 480, 960],
        4: [24, 272, 544, 1088],
        8: [24, 384, 768, 1536]
    }

    def __init__(
        self,
        num_classes: int = 1000,
        groups: int = 3,
        stage_repeats: List[int] = [4, 8, 4]
    ):
        super().__init__()

        if groups not in self.CHANNEL_CONFIGS:
            raise ValueError(
                f"Unsupported group count: {groups}. "
                f"Choose from {list(self.CHANNEL_CONFIGS.keys())}."
            )

        if len(stage_repeats) != 3:
            raise ValueError("stage_repeats must contain three values.")

        stage_channels = self.CHANNEL_CONFIGS[groups]

        # Initial spatial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=stage_channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(stage_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.stage2 = self._make_stage(
            in_channels=stage_channels[0],
            out_channels=stage_channels[1],
            repeat=stage_repeats[0],
            groups=groups,
            first_stage=True
        )

        self.stage3 = self._make_stage(
            in_channels=stage_channels[1],
            out_channels=stage_channels[2],
            repeat=stage_repeats[1],
            groups=groups,
            first_stage=False
        )

        self.stage4 = self._make_stage(
            in_channels=stage_channels[2],
            out_channels=stage_channels[3],
            repeat=stage_repeats[2],
            groups=groups,
            first_stage=False
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
        first_stage: bool
    ) -> nn.Sequential:
        if repeat < 1:
            raise ValueError("Each stage must contain at least one unit.")

        layers = [
            # The first unit of each stage performs spatial downsampling.
            ShuffleNetUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                groups=groups,
                stride=2,
                first_group=first_stage
            )
        ]

        # The remaining units preserve feature resolution and use
        # residual addition.
        for _ in range(repeat - 1):
            layers.append(
                ShuffleNetUnit(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    groups=groups,
                    stride=1
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        return self.classifier(x)


if __name__ == "__main__":
    model = ShuffleNetV1(
        num_classes=1000,
        groups=3
    )

    sample_input = torch.randn(2, 3, 224, 224)
    logits = model(sample_input)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(model)
    print(f"Input shape:  {sample_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameters:   {parameter_count / 1e6:.2f} M")