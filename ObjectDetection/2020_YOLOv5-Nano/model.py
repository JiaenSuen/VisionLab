"""
YOLOv5nano.py

A YOLOv5-Nano-style detector reimplemented in plain PyTorch.
This file does not use the Ultralytics package.

Output format:
    [
        large_object_output,   # shape: (B, 3, S/32, S/32, 5 + num_classes)
        medium_object_output,  # shape: (B, 3, S/16, S/16, 5 + num_classes)
        small_object_output,   # shape: (B, 3, S/8,  S/8,  5 + num_classes)
    ]

This output order is intentionally compatible with the existing dataset/loss/utils
pipeline where ANCHORS[0] corresponds to the smallest grid / largest objects.
"""

import torch
import torch.nn as nn


class Conv(nn.Module):
    """Standard Conv-BN-SiLU block used in YOLOv5-style networks."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """YOLOv5-style bottleneck block."""

    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1)
        self.use_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.use_shortcut else y


class C3(nn.Module):
    """YOLOv5 C3 block."""

    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv3 = Conv(2 * hidden_channels, out_channels, kernel_size=1, stride=1)
        self.blocks = nn.Sequential(
            *[
                Bottleneck(
                    hidden_channels,
                    hidden_channels,
                    shortcut=shortcut,
                    expansion=1.0,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast block used by YOLOv5."""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = Conv(hidden_channels * 4, out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class PredictionHead(nn.Module):
    """Prediction head that reshapes output to (B, 3, S, S, 5 + num_classes)."""

    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.pred = nn.Conv2d(
            in_channels,
            num_anchors * (num_classes + 5),
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        batch_size, _, grid_h, grid_w = x.shape
        x = self.pred(x)
        x = x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_h, grid_w)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        return x


class YOLOv5Nano(nn.Module):
    """
    YOLOv5-Nano-style detector.

    This is not an Ultralytics weight-compatible implementation.
    It is a PyTorch model designed to work with the existing
    anchor-based YOLO training pipeline in this project.
    """

    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes

        # Backbone.
        self.stem = Conv(in_channels, 16, kernel_size=6, stride=2, padding=2)  # 320 -> 160
        self.down1 = Conv(16, 32, kernel_size=3, stride=2)                    # 160 -> 80
        self.c3_1 = C3(32, 32, num_blocks=1)
        self.down2 = Conv(32, 64, kernel_size=3, stride=2)                    # 80 -> 40
        self.c3_2 = C3(64, 64, num_blocks=1)
        self.down3 = Conv(64, 128, kernel_size=3, stride=2)                   # 40 -> 20
        self.c3_3 = C3(128, 128, num_blocks=2)
        self.down4 = Conv(128, 256, kernel_size=3, stride=2)                  # 20 -> 10
        self.c3_4 = C3(256, 256, num_blocks=1)
        self.sppf = SPPF(256, 256)

        # FPN top-down path.
        self.reduce_p5 = Conv(256, 128, kernel_size=1, stride=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3_fpn_p4 = C3(256, 128, num_blocks=1, shortcut=False)
        self.reduce_p4 = Conv(128, 64, kernel_size=1, stride=1)
        self.c3_fpn_p3 = C3(128, 64, num_blocks=1, shortcut=False)

        # PAN bottom-up path.
        self.down_p3 = Conv(64, 64, kernel_size=3, stride=2)
        self.c3_pan_p4 = C3(192, 128, num_blocks=1, shortcut=False)
        self.down_p4 = Conv(128, 128, kernel_size=3, stride=2)
        self.c3_pan_p5 = C3(256, 256, num_blocks=1, shortcut=False)

        # Detection heads.
        self.head_large = PredictionHead(256, num_classes)   # S/32
        self.head_medium = PredictionHead(128, num_classes)  # S/16
        self.head_small = PredictionHead(64, num_classes)    # S/8

    def forward(self, x):
        # Backbone.
        x = self.stem(x)
        x = self.c3_1(self.down1(x))
        p3 = self.c3_2(self.down2(x))
        p4 = self.c3_3(self.down3(p3))
        p5 = self.sppf(self.c3_4(self.down4(p4)))

        # FPN.
        p5_reduced = self.reduce_p5(p5)
        fpn_p4 = self.c3_fpn_p4(torch.cat((self.up(p5_reduced), p4), dim=1))
        p4_reduced = self.reduce_p4(fpn_p4)
        fpn_p3 = self.c3_fpn_p3(torch.cat((self.up(p4_reduced), p3), dim=1))

        # PAN.
        pan_p4 = self.c3_pan_p4(torch.cat((self.down_p3(fpn_p3), fpn_p4), dim=1))
        pan_p5 = self.c3_pan_p5(torch.cat((self.down_p4(pan_p4), p5_reduced), dim=1))

        # Return order:
        #   output[0] -> largest objects, smallest grid
        #   output[1] -> medium objects
        #   output[2] -> smallest objects, largest grid
        out_large = self.head_large(pan_p5)
        out_medium = self.head_medium(pan_p4)
        out_small = self.head_small(fpn_p3)
        return [out_large, out_medium, out_small]


if __name__ == "__main__":
    model = YOLOv5Nano(in_channels=3, num_classes=20)
    x = torch.randn(2, 3, 320, 320)
    outputs = model(x)
    assert outputs[0].shape == (2, 3, 10, 10, 25)
    assert outputs[1].shape == (2, 3, 20, 20, 25)
    assert outputs[2].shape == (2, 3, 40, 40, 25)
    params = sum(p.numel() for p in model.parameters())
    print("YOLOv5Nano output shapes:", [o.shape for o in outputs])
    print(f"Total parameters: {params:,}")
