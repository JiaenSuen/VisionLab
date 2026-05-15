# Inception-ResNet-v2tiny.py
import torch
import torch.nn as nn


def build_inception_resnet_v2_tiny(num_classes):
    return InceptionResNetV2Tiny(num_classes)


# Basic Convolutional Layer
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# Linear Convolutional Layer (no BN or ReLU, for residual projection)
class LinearConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


# Inception-ResNet Stem tiny (simplified sequential with maxpool, following original philosophy)
class TinyStemResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            BasicConv2d(3, 32, 3, 1, 1),
            BasicConv2d(32, 32, 3, 1, 1),
            BasicConv2d(32, 64, 3, 2, 1),
            nn.MaxPool2d(3, 2, 1)
        )

    def forward(self, x):
        return self.stem(x)  # Out: 8x8x64 for 32x32 input


# Inception-ResNet-A tiny (branches + linear projection + scaled residual add + post-add ReLU)
class TinyInceptionResA(nn.Module):
    def __init__(self, in_ch, scale=0.17):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        self.b1 = BasicConv2d(in_ch, 32, 1)

        self.b2 = nn.Sequential(
            BasicConv2d(in_ch, 32, 1),
            BasicConv2d(32, 32, 3, 1, 1)
        )

        self.b3 = nn.Sequential(
            BasicConv2d(in_ch, 32, 1),
            BasicConv2d(32, 48, 3, 1, 1),
            BasicConv2d(48, 48, 3, 1, 1)
        )

        cat_ch = 32 + 32 + 48
        self.lin = LinearConv2d(cat_ch, in_ch)

    def forward(self, x):
        branches = torch.cat([self.b1(x), self.b2(x), self.b3(x)], 1)
        branches = self.lin(branches)
        return self.relu(x + branches * self.scale)


# Reduction-A for Res tiny (concat branches with s2, no residual)
class TinyReductionA_Res(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.b1 = nn.MaxPool2d(3, 2, 1)

        self.b2 = BasicConv2d(in_ch, 96, 3, 2, 1)

        self.b3 = nn.Sequential(
            BasicConv2d(in_ch, 64, 1),
            BasicConv2d(64, 96, 3, 1, 1),
            BasicConv2d(96, 96, 3, 2, 1)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], 1)


# Inception-ResNet-B tiny (asymmetric branches + linear + scaled residual + post-add ReLU)
class TinyInceptionResB(nn.Module):
    def __init__(self, in_ch, scale=0.10):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        self.b1 = BasicConv2d(in_ch, 64, 1)

        self.b2 = nn.Sequential(
            BasicConv2d(in_ch, 64, 1),
            BasicConv2d(64, 64, (1, 3), 1, (0, 1)),
            BasicConv2d(64, 96, (3, 1), 1, (1, 0))
        )

        cat_ch = 64 + 96
        self.lin = LinearConv2d(cat_ch, in_ch)

    def forward(self, x):
        branches = torch.cat([self.b1(x), self.b2(x)], 1)
        branches = self.lin(branches)
        return self.relu(x + branches * self.scale)


# Reduction-B for Res tiny (multiple s2 branches with asymmetry and extra conv path)
class TinyReductionB_Res(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.b1 = nn.MaxPool2d(3, 2, 1)

        self.b2 = nn.Sequential(
            BasicConv2d(in_ch, 96, 1),
            BasicConv2d(96, 128, 3, 2, 1)
        )

        self.b3 = nn.Sequential(
            BasicConv2d(in_ch, 96, 1),
            BasicConv2d(96, 96, (1, 3), 1, (0, 1)),
            BasicConv2d(96, 128, (3, 1), 2, (1, 0))
        )

        self.b4 = nn.Sequential(
            BasicConv2d(in_ch, 96, 1),
            BasicConv2d(96, 96, 3, 1, 1),
            BasicConv2d(96, 128, 3, 2, 1)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)


# Inception-ResNet-C tiny (asymmetric chain + linear + scaled residual + post-add ReLU)
class TinyInceptionResC(nn.Module):
    def __init__(self, in_ch, scale=0.20):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        self.b1 = BasicConv2d(in_ch, 96, 1)

        self.b2 = nn.Sequential(
            BasicConv2d(in_ch, 96, 1),
            BasicConv2d(96, 96, (1, 3), 1, (0, 1)),
            BasicConv2d(96, 128, (3, 1), 1, (1, 0))
        )

        cat_ch = 96 + 128
        self.lin = LinearConv2d(cat_ch, in_ch)

    def forward(self, x):
        branches = torch.cat([self.b1(x), self.b2(x)], 1)
        branches = self.lin(branches)
        return self.relu(x + branches * self.scale)


# InceptionResNetV2Tiny (residuals with scaling, reduced blocks/channels for CIFAR; follows v2 structure with post-add activations)
class InceptionResNetV2Tiny(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = TinyStemResNet()  # Out: 64 channels at 8x8

        self.a1 = TinyInceptionResA(64, 0.17)
        self.a2 = TinyInceptionResA(64, 0.17)

        self.redA = TinyReductionA_Res(64)  # Out: 256 at 4x4

        self.b1 = TinyInceptionResB(256, 0.10)
        self.b2 = TinyInceptionResB(256, 0.10)
        self.b3 = TinyInceptionResB(256, 0.10)

        self.redB = TinyReductionB_Res(256)  # Out: 640 at 2x2

        self.c1 = TinyInceptionResC(640, 0.20)
        self.c2 = TinyInceptionResC(640, 0.20)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(640, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.a1(x)
        x = self.a2(x)

        x = self.redA(x)

        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)

        x = self.redB(x)

        x = self.c1(x)
        x = self.c2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


if __name__ == "__main__":
    model = InceptionResNetV2Tiny(num_classes=10)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # CIFAR-sized input
    with torch.no_grad():
        output = model(dummy_input)
        print(f"out shape: {output.shape}")
        print(f"Output [5] :\n{output[0, :5]}")