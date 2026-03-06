import torch
import torch.nn as nn
def build_inception4_tiny(num_classes):
    return InceptionV4Tiny(num_classes)

 
# InceptionV4Tiny.py
import torch
import torch.nn as nn


 


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


# Inception V4 Tiny Stem (simplified with shared convs and concatenated branches, following original stem philosophy)
class TinyStemV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BasicConv2d(3, 32, 3, 1, 1)
        self.conv2 = BasicConv2d(32, 32, 3, 1, 1)
        self.branch1 = BasicConv2d(32, 64, 3, 2, 1)
        self.branch2 = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return torch.cat([self.branch1(x), self.branch2(x)], 1)  # Output: 16x16x96 for 32x32 input


# Inception-A for V4 tiny (branches with 1x1, 3x3, double 3x3, avgpool; reduced channels following original proportions)
class TinyInceptionA_V4(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

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

        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BasicConv2d(in_ch, 32, 1)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)  # 32+32+48+32=144


# Reduction-A for V4 tiny (maxpool, direct 3x3 s2, multi-conv s2; reduced following original)
class TinyReductionA_V4(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.b1 = nn.MaxPool2d(3, 2, 1)

        self.b2 = BasicConv2d(in_ch, 96, 3, 2, 1)

        self.b3 = nn.Sequential(
            BasicConv2d(in_ch, 48, 1),
            BasicConv2d(48, 64, 3, 1, 1),
            BasicConv2d(64, 96, 3, 2, 1)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], 1)  # in_ch + 96 + 96


# Inception-B for V4 tiny (1x1, asymmetric conv chains, avgpool; reduced)
class TinyInceptionB_V4(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.b1 = BasicConv2d(in_ch, 64, 1)

        self.b2 = nn.Sequential(
            BasicConv2d(in_ch, 48, 1),
            BasicConv2d(48, 48, (1, 3), 1, (0, 1)),
            BasicConv2d(48, 64, (3, 1), 1, (1, 0))
        )

        self.b3 = nn.Sequential(
            BasicConv2d(in_ch, 48, 1),
            BasicConv2d(48, 48, (1, 3), 1, (0, 1)),
            BasicConv2d(48, 48, (3, 1), 1, (1, 0)),
            BasicConv2d(48, 48, (1, 3), 1, (0, 1)),
            BasicConv2d(48, 64, (3, 1), 1, (1, 0))
        )

        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BasicConv2d(in_ch, 32, 1)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)  # 64+64+64+32=224


# Reduction-B for V4 tiny (maxpool, conv chains with asymmetry s2; reduced)
class TinyReductionB_V4(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.b1 = nn.MaxPool2d(3, 2, 1)

        self.b2 = nn.Sequential(
            BasicConv2d(in_ch, 64, 1),
            BasicConv2d(64, 96, 3, 2, 1)
        )

        self.b3 = nn.Sequential(
            BasicConv2d(in_ch, 64, 1),
            BasicConv2d(64, 64, (1, 3), 1, (0, 1)),
            BasicConv2d(64, 96, (3, 1), 2, (1, 0))
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], 1)  # in_ch + 96 + 96


# Inception-C for V4 tiny (1x1, split asymmetric branches with serial and parallel convs, avgpool; reduced to fit tiny model)
class TinyInceptionC_V4(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.b1 = BasicConv2d(in_ch, 64, 1)

        self.b2_1 = BasicConv2d(in_ch, 96, 1)
        self.b2_2a = BasicConv2d(96, 64, (1, 3), 1, (0, 1))
        self.b2_2b = BasicConv2d(96, 64, (3, 1), 1, (1, 0))

        self.b3_1 = BasicConv2d(in_ch, 96, 1)
        self.b3_2a = BasicConv2d(96, 112, (1, 3), 1, (0, 1))
        self.b3_2b = BasicConv2d(112, 128, (3, 1), 1, (1, 0))
        self.b3_3a = BasicConv2d(128, 64, (1, 3), 1, (0, 1))
        self.b3_3b = BasicConv2d(128, 64, (3, 1), 1, (1, 0))

        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BasicConv2d(in_ch, 32, 1)
        )

    def forward(self, x):
        b2 = self.b2_1(x)
        b2_a = self.b2_2a(b2)
        b2_b = self.b2_2b(b2)
        b2 = torch.cat([b2_a, b2_b], 1)  # 64+64=128

        b3 = self.b3_1(x)
        b3 = self.b3_2a(b3)  # to 112
        b3 = self.b3_2b(b3)  # to 128
        b3_a = self.b3_3a(b3)
        b3_b = self.b3_3b(b3)
        b3 = torch.cat([b3_a, b3_b], 1)  # 64+64=128

        return torch.cat([self.b1(x), b2, b3, self.b4(x)], 1)  # 64+128+128+32=352


# InceptionV4Tiny (reduced blocks and channels for CIFAR 32x32; follows V4 structure with stem, A/B/C modules, reductions)
class InceptionV4Tiny(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = TinyStemV4()  # Out: 96 channels at 16x16

        self.a1 = TinyInceptionA_V4(96)
        self.a2 = TinyInceptionA_V4(144)

        self.redA = TinyReductionA_V4(144)  # Out: 336 at 8x8

        self.b1 = TinyInceptionB_V4(336)
        self.b2 = TinyInceptionB_V4(224)
        self.b3 = TinyInceptionB_V4(224)

        self.redB = TinyReductionB_V4(224)  # Out: 416 at 4x4

        self.c1 = TinyInceptionC_V4(416)
        self.c2 = TinyInceptionC_V4(352)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(352, num_classes)

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
    model = InceptionV4Tiny(num_classes=10)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # CIFAR-sized input
    with torch.no_grad():
        output = model(dummy_input)
        print(f"out shape: {output.shape}")
        print(f"Output [5] :\n{output[0, :5]}")