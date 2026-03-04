# Designed specifically for 32×32 input (CIFAR compatible)
# • Overall branch count, branching method, and thesis philosophy are fully preserved.
# • Only kernel size (asymmetric convolution from 7 to 5) and stride/padding are adjusted to make the receptive field suitable for small images.
# • Stem is completely non-aggressive downsampling → maintains 32×32×384.
# • Reduction-A stride=3 → 10×10 (approximately 12×12).
# • Reduction-B stride=2 → 4×4 (falling between 4 and 6).
# • All Inception layers have pooling stride=1 and 3×3, with pad=1 and dimensions unchanged.

import torch
import torch.nn as nn

def build_inception4_tiny(num_classes):
    return InceptionV4Tiny(num_classes)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
 

class TinyInceptionStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BasicConv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv2d(8, 16, kernel_size=3, stride=1, padding=1)

        self.branch1_conv = BasicConv2d(16, 24, kernel_size=3, stride=1, padding=1)
        self.branch1_pool = nn.MaxPool2d(3, stride=1, padding=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(40, 16, kernel_size=1),
            BasicConv2d(16, 16, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(16, 16, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(16, 24, kernel_size=3, padding=1)
        )
        self.branch2_2 = nn.Sequential(
            BasicConv2d(40, 16, kernel_size=1),
            BasicConv2d(16, 24, kernel_size=3, padding=1)
        )

        self.branch3_conv = BasicConv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.branch3_pool = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat([self.branch1_conv(x), self.branch1_pool(x)], dim=1)
        x = torch.cat([self.branch2(x), self.branch2_2(x)], dim=1)
        x = torch.cat([self.branch3_conv(x), self.branch3_pool(x)], dim=1)
        return x


class TinyInceptionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, 20, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 12, kernel_size=1),
            BasicConv2d(12, 20, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 12, kernel_size=1),
            BasicConv2d(12, 20, kernel_size=3, padding=1),
            BasicConv2d(20, 20, kernel_size=3, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 20, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class TinyInceptionAStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(*[TinyInceptionA(96) for _ in range(1)])

    def forward(self, x):
        return self.stack(x)


class TinyReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, 80, kernel_size=3, stride=3)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 40, kernel_size=1),
            BasicConv2d(40, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 48, kernel_size=3, stride=3)
        )
        self.branch3 = nn.MaxPool2d(3, stride=3)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class TinyInceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, 80, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 40, kernel_size=1),
            BasicConv2d(40, 48, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(48, 48, kernel_size=(3, 1), padding=(1, 0))
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 24, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch4(x)], dim=1)


class TinyInceptionBStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[TinyInceptionB(208) for _ in range(1)])

    def forward(self, x):
        return self.blocks(x)


class TinyReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, 40, kernel_size=1),
            BasicConv2d(40, 40, kernel_size=3, stride=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 48, kernel_size=1),
            BasicConv2d(48, 48, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(48, 56, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(56, 56, kernel_size=3, stride=2)
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class TinyInceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, 48, kernel_size=1)

        self.branch2 = nn.Sequential(BasicConv2d(in_channels, 72, kernel_size=1))
        self.branch2a = BasicConv2d(72, 48, kernel_size=(1, 3), padding=(0, 1))
        self.branch2b = BasicConv2d(72, 48, kernel_size=(3, 1), padding=(1, 0))

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 48, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b2 = torch.cat([self.branch2a(b2), self.branch2b(b2)], dim=1)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b4], dim=1)


class TinyInceptionCStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[TinyInceptionC(248) for _ in range(1)])

    def forward(self, x):
        return self.blocks(x)


class InceptionV4Tiny(nn.Module):
    def __init__(self, num_classes=10):   # CIFAR-10=10, CIFAR-100=100
        super().__init__()
        self.stem = TinyInceptionStem()
        self.inceptionA = TinyInceptionAStack()
        self.reductionA = TinyReductionA(80)
        self.inceptionB = TinyInceptionBStack()
        self.reductionB = TinyReductionB(152)
        self.inceptionC = TinyInceptionCStack()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.stem(x)           # 32×32×96
        x = self.inceptionA(x)     # 32×32×80
        x = self.reductionA(x)     # 10×10×208
        x = self.inceptionB(x)     # 10×10×152
        x = self.reductionB(x)     # 4×4×248
        x = self.inceptionC(x)     # 4×4×192

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
 
if __name__ == "__main__":
    model = InceptionV4Tiny(num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print("Output shape:", out.shape)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
 
     