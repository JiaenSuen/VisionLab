# Xception.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# Depthwise Separable Convolution
class SeparableConv2d(nn.Module):
    """
    Depthwise separable convolution:
    depthwise conv -> pointwise conv
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding,
            groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


# Entry Flow Block
class EntryBlock(nn.Module):
    """
    Entry flow block with residual connection and downsampling
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.sepconv1 = SeparableConv2d(in_ch, out_ch)
        self.sepconv2 = SeparableConv2d(out_ch, out_ch)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.residual = nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        res = self.bn(self.residual(x))

        x = F.relu(x)
        x = self.sepconv1(x)
        x = F.relu(x)
        x = self.sepconv2(x)
        x = self.pool(x)

        return x + res


# Middle Flow Block
class MiddleBlock(nn.Module):
    """
    Middle flow block with three separable conv layers and residual
    """
    def __init__(self, ch):
        super().__init__()

        self.sepconv1 = SeparableConv2d(ch, ch)
        self.sepconv2 = SeparableConv2d(ch, ch)
        self.sepconv3 = SeparableConv2d(ch, ch)

    def forward(self, x):
        res = x

        x = F.relu(x)
        x = self.sepconv1(x)
        x = F.relu(x)
        x = self.sepconv2(x)
        x = F.relu(x)
        x = self.sepconv3(x)

        return x + res


# Exit Flow Block
class ExitBlock(nn.Module):
    """
    Exit flow block with channel expansion and downsampling
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.sepconv1 = SeparableConv2d(in_ch, in_ch)
        self.sepconv2 = SeparableConv2d(in_ch, out_ch)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.residual = nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        res = self.bn(self.residual(x))

        x = F.relu(x)
        x = self.sepconv1(x)
        x = F.relu(x)
        x = self.sepconv2(x)
        x = self.pool(x)

        return x + res


# Xception Network
class Xception(nn.Module):
    """
    Original Xception architecture
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Entry Flow
        self.block1 = EntryBlock(64, 128)
        self.block2 = EntryBlock(128, 256)
        self.block3 = EntryBlock(256, 728)

        # Middle Flow (8 blocks)
        self.middle = nn.Sequential(*[MiddleBlock(728) for _ in range(8)])

        # Exit Flow
        self.exit = ExitBlock(728, 1024)
        self.sepconv1 = SeparableConv2d(1024, 1536)
        self.sepconv2 = SeparableConv2d(1536, 2048)

        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Stem
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Entry
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle
        x = self.middle(x)

        # Exit
        x = self.exit(x)
        x = F.relu(self.sepconv1(x))
        x = F.relu(self.sepconv2(x))

        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)