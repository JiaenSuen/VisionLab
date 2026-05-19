# XceptionTiny.py
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
For CIFAR-sized images,
reducing channel width is more effective than reducing depth 
because the computational cost of depthwise separable convolution is 
dominated by the pointwise 1x1 convolution.
'''


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.pw(self.dw(x)))


class EntryBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.sep1 = SeparableConv2d(in_ch, out_ch)
        self.pool = nn.MaxPool2d(3, 2, 1)

        self.res = nn.Conv2d(in_ch, out_ch, 1, 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        res = self.bn(self.res(x))
        x = F.relu(self.sep1(x))
        x = self.pool(x)
        return x + res


class MiddleBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.sep = SeparableConv2d(ch, ch)

    def forward(self, x):
        return x + F.relu(self.sep(x))


class XceptionTiny(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 3, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # 16x16
        self.entry = EntryBlock(64, 128)  # 8x8

        # 8x8
        self.middle = nn.Sequential(
            MiddleBlock(128),
            MiddleBlock(128),
        )

        # 8x8 → 4x4
        self.exit = EntryBlock(128, 256)

        self.sep = SeparableConv2d(256, 256)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        x = F.relu(self.sep(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    

def xception_tiny(num_classes=10):
    return XceptionTiny(num_classes=num_classes)
