# GoogLeNet - Inception V2
# Change the 5x5 branch to two 3x3 branches.  (Similiar to VGG)
# Change the maxpool downsampling layer to a "downsample module".
# Architecture Based on GoogLeNet ,
# replace InceptionV1 Block with InceptionV2_block 
# & replace maxpool with "downsample module"

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_inception2(num_classes=1000):
    return InceptionV2(num_classes=num_classes)

class InceptionV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = Stem()

        self.in3a = InceptionV2Block(192, 64, 64, 64, 64, 96, 32)
        self.in3b = InceptionV2Block(256, 64, 64, 96, 64, 96, 64)

        self.reduction3 = Reduction(320, 128, 160, 64, 96)

        self.in4a = InceptionV2Block(576, 224, 64, 96, 96, 128, 128)
        self.in4b = InceptionV2Block(576, 192, 96, 128, 96, 128, 128)

        self.reduction4 = Reduction(576, 128, 192, 128, 192)

        self.in5a = InceptionV2Block(960, 352, 192, 320, 160, 224, 128)
        self.in5b = InceptionV2Block(1024, 352, 192, 320, 192, 224, 128)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.in3a(x)
        x = self.in3b(x)
        x = self.reduction3(x)

        x = self.in4a(x)
        x = self.in4b(x)
        x = self.reduction4(x)

        x = self.in5a(x)
        x = self.in5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)



class InceptionV2Block(nn.Module):
    def __init__(self, in_ch, c1, c3r, c3, c5r, c5, pool):
        super().__init__()

        self.branch1 = ConvBN(in_ch, c1, 1)

        self.branch2 = nn.Sequential(
            ConvBN(in_ch, c3r, 1),
            ConvBN(c3r, c3, 3, 1, 1)
        )

        # 5x5 -> 3x3 + 3x3
        self.branch3 = nn.Sequential(
            ConvBN(in_ch, c5r, 1),
            ConvBN(c5r, c5, 3, 1, 1),
            ConvBN(c5, c5, 3, 1, 1)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            ConvBN(in_ch, pool, 1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], 1)



class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBN(3, 32, 3, 2, 1),
            ConvBN(32, 32, 3, 1, 1),
            ConvBN(32, 64, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            ConvBN(64, 80, 1),
            ConvBN(80, 192, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1)
        )

    def forward(self, x):
        return self.stem(x)



# V2 Update : DownSampling Module
class Reduction(nn.Module):
    def __init__(self, in_ch, k, l, m, n):
        super().__init__()

        # 1x1 -> 3x3 stride=2
        self.branch1 = nn.Sequential(
            ConvBN(in_ch, k, 1),
            ConvBN(k, l, 3, 2, 1)
        )

        # 1x1 -> 3x3 -> 3x3 stride=2
        self.branch2 = nn.Sequential(
            ConvBN(in_ch, m, 1),
            ConvBN(m, n, 3, 1, 1),
            ConvBN(n, n, 3, 2, 1)
        )

        # pooling branch
        self.branch3 = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x)
        ], 1)