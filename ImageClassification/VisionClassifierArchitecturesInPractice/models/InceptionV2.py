# GoogLeNet - Inception V2
# Change the 5x5 branch to two 3x3 branches.  (Similiar to VGG)
# Change the maxpool downsampling layer to a "downsample module".
# Architecture Based on GoogLeNet ,
# replace InceptionV1 Block with InceptionV2_block 
# & replace maxpool with "downsample module"

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_googlenet2(num_classes=1000):
    return InceptionNetV2(num_classes=num_classes)

class InceptionNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionNetV2, self).__init__()

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionV2_block(64, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionV2_block(256, 128, 128, 192, 32, 96, 64) 
        self.downsample3 = Inception_downsample(480, 480)   # V2

        self.inception4a = InceptionV2_block(480, 192, 96 , 208, 16, 48, 64)
        self.inception4b = InceptionV2_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionV2_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionV2_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionV2_block(528, 256, 160, 320, 32, 128, 128)
        self.downsample4 = Inception_downsample(832, 832)  # V2

        self.inception5a = InceptionV2_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionV2_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):  
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.downsample3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.downsample4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




class InceptionV2_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionV2_block,self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(   # V2
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=3, padding=1),
            conv_block(out_5x5, out_5x5, kernel_size=3, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_pool, kernel_size=1)
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(conv_block,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
# V2 Update : DownSampling Module
class Inception_downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch1 = conv_block(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1  
        )

        self.branch2 = conv_block(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1 
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x)
        ], 1)