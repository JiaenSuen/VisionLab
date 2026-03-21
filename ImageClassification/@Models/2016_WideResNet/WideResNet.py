import torch
import torch.nn as nn


'''
Key Changes (Compare to ResNet)

Distinguish between BasicBlock (full-channel widening) and Bottleneck (internal 3x3 widening only, consistent with the ImageNet WRN-50-2 experiments in the paper).
Update the ResNet categories and build functions to ensure consistency in channels/parameters with the original paper. Widening results in faster training and higher accuracy.
Retain the original architecture; these three adjustments are sufficient for reproduction.

'''


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, middle_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, middle_channels * Bottleneck.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(middle_channels * Bottleneck.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes, widen_factor=1):
        super().__init__()
        self.block = block  
        self.in_channels = 64
        conv1_out = 64 * widen_factor if not issubclass(block, Bottleneck) else 64
        self.conv1 = nn.Conv2d(image_channels, conv1_out, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(conv1_out)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = conv1_out  

        self.layer1 = self._make_layer(block, layers[0], 64, stride=1, widen_factor=widen_factor)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2, widen_factor=widen_factor)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2, widen_factor=widen_factor)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2, widen_factor=widen_factor)

        last_base = 512 * widen_factor if not issubclass(block, Bottleneck) else 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_base * block.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, base_out_channels, stride, widen_factor):
        layers = []
        expansion = block.expansion
        # middle / block_channels /stage_out
        if issubclass(block, Bottleneck):
            middle_channels = base_out_channels * widen_factor
            stage_out_channels = base_out_channels * expansion
        else:
            middle_channels = base_out_channels * widen_factor
            stage_out_channels = middle_channels * expansion

        identity_downsample = None
        if stride != 1 or self.in_channels != stage_out_channels:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, stage_out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(stage_out_channels)
            )

        # block
        if issubclass(block, Bottleneck):
            layers.append(block(self.in_channels, middle_channels, identity_downsample, stride))
        else:
            layers.append(block(self.in_channels, middle_channels, identity_downsample, stride))
        self.in_channels = stage_out_channels

        # blocks（stride=1）
        for _ in range(num_residual_blocks - 1):
            if issubclass(block, Bottleneck):
                layers.append(block(self.in_channels, middle_channels, None, 1))
            else:
                layers.append(block(self.in_channels, middle_channels, None, 1))

        return nn.Sequential(*layers)



def WideResNet18(num_classes=1000, img_channels=3, widen_factor=1):
    return ResNet(BasicBlock, [2, 2, 2, 2], img_channels, num_classes, widen_factor)

def WideResNet34(num_classes=1000, img_channels=3, widen_factor=1):
    return ResNet(BasicBlock, [3, 4, 6, 3], img_channels, num_classes, widen_factor)

def WideResNet50(num_classes=1000, img_channels=3, widen_factor=1):
    return ResNet(Bottleneck, [3, 4, 6, 3], img_channels, num_classes, widen_factor)

def WideResNet101(num_classes=1000, img_channels=3, widen_factor=1):
    return ResNet(Bottleneck, [3, 4, 23, 3], img_channels, num_classes, widen_factor)

def WideResNet152(num_classes=1000, img_channels=3, widen_factor=1):
    return ResNet(Bottleneck, [3, 8, 36, 3], img_channels, num_classes, widen_factor)

