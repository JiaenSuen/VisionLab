import torch
import torch.nn as nn


# Pre-Activation Basic Block
class PreActBasic(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = shortcut

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)

        if self.shortcut is not None:
            identity = self.shortcut(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += identity
        return out


# Pre-Activation Bottleneck
class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_ch, out_ch * self.expansion, kernel_size=1, bias=False)

        self.shortcut = shortcut

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)

        if self.shortcut is not None:
            identity = self.shortcut(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out += identity
        return out


# ResNet V2 Main Network
class ResNetV2(nn.Module):
    def __init__(self, block, layers, img_ch=3, num_classes=1000):
        super().__init__()

        self.input_ch = 64

        self.stem_conv = nn.Conv2d(img_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._build_stage(block, layers[0], 64, stride=1)
        self.stage2 = self._build_stage(block, layers[1], 128, stride=2)
        self.stage3 = self._build_stage(block, layers[2], 256, stride=2)
        self.stage4 = self._build_stage(block, layers[3], 512, stride=2)

        self.final_bn = nn.BatchNorm2d(512 * block.expansion)
        self.final_relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_pool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.final_bn(x)
        x = self.final_relu(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def _build_stage(self, block, num_blocks, out_ch, stride):
        shortcut = None
        layers = []

        if stride != 1 or self.input_ch != out_ch * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.input_ch, out_ch * block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers.append(block(self.input_ch, out_ch, stride, shortcut))
        self.input_ch = out_ch * block.expansion

        for _ in range(num_blocks - 1):
            layers.append(block(self.input_ch, out_ch))

        return nn.Sequential(*layers)


# Factory Functions
def ResNetV2_18(num_classes, img_ch=3):
    return ResNetV2(PreActBasic, [2,2,2,2], img_ch, num_classes)

def ResNetV2_34(num_classes, img_ch=3):
    return ResNetV2(PreActBasic, [3,4,6,3], img_ch, num_classes)

def ResNetV2_50(num_classes, img_ch=3):
    return ResNetV2(PreActBottleneck, [3,4,6,3], img_ch, num_classes)

def ResNetV2_101(num_classes, img_ch=3):
    return ResNetV2(PreActBottleneck, [3,4,23,3], img_ch, num_classes)

def ResNetV2_152(num_classes, img_ch=3):
    return ResNetV2(PreActBottleneck, [3,8,36,3], img_ch, num_classes)