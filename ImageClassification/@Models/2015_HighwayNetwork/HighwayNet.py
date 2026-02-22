import torch
import torch.nn as nn
import torch.nn.functional as F


def build_HighwayNet(num_classes=1000):
    return DeepStageHighwayNet(num_classes=num_classes)

# memory-efficient
class ConvHighwayBlock(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(
            channels, channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)

        self.gate = nn.Conv2d(
            channels, channels, 3,
            stride=stride, padding=1, bias=True
        )
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)))
        t = torch.sigmoid(self.gate(x))
        return x + t * (h - x)



# Stage 
class HighwayStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, downsample):
        super().__init__()

        layers = []

        if downsample:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        else:
            if in_channels != out_channels:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )

        for _ in range(num_blocks):
            layers.append(ConvHighwayBlock(out_channels))

        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


# Deep Stage HighwayNet ( 23-layer )
class DeepStageHighwayNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Stem 
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.stage1 = HighwayStage(64, 64, num_blocks=2, downsample=False)
        self.stage2 = HighwayStage(64, 128, num_blocks=2, downsample=True)
        self.stage3 = HighwayStage(128, 256, num_blocks=3, downsample=True)
        self.stage4 = HighwayStage(256, 512, num_blocks=2, downsample=True)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)      # 224 → 56
        x = self.stage1(x)    # 56
        x = self.stage2(x)    # 28
        x = self.stage3(x)    # 14
        x = self.stage4(x)    # 7
        x = self.pool(x).flatten(1)
        return self.fc(x)