import torch
import torch.nn as nn


def build_inception_resnet_v2_tiny(num_classes=1000):
    return InceptionResNetV2Tiny(num_classes=num_classes)



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


class InceptionStemTiny(nn.Module):
    def __init__(self):
        super().__init__()
        # 32x32 -> 32x32
        self.conv1 = BasicConv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Reduction 1: 32x32 -> 16x16  
        self.branch1_conv = BasicConv2d(64, 48, kernel_size=3, stride=2, padding=1)
        self.branch1_pool = nn.MaxPool2d(3, stride=2, padding=1)
        # Branch2 at 16x16 
        self.branch2 = nn.Sequential(
            BasicConv2d(112, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=(7,1), padding=(3,0)),
            BasicConv2d(32, 32, kernel_size=(1,7), padding=(0,3)),
            BasicConv2d(32, 48, kernel_size=3, padding=1)
        )
        self.branch2_2 = nn.Sequential(
            BasicConv2d(112, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1)
        )
        # Reduction 2: 16x16 -> 8x8
        self.branch3_conv = BasicConv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.branch3_pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Red1
        x = torch.cat([self.branch1_conv(x), self.branch1_pool(x)], dim=1)  # 112ch

        # Branch2  
        x = torch.cat([self.branch2(x), self.branch2_2(x)], dim=1)  # 96ch

        # Red2
        x = torch.cat([self.branch3_conv(x), self.branch3_pool(x)], dim=1)  # 192ch @ 8x8
        return x


 
class ResInceptionATiny(nn.Module):
    def __init__(self, in_channels=192, scale=0.1):
        super().__init__()
        self.scale = scale
        self.branch1 = BasicConv2d(in_channels, 16, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 16, kernel_size=1),
            BasicConv2d(16, 16, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 16, kernel_size=1),
            BasicConv2d(16, 24, kernel_size=3, padding=1),
            BasicConv2d(24, 32, kernel_size=3, padding=1)
        )
        self.conv = nn.Conv2d(64, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.conv(out)
        out = residual + self.scale * out
        out = self.relu(out)
        return out


class ResInceptionBTiny(nn.Module):
    def __init__(self, in_channels=576, scale=0.1):
        super().__init__()
        self.scale = scale
        self.branch1 = BasicConv2d(in_channels, 96, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 80, kernel_size=(1,7), padding=(0,3)),
            BasicConv2d(80, 96, kernel_size=(7,1), padding=(3,0))
        )
        self.conv = nn.Conv2d(192, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        out = self.conv(out)
        out = residual + self.scale * out
        out = self.relu(out)
        return out


class ResInceptionCTiny(nn.Module):
    def __init__(self, in_channels=1072, scale=0.1):
        super().__init__()
        self.scale = scale
        self.branch1 = BasicConv2d(in_channels, 96, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 96, kernel_size=1),
            BasicConv2d(96, 112, kernel_size=(1,3), padding=(0,1)),
            BasicConv2d(112, 128, kernel_size=(3,1), padding=(1,0))
        )
        self.conv = nn.Conv2d(224, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        out = self.conv(out)
        out = residual + self.scale * out
        out = self.relu(out)
        return out


 
class InceptionAStackTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(*[ResInceptionATiny(192) for _ in range(3)])

    def forward(self, x):
        return self.stack(x)


class ReductionATiny(nn.Module):
    def __init__(self, in_channels=192):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, 192, kernel_size=3, stride=2, padding=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 96, kernel_size=1),
            BasicConv2d(96, 112, kernel_size=3, padding=1),
            BasicConv2d(112, 192, kernel_size=3, stride=2, padding=1)
        )
        self.branch3 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class InceptionBStackTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[ResInceptionBTiny(576) for _ in range(5)])

    def forward(self, x):
        return self.blocks(x)


class ReductionBTiny(nn.Module):
    def __init__(self, in_channels=576):
        super().__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size=1),
            BasicConv2d(128, 192, kernel_size=3, stride=2, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size=1),
            BasicConv2d(128, 144, kernel_size=3, stride=2, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size=1),
            BasicConv2d(128, 144, kernel_size=3, padding=1),
            BasicConv2d(144, 160, kernel_size=3, stride=2, padding=1)
        )
        self.branch4 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class InceptionCStackTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[ResInceptionCTiny(1072) for _ in range(2)])

    def forward(self, x):
        return self.blocks(x)


 
class InceptionResNetV2Tiny(nn.Module):
    def __init__(self, num_classes=10):  
        super().__init__()
        self.stem = InceptionStemTiny()
        self.inceptionA = InceptionAStackTiny()
        self.reductionA = ReductionATiny(192)
        self.inceptionB = InceptionBStackTiny()
        self.reductionB = ReductionBTiny(576)
        self.inceptionC = InceptionCStackTiny()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1072, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inceptionA(x)
        x = self.reductionA(x)
        x = self.inceptionB(x)
        x = self.reductionB(x)
        x = self.inceptionC(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


 
if __name__ == "__main__":
    model = InceptionResNetV2Tiny(num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print("Out Shape:", out.shape)          # torch.Size([1, 10])
    print("Parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")  # ≈4.8M