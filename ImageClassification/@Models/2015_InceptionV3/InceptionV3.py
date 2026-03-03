import torch
import torch.nn as nn

# API
def build_inception3(num_classes):
    return InceptionV3(num_classes)

def build_inception3_tiny(num_classes):
    return InceptionV3Tiny(num_classes)

# Inception-v3 Network Architecture [192 → 256 → 288 → 768 → 1280 → 2048]
# Inception-v3 TinyNet Architecture [64 → 96 → 128 → 256 → 384 → 512] with Tiny Version Modules

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
    

# Inception A Module : ( 35 × 35 )
class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1)
        )

        self.branch3x3dbl = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1),
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_features, kernel_size=1)
        )

    def forward(self, x):
        outputs = [
            self.branch1x1(x),
            self.branch5x5(x),
            self.branch3x3dbl(x),
            self.branch_pool(x),
        ]
        return torch.cat(outputs, 1)
    
# Efficient Grid Size Reduction Module ( 35 -> 17 )
class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=2),
        )

        self.branch_pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch3x3(x),
            self.branch3x3dbl(x),
            self.branch_pool(x)
        ], 1)
    


# Inception B Module : ( 17 × 17 ) { Spatial Factorization into Asymmetric Convolutions }
class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size=1),
            BasicConv2d(128, 128, kernel_size=(1,7), padding=(0,3)),
            BasicConv2d(128, 192, kernel_size=(7,1), padding=(3,0))
        )

        self.branch7x7dbl = nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size=1),
            BasicConv2d(128, 128, kernel_size=(7,1), padding=(3,0)),
            BasicConv2d(128, 128, kernel_size=(1,7), padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), padding=(3,0)),
            BasicConv2d(128, 192, kernel_size=(1,7), padding=(0,3)),
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1x1(x),
            self.branch7x7(x),
            self.branch7x7dbl(x),
            self.branch_pool(x)
        ], 1)
    

# Efficient Grid Size Reduction Module B ( 17 -> 8 ) { Spatial Factorization into Asymmetric Convolutions }
class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 320, kernel_size=3, stride=2)
        )

        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1,7), padding=(0,3)),
            BasicConv2d(192, 192, kernel_size=(7,1), padding=(3,0)),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branch_pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch3x3(x),
            self.branch7x7(x),
            self.branch_pool(x)
        ], 1)
    

# Inception C Module : ( 8 × 8 ) { Spatial Factorization into Asymmetric Convolutions }
class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, 384, kernel_size=1),
        )

        self.branch3x3_1 = BasicConv2d(384, 384, kernel_size=(1,3), padding=(0,1))
        self.branch3x3_2 = BasicConv2d(384, 384, kernel_size=(3,1), padding=(1,0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3 = torch.cat([
            self.branch3x3_1(branch3x3),
            self.branch3x3_2(branch3x3)
        ], 1)

        return torch.cat([
            self.branch1x1(x),
            branch3x3,
            self.branch_pool(x)
        ], 1)
    





# Inception-v3 Network Architecture [192 → 256 → 288 → 768 → 1280 → 2048]
class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2),
            BasicConv2d(64, 80, kernel_size=1),
            BasicConv2d(80, 192, kernel_size=3),
            nn.MaxPool2d(3, stride=2),
        )

        self.inception_a = nn.Sequential(
            InceptionA(192, 32),
            InceptionA(256, 64),
            InceptionA(288, 64),
        )

        self.reduction_a = ReductionA(288)

        self.inception_b = nn.Sequential(
            InceptionB(768),
            InceptionB(768),
            InceptionB(768),
            InceptionB(768),
        )

        self.reduction_b = ReductionB(768)

        self.inception_c = nn.Sequential(
            InceptionC(1280),
            InceptionC(2048),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
    












# Inception V3 Tiny Version

class TinyStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.stem(x)

class TinyInceptionA(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.b1 = BasicConv2d(in_ch, 32, 1)

        self.b2 = nn.Sequential(
            BasicConv2d(in_ch, 32, 1),
            BasicConv2d(32, 48, 3, 1, 1)
        )

        self.b3 = nn.Sequential(
            BasicConv2d(in_ch, 32, 1),
            BasicConv2d(32, 48, 3, 1, 1),
            BasicConv2d(48, 48, 3, 1, 1)
        )

        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BasicConv2d(in_ch, 16, 1)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x),
                          self.b3(x), self.b4(x)], 1)
    
class TinyReductionA(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.b1 = BasicConv2d(in_ch, 96, 3, 2, 1)

        self.b2 = nn.Sequential(
            BasicConv2d(in_ch, 64, 1),
            BasicConv2d(64, 96, 3, 2, 1)
        )

        self.b3 = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], 1)
    
class TinyInceptionB(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.b1 = BasicConv2d(in_ch, 64, 1)

        self.b2 = nn.Sequential(
            BasicConv2d(in_ch, 64, 1),
            BasicConv2d(64, 64, (1,3), 1, (0,1)),
            BasicConv2d(64, 96, (3,1), 1, (1,0)),
        )

        self.b3 = nn.Sequential(
            BasicConv2d(in_ch, 64, 1),
            BasicConv2d(64, 64, (3,1), 1, (1,0)),
            BasicConv2d(64, 64, (1,3), 1, (0,1)),
            BasicConv2d(64, 96, (3,1), 1, (1,0)),
        )

        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BasicConv2d(in_ch, 32, 1)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x),
                          self.b3(x), self.b4(x)], 1)
    

class TinyInceptionC(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.b1 = BasicConv2d(in_ch, 128, 1)

        self.b2_1 = BasicConv2d(in_ch, 128, 1)
        self.b2_2a = BasicConv2d(128, 128, (1,3), 1, (0,1))
        self.b2_2b = BasicConv2d(128, 128, (3,1), 1, (1,0))

        self.b3 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BasicConv2d(in_ch, 64, 1)
        )

    def forward(self, x):
        b2 = self.b2_1(x)
        b2 = torch.cat([self.b2_2a(b2), self.b2_2b(b2)], 1)

        return torch.cat([self.b1(x), b2, self.b3(x)], 1)

# Inception-v3 Network Architecture [64 → 96 → 128 → 256 → 384 → 512] with Tiny Version Modules
class InceptionV3Tiny(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = TinyStem()

        self.a1 = TinyInceptionA(64)
        self.a2 = TinyInceptionA(144)

        self.redA = TinyReductionA(144)

        self.b1 = TinyInceptionB(336)
        self.b2 = TinyInceptionB(288)

        self.redB = TinyReductionA(288)

        self.c1 = TinyInceptionC(480)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(448, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.a1(x)
        x = self.a2(x)

        x = self.redA(x)

        x = self.b1(x)
        x = self.b2(x)

        x = self.redB(x)

        x = self.c1(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)