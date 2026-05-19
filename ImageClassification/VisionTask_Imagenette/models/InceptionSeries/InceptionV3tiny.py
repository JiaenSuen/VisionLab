import torch
import torch.nn as nn


def build_inception3_tiny(num_classes):
    return InceptionV3tiny(num_classes)


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
class InceptionV3tiny(nn.Module):
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



if __name__ == "__main__":
    model = InceptionV3tiny(num_classes=1000)
    model.eval()            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dummy_input = torch.randn(1, 3, 299, 299).to(device) 
    with torch.no_grad():
        output = model(dummy_input)
        print(f"out shape: {output.shape}") 
        print(f"Output [5] :\n{output[0, :5]}")