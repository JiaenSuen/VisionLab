import torch
import torch.nn as nn


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

        self.conv1 = BasicConv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(32, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x  # 32x32x64
    
class Conv1x1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = BasicConv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Inception A : The channel is reduced by half, but the 4-branch structure is maintained.
class InceptionATiny(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, 32, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 32, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
    

class ReductionATiny(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = BasicConv2d(
            in_channels, 128,
            kernel_size=3, stride=2
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 96, kernel_size=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 128, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x)
        ], dim=1)
    
#Inception B : Retain the 1x7 / 7x1 factorization concept, but change it to 1x3 / 3x1.
class InceptionBTiny(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(1,3), padding=(0,1)),
            BasicConv2d(64, 64, kernel_size=(3,1), padding=(1,0))
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(3,1), padding=(1,0)),
            BasicConv2d(64, 64, kernel_size=(1,3), padding=(0,1)),
            BasicConv2d(64, 64, kernel_size=(3,1), padding=(1,0)),
            BasicConv2d(64, 64, kernel_size=(1,3), padding=(0,1))
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 64, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
    

class ReductionBTiny(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size=1),
            BasicConv2d(128, 128, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size=1),
            BasicConv2d(128, 128, kernel_size=(1,3), padding=(0,1)),
            BasicConv2d(128, 128, kernel_size=(3,1), padding=(1,0)),
            BasicConv2d(128, 128, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x)
        ], dim=1)
    



class InceptionCTiny(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, 128, kernel_size=1)

        self.branch2 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.branch2a = BasicConv2d(128, 128, kernel_size=(1,3), padding=(0,1))
        self.branch2b = BasicConv2d(128, 128, kernel_size=(3,1), padding=(1,0))

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size=1),
            BasicConv2d(128, 128, kernel_size=(3,1), padding=(1,0)),
            BasicConv2d(128, 128, kernel_size=(1,3), padding=(0,1))
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 128, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b2 = torch.cat([self.branch2a(b2), self.branch2b(b2)], dim=1)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)
    






class InceptionV4Tiny(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = InceptionStemTiny()

        self.inceptionA = nn.Sequential(
            InceptionATiny(64),
            Conv1x1(160, 128),
            InceptionATiny(128),
            Conv1x1(160, 128)
        )

        self.reductionA = ReductionATiny(128)

        self.inceptionB = nn.Sequential(
            InceptionBTiny(384),
            InceptionBTiny(256),
            InceptionBTiny(256)
        )

        self.reductionB = ReductionBTiny(256)

        self.inceptionC = nn.Sequential(
            InceptionCTiny(512),
            InceptionCTiny(640)
        )



        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(640, num_classes)

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

    model = InceptionV4Tiny()
    x = torch.randn(1, 3, 32, 32)
    out = model(x)

    print(out.shape)
