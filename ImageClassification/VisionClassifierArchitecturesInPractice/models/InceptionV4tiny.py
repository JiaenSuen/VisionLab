# Designed specifically for 32×32 input (CIFAR compatible)
# • Overall branch count, branching method, and thesis philosophy are fully preserved.
# • Only kernel size (asymmetric convolution from 7 to 5) and stride/padding are adjusted to make the receptive field suitable for small images.
# • Stem is completely non-aggressive downsampling → maintains 32×32×384.
# • Reduction-A stride=3 → 10×10 (approximately 12×12).
# • Reduction-B stride=2 → 4×4 (falling between 4 and 6).
# • All Inception layers have pooling stride=1 and 3×3, with pad=1 and dimensions unchanged.

import torch
import torch.nn as nn


def build_inception4_tiny(num_classes):
    return InceptionV4Tiny(num_classes)

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
 



class TinyInceptionStem(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = BasicConv2d(3, 32, kernel_size=3, stride=1, padding=1)   # 32
        self.conv2 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)   # 32
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)   # 32

        # 32 → 32
        self.branch1_conv = BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.branch1_pool = nn.MaxPool2d(3, stride=1, padding=1)

        # 32 → 32 （5×1 / 1×5）
        self.branch2 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(64, 96, kernel_size=3, padding=1)       
        )

        self.branch2_2 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )

        # 32 → 32
        self.branch3_conv = BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.branch3_pool = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 32 → 32 (160 ch)
        x = torch.cat([
            self.branch1_conv(x),
            self.branch1_pool(x)
        ], dim=1)

        # 32 → 32 (192 ch)
        x = torch.cat([
            self.branch2(x),
            self.branch2_2(x)
        ], dim=1)

        # 32 → 32 (384 ch)
        x = torch.cat([
            self.branch3_conv(x),
            self.branch3_pool(x)
        ], dim=1)

        return x







class TinyInceptionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, 96, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 96, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class TinyInceptionAStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(*[TinyInceptionA(384) for _ in range(4)])

    def forward(self, x):
        return self.stack(x)


# Reduction-A（stride=3 let 32->10,above 12）
class TinyReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = BasicConv2d(
            in_channels, 384, kernel_size=3, stride=3
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=3, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=3)
        )

        self.branch3 = nn.MaxPool2d(3, stride=3)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)
    



class TinyInceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, 384, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(224, 256, kernel_size=(5, 1), padding=(2, 0))
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(224, 224, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(224, 224, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 5), padding=(0, 2))
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 128, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


class TinyInceptionBStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[TinyInceptionB(1024) for _ in range(7)])

    def forward(self, x):
        return self.blocks(x)


# Reduction-B（stride=2 let 10->4）
class TinyReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(256, 320, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x)
        ], dim=1)
    

class TinyInceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, 256, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 384, kernel_size=1),
        )
        self.branch2a = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch2b = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 384, kernel_size=1),
            BasicConv2d(384, 448, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(448, 512, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.branch3a = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3b = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 256, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)

        b2 = self.branch2(x)
        b2 = torch.cat([self.branch2a(b2), self.branch2b(b2)], dim=1)

        b3 = self.branch3(x)
        b3 = torch.cat([self.branch3a(b3), self.branch3b(b3)], dim=1)

        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)


class TinyInceptionCStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[TinyInceptionC(1536) for _ in range(3)])

    def forward(self, x):
        return self.blocks(x)


class InceptionV4Tiny(nn.Module):
    def __init__(self, num_classes=100):          
        super().__init__() 
        self.stem = TinyInceptionStem()
        self.inceptionA = TinyInceptionAStack()
        self.reductionA = TinyReductionA(384)
        self.inceptionB = TinyInceptionBStack()
        self.reductionB = TinyReductionB(1024)
        self.inceptionC = TinyInceptionCStack()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stem(x)           # 32×32×384
        x = self.inceptionA(x)     # 32×32×384
        x = self.reductionA(x)     # 10×10×1024
        x = self.inceptionB(x)     # 10×10×1024
        x = self.reductionB(x)     # 4×4×1536
        x = self.inceptionC(x)     # 4×4×1536

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

 
if __name__ == "__main__":
    model = InceptionV4Tiny(num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print("Output shape:", out.shape)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
 
    ## Module Test
    print("-"*25)
    x = torch.randn(1, 3, 32, 32)

    stem     = TinyInceptionStem() 
    stackA   = TinyInceptionAStack()
    redA     = TinyReductionA(384)



    x = stem(x)     
    print(x.shape) 
    x = stackA(x)    
    print(f"Inception A : {x.shape}") 
    x = redA(x)
    print(f"Reduction A : {x.shape}")


    stackB   = TinyInceptionBStack()
    redB     = TinyReductionB(1024)
    x = stackB(x) 
    print(f"Inception B : {x.shape}")
    x = redB(x)
    print(f"Reduction B : {x.shape}")


    stackC   = TinyInceptionCStack()
    x = stackC(x) 
    print(f"Inception C : {x.shape}")