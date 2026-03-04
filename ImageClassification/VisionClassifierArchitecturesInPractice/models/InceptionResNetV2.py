import torch
import torch.nn as nn
 

def build_inception_resnet_v2(num_classes=1000):
    return InceptionResNetV2(num_classes=num_classes)


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

class InceptionStem(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = BasicConv2d(3, 32, kernel_size=3, stride=2)   # 149
        self.conv2 = BasicConv2d(32, 32, kernel_size=3)            # 147
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, padding=1) # 147

        # 147 → 73
        self.branch1_conv = BasicConv2d(64, 96, kernel_size=3, stride=2)
        self.branch1_pool = nn.MaxPool2d(3, stride=2)

        # 73 → 71
        self.branch2 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(7,1), padding=(3,0)),
            BasicConv2d(64, 64, kernel_size=(1,7), padding=(0,3)),
            BasicConv2d(64, 96, kernel_size=3)
        )

        self.branch2_2 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3)
        )

        # 71 → 35
        self.branch3_conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.branch3_pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 147 → 73
        x = torch.cat([
            self.branch1_conv(x),
            self.branch1_pool(x)
        ], dim=1)

        # 73 → 71
        x = torch.cat([
            self.branch2(x),
            self.branch2_2(x)
        ], dim=1)

        # 71 → 35
        x = torch.cat([
            self.branch3_conv(x),
            self.branch3_pool(x)
        ], dim=1)

        return x

# Inception A
class ResInceptionA(nn.Module):
    def __init__(self, in_channels=384, scale=0.1):
        super().__init__()
        self.scale = scale

        # Branch 1
        self.branch1 = BasicConv2d(in_channels, 32, kernel_size=1)

        # Branch 2
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        )

        # Branch 3
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1)
        )

        # concat channel = 32+32+64 = 128
        self.conv = nn.Conv2d(128, in_channels, kernel_size=1)

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



class InceptionAStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            *[ResInceptionA(384) for _ in range(5)]
        )
    def forward(self, x):
        x = self.stack(x)
        return x
    
class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Branch 1
        self.branch1 = BasicConv2d(
            in_channels, 384,
            kernel_size=3, stride=2
        )

        # Branch 2
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=3, padding=1),
            BasicConv2d(224, 384, kernel_size=3, stride=2)
        )

        # Branch 3
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        return torch.cat([b1, b2, b3], dim=1)

# Inception B 
class ResInceptionB(nn.Module):
    def __init__(self, in_channels=1152, scale=0.1):
        super().__init__()
        self.scale = scale

        self.branch1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size=1),
            BasicConv2d(128, 160, kernel_size=(1,7), padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), padding=(3,0))
        )

        # concat = 192+192 = 384
        self.conv = nn.Conv2d(384, in_channels, kernel_size=1)
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
    

class InceptionBStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResInceptionB(1152) for _ in range(10)
        ])

    def forward(self, x):
        return self.blocks(x)



class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Branch 1
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        # Branch 2
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        # Branch 3
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=1),
            BasicConv2d(256, 288, kernel_size=3, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        # Branch 4
        self.branch4 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)




# Inception C
class ResInceptionC(nn.Module):
    def __init__(self, in_channels=2144, scale=0.1):
        super().__init__()
        self.scale = scale

        self.branch1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1,3), padding=(0,1)),
            BasicConv2d(224, 256, kernel_size=(3,1), padding=(1,0))
        )

        self.conv = nn.Conv2d(192+256, in_channels, kernel_size=1)
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


class InceptionCStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResInceptionC(2144) for _ in range(5)
        ])

    def forward(self, x):
        return self.blocks(x)
    



class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = InceptionStem()
        self.inceptionA = InceptionAStack()
        self.reductionA = ReductionA(384)
        self.inceptionB = InceptionBStack()
        self.reductionB = ReductionB(1152)
        self.inceptionC = InceptionCStack()

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.2)  # keep=0.8
        self.fc = nn.Linear(2144, num_classes)

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

    model = InceptionResNetV2()
    x = torch.randn(1, 3, 299, 299)
    out = model(x)

    print(out.shape)

    ## Module Test
    print("-"*25)
    x = torch.randn(1, 3, 299, 299)

    stem     = InceptionStem() 
    stackA   = InceptionAStack()
    redA     = ReductionA(384)



    x = stem(x)     
    print(x.shape) 
    x = stackA(x)    
    print(f"Inception A : {x.shape}") 
    x = redA(x)
    print(f"Reduction A : {x.shape}")


    stackB   = InceptionBStack()
    redB     = ReductionB(1152)
    x = stackB(x) 
    print(f"Inception B : {x.shape}")
    x = redB(x)
    print(f"Reduction B : {x.shape}")


    stackC   = InceptionCStack()
    x = stackC(x) 
    print(f"Inception C : {x.shape}")

 