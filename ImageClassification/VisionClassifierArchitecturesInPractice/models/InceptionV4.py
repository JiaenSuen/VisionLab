import torch
import torch.nn as nn
 


def build_inception4(num_classes):
    return InceptionV4(num_classes)


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

        self.conv1 = BasicConv2d(3, 32, kernel_size=3, stride=2)  # 149
        self.conv2 = BasicConv2d(32, 32, kernel_size=3)           # 147
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, padding=1)

        self.maxpool1 = nn.MaxPool2d(3, stride=2)  # 73

        self.conv4 = BasicConv2d(64, 80, kernel_size=1)
        self.conv5 = BasicConv2d(80, 192, kernel_size=3)

        self.maxpool2 = nn.MaxPool2d(3, stride=2)  # 35

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool2(x)

        return x

# Inception A
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Branch 1
        self.branch1 = BasicConv2d(in_channels, 96, kernel_size=1)

        # Branch 2
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )

        # Branch 3
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )

        # Branch 4
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




class InceptionAStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            InceptionA(192),
            InceptionA(384),
            InceptionA(384),
            InceptionA(384),
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
            BasicConv2d(224, 256, kernel_size=3, stride=2)
        )

        # Branch 3
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        return torch.cat([b1, b2, b3], dim=1)

# Inception B 
class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, 384, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1,7), padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(7,1), padding=(3,0))
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(7,1), padding=(3,0)),
            BasicConv2d(224, 224, kernel_size=(1,7), padding=(0,3)),
            BasicConv2d(224, 224, kernel_size=(7,1), padding=(3,0)),
            BasicConv2d(224, 256, kernel_size=(1,7), padding=(0,3))
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
    

class InceptionBStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[
            InceptionB(1024) for _ in range(7)
        ])

    def forward(self, x):
        return self.blocks(x)



class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=(1,7), padding=(0,3)),
            BasicConv2d(256, 320, kernel_size=(7,1), padding=(3,0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x)
        ], dim=1)




# Inception C
class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, 256, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 384, kernel_size=1),
        )
        self.branch2a = BasicConv2d(384, 256, kernel_size=(1,3), padding=(0,1))
        self.branch2b = BasicConv2d(384, 256, kernel_size=(3,1), padding=(1,0))

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 384, kernel_size=1),
            BasicConv2d(384, 448, kernel_size=(3,1), padding=(1,0)),
            BasicConv2d(448, 512, kernel_size=(1,3), padding=(0,1)),
        )
        self.branch3a = BasicConv2d(512, 256, kernel_size=(1,3), padding=(0,1))
        self.branch3b = BasicConv2d(512, 256, kernel_size=(3,1), padding=(1,0))

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


class InceptionCStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[
            InceptionC(1536) for _ in range(3)
        ])

    def forward(self, x):
        return self.blocks(x)
    



class InceptionV4(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = InceptionStem()
        self.inceptionA = InceptionAStack()
        self.reductionA = ReductionA(384)
        self.inceptionB = InceptionBStack()
        self.reductionB = ReductionB(1024)
        self.inceptionC = InceptionCStack()

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.2)  # keep=0.8
        self.fc = nn.Linear(1536, num_classes)

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

    model = InceptionV4()
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
    redB     = ReductionB(1024)
    x = stackB(x) 
    print(f"Inception B : {x.shape}")
    x = redB(x)
    print(f"Reduction B : {x.shape}")


    stackC   = InceptionCStack()
    x = stackC(x) 
    print(f"Inception C : {x.shape}")


'''
    torch.Size([1, 1000])
    -------------------------
    torch.Size([1, 192, 35, 35])
    Inception A : torch.Size([1, 384, 35, 35])
    Reduction A : torch.Size([1, 1024, 17, 17])
    Inception B : torch.Size([1, 1024, 17, 17])
    Reduction B : torch.Size([1, 1536, 8, 8])
    Inception C : torch.Size([1, 1536, 8, 8])
'''