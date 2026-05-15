import torch
import torch.nn as nn

# API
def build_inception3(num_classes):
    return InceptionV3(num_classes)



# Inception-v3 Network Architecture 
# Inception-v3 TinyNet Architecture with Tiny Version Modules

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

        # Branch 1 
        self.branch1 = BasicConv2d(in_channels, 320, kernel_size=1)

        # Branch 2: 1×1 →  (1×3 + 3×1) ， 768
        self.branch2_1x1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch2_1x3 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch2_3x1 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # Branch 3: 
        self.branch3_1x1 = BasicConv2d(in_channels, 384, kernel_size=1) 
        self.branch3_3x3 = BasicConv2d(384, 384, kernel_size=3, padding=1)     
        self.branch3_1x3 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3_3x1 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # Branch 4: pool + 1×1
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        # Branch 1
        b1 = self.branch1(x)

        # Branch 2
        tmp2 = self.branch2_1x1(x)
        b2 = torch.cat([self.branch2_1x3(tmp2), self.branch2_3x1(tmp2)], dim=1)

        # Branch 3
        tmp3 = self.branch3_1x1(x)
        b3 = torch.cat([self.branch3_1x3(tmp3), self.branch3_3x1(tmp3)], dim=1)

        # Branch pool
        b4 = self.branch_pool(x)

 
        return torch.cat([b1, b2, b3, b4], dim=1)




# Inception-v3 Network Architecture 
class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Stem: 299×299×3 → 35×35×192
        self.stem = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2),
            BasicConv2d(64, 80, kernel_size=1),
            BasicConv2d(80, 192, kernel_size=3),
            nn.MaxPool2d(3, stride=2),
        )

        # 35×35×192 → 35×35×288
        self.inception_a = nn.Sequential(
            InceptionA(192, pool_features=32),
            InceptionA(256, pool_features=64),
            InceptionA(288, pool_features=64),
            InceptionA(288, pool_features=64),
            InceptionA(288, pool_features=64),
        )

        # 35×35×288 → 17×17×768
        self.reduction_a = ReductionA(288)

        # 17×17×768 → 17×17×768 
        self.inception_b = nn.Sequential(
            InceptionB(768),
            InceptionB(768),
            InceptionB(768),
            InceptionB(768),
        )

        # 17×17×768 → 8×8×1280
        self.reduction_b = ReductionB(768)

        # 8×8×1280 → 8×8×2048
        self.inception_c = nn.Sequential(
            InceptionC(1280),
            InceptionC(2048),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)  
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
        x = self.fc(x)
        return x



 




 

if __name__ == "__main__":
    model = InceptionV3(num_classes=1000)
    model.eval()            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dummy_input = torch.randn(1, 3, 299, 299).to(device) 
    with torch.no_grad():
        output = model(dummy_input)
        print(f"out shape: {output.shape}") 
        print(f"Output [5] :\n{output[0, :5]}")