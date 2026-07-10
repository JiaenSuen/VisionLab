# ResNeXt
'''
The paper proposes a "simplified version of split-transform-merge": 
the input is split into low-dimensional embeddings, each undergoes the same topological transformation,
and finally summed (not concat). This allows for an unlimited increase in branches (cardinality) without manual tuning.
This design has three equivalent forms (Fig. 3 below), and the simplest implementation is achieved using grouped convolution.

For CIFAR : 29-layer ResNeXt(bottleneck template) , Increasing the value (C) is more effective than increasing the width; 16×64d achieves 3.58% CIFAR-10 accuracy.

'''


import torch.nn as nn

# ResNeXt Block : grouped convolution
class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, cardinality, group_width, stride=1):
        super().__init__()
        mid_planes = cardinality * group_width                  # C × d
        
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_planes)
        
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3,
                               stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_planes)
        
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_planes)
        
        self.relu  = nn.ReLU(inplace=True)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
    


class ResNeXt50(nn.Module):
    def __init__(self, num_classes=1000, cardinality=32, group_width_base=4):
        super().__init__()
        self.cardinality = cardinality
        self.group_width_base = group_width_base
        
        # Stem (conv1 of Table 1 in the paper)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 4 stage (group_width doubles with stage, consistent with Table 1 in the paper)
        self.layer1 = self._make_layer(64,  256, num_blocks=3,  stride=1, group_width=group_width_base)
        self.layer2 = self._make_layer(256, 512, num_blocks=4,  stride=2, group_width=group_width_base*2)
        self.layer3 = self._make_layer(512,1024, num_blocks=6,  stride=2, group_width=group_width_base*4)
        self.layer4 = self._make_layer(1024,2048,num_blocks=3,  stride=2, group_width=group_width_base*8)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride, group_width):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNeXtBottleneck(in_planes, out_planes, self.cardinality, group_width, s))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x















class ResNeXtBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, cardinality=8):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        
        # Grouped conv (ResNeXt Core)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out



class ResNeXt18_Tiny(nn.Module):
    def __init__(self, num_classes=10, cardinality=8):
        super().__init__()
        self.in_planes = 64
        
        # CIFAR stem
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet18 structure: 2-2-2-2
        self.layer1 = self._make_layer(64,  2, stride=1, cardinality=cardinality)
        self.layer2 = self._make_layer(128, 2, stride=2, cardinality=cardinality)
        self.layer3 = self._make_layer(256, 2, stride=2, cardinality=cardinality)
        self.layer4 = self._make_layer(512, 2, stride=2, cardinality=cardinality)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_planes, blocks, stride, cardinality):
        layers = []
        layers.append(ResNeXtBasicBlock(self.in_planes, out_planes, stride, cardinality))
        self.in_planes = out_planes
        
        for _ in range(1, blocks):
            layers.append(ResNeXtBasicBlock(self.in_planes, out_planes, 1, cardinality))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
 

def resnext50_32x4d(num_classes=1000):
    """224x224 """
    return ResNeXt50(num_classes=num_classes, cardinality=32, group_width_base=4)


def resnext18_tiny(num_classes=10):
    return ResNeXt18_Tiny(num_classes=num_classes, cardinality=8)

if __name__ == "__main__":
    model_224 = resnext50_32x4d(num_classes=1000)
    model_32  = resnext18_tiny(num_classes=10)
    
    print("224x224 model params:", sum(p.numel() for p in model_224.parameters()) / 1e6, "M")
    print("32x32  model params:", sum(p.numel() for p in model_32.parameters()) / 1e6, "M")
    
 
    import torch
    x224 = torch.randn(2, 3, 224, 224)
    x32  = torch.randn(2, 3, 32, 32)
    print("224 output shape:", model_224(x224).shape)
    print("32  output shape:", model_32(x32).shape)