from torch import nn
from torch.nn import functional as F


VGG_types = {
    "VGG11": [64, "M",
              128, "M",
              256, 256, "M",
              512, 512, "M",
              512, 512, "M"],

    "VGG13": [64, 64, "M",
              128, 128, "M",
              256, 256, "M",
              512, 512, "M",
              512, 512, "M"],

    "VGG16": [64, 64, "M",
              128, 128, "M",
              256, 256, 256, "M",
              512, 512, 512, "M",
              512, 512, 512, "M"],

    "VGG19": [64, 64, "M",
              128, 128, "M",
              256, 256, 256, 256, "M",
              512, 512, 512, 512, "M",
              512, 512, 512, 512, "M"],
}


class VGG_Net(nn.Module):
    def __init__(self,vgg_arch):
        super(VGG_Net,self).__init__()
        self.conv_layer = self.vgg_block(vgg_arch)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Sequential(
            nn.Linear(512*7 *7 , 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096 , 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024 , 10 )
        )

    def vgg_block(self,vgg_arch):
        layers = []
        in_channels = 3

        for v in vgg_arch:
            if v == 'M':
                layers.append(nn.MaxPool2d(2,2))
            else:
                layers.append(nn.Conv2d(in_channels= in_channels , out_channels= v , kernel_size= 3 ,padding= 1 ))
                layers.append(nn.ReLU())
                in_channels = v
        return nn.Sequential(*layers)
    def forward(self , x):
        x = self.conv_layer(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear_layer(x)
        return x
    

