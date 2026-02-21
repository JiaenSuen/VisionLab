import torch
from torch import nn
from torch.nn import functional as F


param = [[3, 64, 11, 0, 4], [64, 192, 5, 2, 1], [192, 2, 3, 1, 1]]  # in_channels, out_channels, kernel_size, padding, stride

class NiN(nn.Module):
    def __init__(self, param, num_classes):
        super(NiN, self).__init__()
        self.features1 = self.nin_block(param[0])
        self.features2 = self.nin_block(param[1])
        self.features3 = self.nin_block(param[2])

    def nin_block(self, a_param):
        layers = []

        layers.append(nn.Conv2d(in_channels=a_param[0], out_channels=a_param[1], kernel_size=a_param[2], padding=a_param[3], stride=a_param[4]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=a_param[1], out_channels=a_param[1], kernel_size=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=a_param[1], out_channels=a_param[1], kernel_size=1, padding=1))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)       

    def forward(self, x):
        x = self.features1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.features2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.features3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)) # Global Average Pooling ( Implement by adaptive average pooling )
        x = torch.flatten(x, start_dim=1)

        return x