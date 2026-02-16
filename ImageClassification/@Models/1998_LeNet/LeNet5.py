import torch.nn as nn
from torchvision.transforms import transforms
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)    # C1
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)   # C3
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0) # C5
        self.fc1   = nn.Linear(120, 84)  # F6
        self.fc2   = nn.Linear(84, 10)   # Output layer
        self.tanh  = nn.Tanh()
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.tanh(self.conv1(x)) # C1 : Convolution + Activation
        x = self.tanh(self.pool(x))  # S2 : Subsampling + Activation
        x = self.tanh(self.conv2(x)) # C3 : Convolution + Activation
        x = self.tanh(self.pool(x))  # S4 : Subsampling + Activation
        x = self.tanh(self.conv3(x)) # C5 : Convolution
        x = x.view(x.size(0), -1)  
        x = self.tanh(self.fc1(x))   # F6 : MLP Linear Fully Connect
        x = self.fc2(x)              # Output layer
        return x


