import torchvision
import torch.nn as nn
from hyperparams import num_classes 

def build_resnet18(num_classes=num_classes):
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Linear(512, 100),
        nn.ReLU(),
        nn.Linear(100, num_classes)
    )
    return model