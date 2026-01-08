import torchvision
import torch.nn as nn


def build_resnet18(num_classes=10):
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Linear(512, 100),
        nn.ReLU(),
        nn.Linear(100, num_classes)
    )
    return model

def build_resnet34(num_classes=10):
    model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Linear(512, 100),
        nn.ReLU(),
        nn.Linear(100, num_classes)
    )
    return model