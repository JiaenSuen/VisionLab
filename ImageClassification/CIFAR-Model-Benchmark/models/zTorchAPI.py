from torchvision.models import googlenet
import torch.nn as nn
import torchvision


def build_googlenet_pt(num_classes=1000):
    model = googlenet(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet18pt(num_classes=10):
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Linear(512, 100),
        nn.ReLU(),
        nn.Linear(100, num_classes)
    )
    return model

def build_resnet34pt(num_classes=10):
    model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Linear(512, 100),
        nn.ReLU(),
        nn.Linear(100, num_classes)
    )
    return model