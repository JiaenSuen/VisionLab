import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet

def build_alexnet_pt(num_classes=1000):
    model = alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model