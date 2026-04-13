import torch
import torch.nn as nn
from torchvision import models

class ModelFactory:
    def __init__(self, name, num_classes=2, strategy="freeze"):
        self.name = name
        self.strategy = strategy
        self.num_classes = num_classes

    def build(self):
        if self.name == "vgg16":
            model = models.vgg16(weights="IMAGENET1K_V1")
            in_f = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_f, self.num_classes)

        elif self.name == "resnet18":
            model = models.resnet18(weights="IMAGENET1K_V1")
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, self.num_classes)

        elif self.name == "efficientnet":
            model = models.efficientnet_b0(weights="IMAGENET1K_V1")
            in_f = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_f, self.num_classes)

        elif self.name == "convnext":
            model = models.convnext_tiny(weights="IMAGENET1K_V1")
            in_f = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_f, self.num_classes)

        self.apply_strategy(model)
        return model

    def apply_strategy(self, model):
        if self.strategy == "freeze":
            for p in model.parameters():
                p.requires_grad = False
            self._unfreeze_head(model)

        elif self.strategy == "partial":
            for p in model.parameters():
                p.requires_grad = False
            self._unfreeze_head(model)
            self._unfreeze_last_block(model)

        elif self.strategy == "full":
            for p in model.parameters():
                p.requires_grad = True

    def _unfreeze_head(self, model):
        if hasattr(model, "fc"):
            for p in model.fc.parameters():
                p.requires_grad = True
        elif hasattr(model, "classifier"):
            for p in model.classifier.parameters():
                p.requires_grad = True

    def _unfreeze_last_block(self, model):
        if self.name == "resnet":
            for p in model.layer4.parameters():
                p.requires_grad = True
        elif self.name == "efficientnet":
            for p in model.features[-1].parameters():
                p.requires_grad = True
        elif self.name == "convnext":
            for p in model.features[-1].parameters():
                p.requires_grad = True
        elif self.name == "vgg16":
            for p in model.features[-5:].parameters():
                p.requires_grad = True