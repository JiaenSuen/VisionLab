from __future__ import annotations

from torch import nn
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)


def build_lraspp_mobilenet_v3_large(num_classes: int, pretrained: bool = True):
    """Build LR-ASPP and replace its VOC classifier with a CamVid classifier."""
    if pretrained:
        model = lraspp_mobilenet_v3_large(
            weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        )
    else:
        model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )

    if model.classifier.low_classifier.out_channels != num_classes:
        low_in = model.classifier.low_classifier.in_channels
        high_in = model.classifier.high_classifier.in_channels
        model.classifier.low_classifier = nn.Conv2d(low_in, num_classes, 1)
        model.classifier.high_classifier = nn.Conv2d(high_in, num_classes, 1)
    return model
