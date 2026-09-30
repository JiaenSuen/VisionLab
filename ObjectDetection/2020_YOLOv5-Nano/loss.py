"""
Anchor-based YOLO loss used by this reproduction.

Prediction shape:
    (batch_size, 3, S, S, 5 + num_classes)

Target shape:
    (batch_size, 3, S, S, 6)

Target format:
    [objectness, x_cell, y_cell, width_cell, height_cell, class_label]

Prediction format:
    [objectness_logit, x_logit, y_logit, w_logit, h_logit, class_logits...]
"""

import torch
import torch.nn as nn

from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    Anchor-based YOLO loss for one prediction scale.
    """

    def __init__(self, num_classes=20):
        super().__init__()

        self.num_classes = num_classes

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()

        self.lambda_box = 10.0
        self.lambda_obj = 1.0
        self.lambda_noobj = 10.0
        self.lambda_class = 1.0

    def forward(self, predictions, target, anchors):
        """
        Calculate the anchor-based YOLO loss for one scale.

        Args:
            predictions: Tensor with shape (batch_size, 3, S, S, 5 + num_classes).
            target: Tensor with shape (batch_size, 3, S, S, 6).
            anchors: Tensor with shape (3, 2), already scaled to the current grid size.

        Returns:
            Scalar loss tensor.
        """

        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # No-object loss. Anchors marked as -1 are ignored.
        no_object_loss = self.bce(
            predictions[..., 0:1][noobj],
            target[..., 0:1][noobj],
        )

        if obj.sum() == 0:
            return self.lambda_noobj * no_object_loss

        anchors = anchors.reshape(1, 3, 1, 1, 2)

        # Decode predicted boxes in cell-space for IoU calculation.
        pred_xy = torch.sigmoid(predictions[..., 1:3])
        pred_wh = torch.exp(predictions[..., 3:5]) * anchors
        pred_boxes_decoded = torch.cat((pred_xy, pred_wh), dim=-1)

        ious = intersection_over_union(
            pred_boxes_decoded[obj],
            target[..., 1:5][obj],
            box_format="midpoint",
        ).detach()

        # Box loss uses the anchor-based parameterization adopted in this project:
        # x/y are trained after sigmoid; w/h are trained in log-space.
        target_xy = target[..., 1:3]
        target_wh = torch.log(1e-6 + target[..., 3:5] / anchors)
        target_boxes = torch.cat((target_xy, target_wh), dim=-1)

        pred_boxes_for_loss = torch.cat(
            (pred_xy, predictions[..., 3:5]),
            dim=-1,
        )

        box_loss = self.mse(
            pred_boxes_for_loss[obj],
            target_boxes[obj],
        )

        object_loss = self.bce(
            predictions[..., 0:1][obj],
            ious * target[..., 0:1][obj],
        )

        class_loss = self.entropy(
            predictions[..., 5:][obj],
            target[..., 5][obj].long(),
        )

        loss = (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

        return loss
