from __future__ import annotations

import torch


class SegmentationConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().reshape(-1).to("cpu")
        target = target.detach().reshape(-1).to("cpu")
        valid = (
            (target != self.ignore_index)
            & (target >= 0)
            & (target < self.num_classes)
        )
        indices = self.num_classes * target[valid] + prediction[valid]
        self.matrix += torch.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        matrix = self.matrix.double()
        true_positive = matrix.diag()
        target_count = matrix.sum(dim=1)
        prediction_count = matrix.sum(dim=0)
        union = target_count + prediction_count - true_positive

        class_iou = torch.where(union > 0, true_positive / union, torch.nan)
        class_accuracy = torch.where(
            target_count > 0, true_positive / target_count, torch.nan
        )
        total = matrix.sum()
        pixel_accuracy = true_positive.sum() / total if total > 0 else torch.tensor(0.0)
        return {
            "pixel_accuracy": float(pixel_accuracy),
            "mean_accuracy": float(torch.nanmean(class_accuracy)),
            "mean_iou": float(torch.nanmean(class_iou)),
            "class_iou": class_iou.tolist(),
        }

    def reset(self) -> None:
        self.matrix.zero_()

