from torch import nn

from .cross_entropy import SegmentationCrossEntropy
from .dice_loss import DiceLoss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce = SegmentationCrossEntropy(ignore_index)
        self.dice = DiceLoss(ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(
            logits, target
        )

