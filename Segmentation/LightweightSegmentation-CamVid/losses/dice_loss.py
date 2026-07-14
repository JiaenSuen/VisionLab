import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probabilities = logits.softmax(dim=1) * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        denominator = probabilities.sum(dims) + one_hot.sum(dims)
        present = one_hot.sum(dims) > 0
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        if not present.any():
            return logits.sum() * 0.0
        return 1.0 - dice[present].mean()

