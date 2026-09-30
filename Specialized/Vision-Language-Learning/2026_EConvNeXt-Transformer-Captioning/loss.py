import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Token-level label smoothing CE with ignore_index support."""

    def __init__(self, smoothing: float = 0.10, ignore_index: int = 0):
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in [0, 1).")
        self.smoothing = float(smoothing)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, T, V], target: [B, T]
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        target = target.reshape(-1)
        valid = target.ne(self.ignore_index)
        if valid.sum() == 0:
            return logits.new_tensor(0.0)

        logits = logits[valid]
        target = target[valid]
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()


def token_accuracy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 0) -> float:
    pred = logits.argmax(dim=-1)
    mask = target.ne(ignore_index)
    denom = mask.sum().item()
    if denom == 0:
        return 0.0
    correct = pred.eq(target).logical_and(mask).sum().item()
    return float(correct / denom)


def perplexity_from_loss(loss_value: float) -> float:
    loss_value = float(loss_value)
    if loss_value > 20:
        return float("inf")
    return float(math.exp(loss_value))


def loss_dict(loss_value: float, acc_value: float) -> Dict[str, float]:
    return {"loss": float(loss_value), "ppl": perplexity_from_loss(loss_value), "token_acc": float(acc_value)}
