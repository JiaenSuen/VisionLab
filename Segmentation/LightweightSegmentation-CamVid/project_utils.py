from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml

from losses import CombinedLoss, DiceLoss, SegmentationCrossEntropy


def deep_update(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    base_name = config.pop("_base_", None)
    if base_name:
        return deep_update(load_config(path.parent / base_name), config)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_logits(output):
    return output["out"] if isinstance(output, dict) else output


def build_loss(config: dict, ignore_index: int):
    name = config.get("name", "cross_entropy").lower()
    if name in {"ce", "cross_entropy"}:
        return SegmentationCrossEntropy(ignore_index)
    if name == "dice":
        return DiceLoss(ignore_index)
    if name == "combined":
        return CombinedLoss(
            ignore_index,
            ce_weight=float(config.get("ce_weight", 1.0)),
            dice_weight=float(config.get("dice_weight", 1.0)),
        )
    raise ValueError(f"Unknown loss: {name}")

