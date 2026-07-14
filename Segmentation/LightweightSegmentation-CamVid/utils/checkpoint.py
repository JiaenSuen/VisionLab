from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch: int, metrics: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        },
        path,
    )


def load_model_checkpoint(path, model, device="cpu") -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint

