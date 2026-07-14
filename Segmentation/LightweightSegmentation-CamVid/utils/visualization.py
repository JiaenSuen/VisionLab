from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from datasets.camvid import CAMVID_PALETTE


def colorize_mask(mask, ignore_index: int = 255) -> Image.Image:
    mask = np.asarray(mask)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(CAMVID_PALETTE):
        rgb[mask == class_id] = color
    rgb[mask == ignore_index] = (0, 0, 0)
    return Image.fromarray(rgb)


def save_prediction(mask, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colorize_mask(mask).save(path)


def save_comparison(
    image: Image.Image,
    target,
    prediction,
    path: str | Path,
) -> None:
    target_image = colorize_mask(target).resize(image.size, Image.Resampling.NEAREST)
    prediction_image = colorize_mask(prediction).resize(
        image.size, Image.Resampling.NEAREST
    )
    canvas = Image.new("RGB", (image.width * 3, image.height))
    canvas.paste(image.convert("RGB"), (0, 0))
    canvas.paste(target_image, (image.width, 0))
    canvas.paste(prediction_image, (image.width * 2, 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)

