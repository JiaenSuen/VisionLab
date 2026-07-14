from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_tensors(image: Image.Image, mask: Image.Image):
    image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
    mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy()).long()
    return image_tensor, mask_tensor


class TrainTransform:
    def __init__(self, image_size: tuple[int, int]) -> None:
        self.image_size = image_size
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __call__(self, image: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        image = self.color_jitter(image)
        image = TF.resize(image, self.image_size, InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, InterpolationMode.NEAREST)
        return _to_tensors(image, mask)


class EvalTransform:
    def __init__(self, image_size: tuple[int, int]) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, self.image_size, InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, InterpolationMode.NEAREST)
        return _to_tensors(image, mask)


class ImageOnlyTransform:
    def __init__(self, image_size: tuple[int, int]) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = TF.resize(image, self.image_size, InterpolationMode.BILINEAR)
        return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)

