"""
dataset.py

Raw Pascal VOC/YOLO-format dataset. CSV must have two columns:
  img,label
where `img` is the image filename under --img-dir and `label` is the txt label
filename under --label-dir. Each label row is YOLO normalized xywh:
  class x_center y_center width height

The dataset returns:
  image: Tensor[3,H,W]
  boxes: Tensor[N,5] with [class, x, y, w, h], normalized to [0,1]
"""

from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as T


VOC_CLASSES: Dict[int, str] = {
    0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
    5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow",
    10: "diningtable", 11: "dog", 12: "horse", 13: "motorbike", 14: "person",
    15: "pottedplant", 16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor",
}


class Compose:
    def __init__(self, transforms: Sequence[object]):
        self.transforms = list(transforms)

    def set_epoch(self, epoch: int) -> None:
        for t in self.transforms:
            if hasattr(t, "set_epoch"):
                t.set_epoch(epoch)

    def __call__(self, img: Image.Image, boxes: List[List[float]]):
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes


class EpochAware:
    def __init__(self):
        self.epoch = 1
        self.total_epochs = 100
        self.close_last_epochs = 15

    def set_schedule(self, total_epochs: int, close_last_epochs: int) -> None:
        self.total_epochs = int(total_epochs)
        self.close_last_epochs = int(close_last_epochs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def strong_aug_active(self) -> bool:
        return self.epoch <= max(1, self.total_epochs - self.close_last_epochs)


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: Image.Image, boxes: List[List[float]]):
        return img.resize(self.size, Image.BILINEAR), boxes


class RandomScaleTranslate(EpochAware):
    """Square-canvas scale/translation using normalized xywh boxes."""

    def __init__(self, p: float = 0.70, scale_range: Tuple[float, float] = (0.80, 1.20), translate: float = 0.08, min_box: float = 0.003):
        super().__init__()
        self.p = float(p)
        self.scale_range = scale_range
        self.translate = float(translate)
        self.min_box = float(min_box)

    def __call__(self, img: Image.Image, boxes: List[List[float]]):
        if (not self.strong_aug_active) or random.random() > self.p or len(boxes) == 0:
            return img, boxes
        size = img.size[0]
        scale = random.uniform(*self.scale_range)
        new_size = max(2, int(round(size * scale)))
        img_resized = img.resize((new_size, new_size), Image.BILINEAR)
        max_shift = int(round(self.translate * size))

        if new_size <= size:
            room = size - new_size
            center = room // 2
            dx = random.randint(max(0, center - max_shift), min(room, center + max_shift))
            dy = random.randint(max(0, center - max_shift), min(room, center + max_shift))
            canvas = Image.new("RGB", (size, size), (114, 114, 114))
            canvas.paste(img_resized, (dx, dy))
            offset_x, offset_y = dx, dy
        else:
            crop = new_size - size
            center = crop // 2
            crop_x = random.randint(max(0, center - max_shift), min(crop, center + max_shift))
            crop_y = random.randint(max(0, center - max_shift), min(crop, center + max_shift))
            canvas = img_resized.crop((crop_x, crop_y, crop_x + size, crop_y + size))
            offset_x, offset_y = -crop_x, -crop_y

        new_boxes: List[List[float]] = []
        for box in boxes:
            cls, x, y, w, h = [float(v) for v in box]
            x1 = (x - w / 2) * new_size + offset_x
            y1 = (y - h / 2) * new_size + offset_y
            x2 = (x + w / 2) * new_size + offset_x
            y2 = (y + h / 2) * new_size + offset_y
            x1 = min(max(x1, 0.0), size)
            y1 = min(max(y1, 0.0), size)
            x2 = min(max(x2, 0.0), size)
            y2 = min(max(y2, 0.0), size)
            bw = (x2 - x1) / size
            bh = (y2 - y1) / size
            if bw <= self.min_box or bh <= self.min_box:
                continue
            nx = ((x1 + x2) * 0.5) / size
            ny = ((y1 + y2) * 0.5) / size
            new_boxes.append([int(cls), nx, ny, bw, bh])
        return canvas, new_boxes


class ColorJitter(EpochAware):
    def __init__(self, brightness: float = 0.18, contrast: float = 0.18, saturation: float = 0.18, hue: float = 0.03):
        super().__init__()
        self.strong = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.weak = T.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.06, hue=0.01)

    def __call__(self, img: Image.Image, boxes: List[List[float]]):
        return (self.strong if self.strong_aug_active else self.weak)(img), boxes


class RandomBlur(EpochAware):
    def __init__(self, p: float = 0.04):
        super().__init__()
        self.p = float(p)

    def __call__(self, img: Image.Image, boxes: List[List[float]]):
        if self.strong_aug_active and random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.9))), boxes
        return img, boxes


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: Image.Image, boxes: List[List[float]]):
        if random.random() >= self.p:
            return img, boxes
        img = ImageOps.mirror(img)
        flipped = []
        for box in boxes:
            cls, x, y, w, h = box
            flipped.append([int(cls), 1.0 - float(x), float(y), float(w), float(h)])
        return img, flipped


class ToTensor:
    def __init__(self):
        self.t = T.ToTensor()

    def __call__(self, img: Image.Image, boxes: List[List[float]]):
        return self.t(img), boxes


class Normalize:
    """ImageNet normalization for E-ConvNeXt transfer."""

    def __init__(self):
        self.n = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img: torch.Tensor, boxes: List[List[float]]):
        return self.n(img), boxes


class RawVOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file: str, img_dir: str, label_dir: str, transform=None):
        self.annotations = pd.read_csv(csv_file)
        if len(self.annotations.columns) < 2:
            raise ValueError(f"CSV must have at least two columns: {csv_file}")
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):
        image_filename = str(self.annotations.iloc[index, 0]).strip()
        label_filename = str(self.annotations.iloc[index, 1]).strip()
        img_path = os.path.join(self.img_dir, image_filename)
        label_path = os.path.join(self.label_dir, label_filename)
        image = Image.open(img_path).convert("RGB")
        boxes: List[List[float]] = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:5])
                    x = min(max(x, 0.0), 0.999999)
                    y = min(max(y, 0.0), 0.999999)
                    w = min(max(w, 1e-6), 1.0)
                    h = min(max(h, 1e-6), 1.0)
                    boxes.append([cls, x, y, w, h])
        if self.transform is not None:
            image, boxes = self.transform(image, boxes)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5), dtype=torch.float32)
        return image, boxes_tensor


def raw_voc_collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def set_transform_schedule(transform, total_epochs: int, close_last_epochs: int) -> None:
    if not hasattr(transform, "transforms"):
        return
    for t in transform.transforms:
        if hasattr(t, "set_schedule"):
            t.set_schedule(total_epochs, close_last_epochs)


def build_transforms(image_size: int, total_epochs: int, close_last_epochs: int):
    train_transform = Compose([
        Resize((image_size, image_size)),
        RandomScaleTranslate(p=0.70, scale_range=(0.80, 1.20), translate=0.08),
        ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.03),
        RandomBlur(p=0.04),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(),
    ])
    set_transform_schedule(train_transform, total_epochs, close_last_epochs)
    eval_transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(),
    ])
    return train_transform, eval_transform


def read_annotation_csv(csv_path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return rows
        # If a header is present, keep natural order. If first row looks like data, include it.
        if len(header) >= 2 and header[0].strip().lower() not in {"img", "image", "filename"}:
            rows.append((header[0].strip(), header[1].strip()))
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                rows.append((row[0].strip(), row[1].strip()))
    return rows


def scan_dataset(csv_path: str, label_dir: str, split_name: str, num_classes: int = 20) -> Dict[str, object]:
    rows = read_annotation_csv(csv_path)
    per_class = {i: 0 for i in range(num_classes)}
    total = found = missing = empty = 0
    for _, label_name in rows:
        path = os.path.join(label_dir, label_name)
        if not os.path.exists(path):
            missing += 1
            continue
        found += 1
        file_boxes = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 5:
                    continue
                cls = int(float(p[0]))
                if 0 <= cls < num_classes:
                    per_class[cls] += 1
                    total += 1
                    file_boxes += 1
        if file_boxes == 0:
            empty += 1
    return {
        "split": split_name,
        "csv_path": csv_path,
        "image_count": len(rows),
        "label_files_found": found,
        "missing_label_files": missing,
        "empty_label_files": empty,
        "total_boxes": total,
        "per_class_counts": per_class,
    }
