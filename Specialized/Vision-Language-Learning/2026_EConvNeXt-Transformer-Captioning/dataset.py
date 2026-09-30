import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    """Lowercase word-tokenizer designed for Flickr-style English image captions."""
    return TOKEN_PATTERN.findall(text.lower().strip())


class Vocabulary:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, min_freq: int = 2, max_size: int = 12000):
        self.min_freq = int(min_freq)
        self.max_size = int(max_size)
        self.itos: List[str] = [self.PAD, self.BOS, self.EOS, self.UNK]
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    @property
    def pad_idx(self) -> int:
        return self.stoi[self.PAD]

    @property
    def bos_idx(self) -> int:
        return self.stoi[self.BOS]

    @property
    def eos_idx(self) -> int:
        return self.stoi[self.EOS]

    @property
    def unk_idx(self) -> int:
        return self.stoi[self.UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def build(self, captions: Iterable[str]) -> "Vocabulary":
        counter = Counter()
        for caption in captions:
            counter.update(tokenize(caption))

        words = [w for w, c in counter.items() if c >= self.min_freq]
        words.sort(key=lambda w: (-counter[w], w))
        if self.max_size is not None and self.max_size > 0:
            words = words[: max(0, self.max_size - len(self.itos))]

        self.itos = [self.PAD, self.BOS, self.EOS, self.UNK] + words
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        return self

    def encode(self, caption: str, max_len: int) -> List[int]:
        # Full sequence: <bos> tokens... <eos>, cropped to max_len.
        tokens = tokenize(caption)
        token_ids = [self.stoi.get(tok, self.unk_idx) for tok in tokens]
        max_content = max(0, max_len - 2)
        token_ids = token_ids[:max_content]
        return [self.bos_idx] + token_ids + [self.eos_idx]

    def decode(self, ids: Sequence[int], remove_special: bool = True) -> str:
        words: List[str] = []
        for idx in ids:
            idx = int(idx)
            if idx < 0 or idx >= len(self.itos):
                continue
            tok = self.itos[idx]
            if remove_special and tok in {self.PAD, self.BOS, self.EOS}:
                if tok == self.EOS:
                    break
                continue
            if tok == self.UNK:
                continue
            words.append(tok)
        return " ".join(words)

    def to_dict(self) -> Dict:
        return {"min_freq": self.min_freq, "max_size": self.max_size, "itos": self.itos}

    @classmethod
    def from_dict(cls, obj: Dict) -> "Vocabulary":
        vocab = cls(min_freq=obj.get("min_freq", 2), max_size=obj.get("max_size", 12000))
        vocab.itos = list(obj["itos"])
        vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
        return vocab

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _strip_image_index(image_id: str) -> str:
    image_id = image_id.strip().strip('"')
    if "#" in image_id:
        image_id = image_id.split("#", 1)[0]
    return os.path.basename(image_id)


def parse_caption_file(captions_file: str) -> List[Dict[str, str]]:
    """Parse common Flickr8k and Flickr30k caption annotation formats.

    Supported formats include:
    - CSV: ``image,caption`` or ``filename,caption``
    - Flickr token text: ``image.jpg#0<TAB>caption``
    - Flickr30k Kaggle-style pipe-delimited files:
      ``image_name|comment_number|comment``
    - Headerless comma/tab/pipe text where the first field is the image name
      and the last textual field is the caption.
    """
    if not os.path.isfile(captions_file):
        raise FileNotFoundError(f"Caption file not found: {captions_file}")

    with open(captions_file, "r", encoding="utf-8-sig", errors="replace") as f:
        raw_lines = [line.rstrip("\n\r") for line in f if line.strip()]
    if not raw_lines:
        raise ValueError(f"No captions found in: {captions_file}")

    first = raw_lines[0]
    records: List[Dict[str, str]] = []

    # Detect a structured header and use csv.DictReader so quoted commas are safe.
    delimiter = None
    for candidate in ("|", ",", "\t"):
        if candidate in first:
            fields = [part.strip().lower() for part in first.split(candidate)]
            has_image = any(name in fields for name in ("image", "image_name", "filename", "file_name"))
            has_caption = any(name in fields for name in ("caption", "comment", "description", "sentence"))
            if has_image and has_caption:
                delimiter = candidate
                break

    if delimiter is not None:
        with open(captions_file, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError("Caption file header is missing.")
            field_map = {name.lower().strip(): name for name in reader.fieldnames if name is not None}
            image_key = next(
                (field_map[k] for k in ("image", "image_name", "filename", "file_name") if k in field_map),
                reader.fieldnames[0],
            )
            caption_key = next(
                (field_map[k] for k in ("caption", "comment", "description", "sentence") if k in field_map),
                reader.fieldnames[-1],
            )
            for row in reader:
                image_id = _strip_image_index(row.get(image_key, "") or "")
                caption = (row.get(caption_key, "") or "").strip().strip('"')
                if image_id and caption:
                    records.append({"image": image_id, "caption": caption})
    else:
        # Headerless or legacy token files.
        for line in raw_lines:
            if "\t" in line:
                parts = line.split("\t")
            elif "|" in line:
                parts = line.split("|")
            elif "," in line:
                parts = line.split(",", 1)
            else:
                continue

            image_id = _strip_image_index(parts[0])
            if len(parts) >= 3 and parts[1].strip().isdigit():
                caption = "|".join(parts[2:]).strip().strip('"')
            else:
                caption = parts[-1].strip().strip('"') if len(parts) > 1 else ""
            if image_id and caption and image_id.lower() not in {"image", "image_name", "filename", "file_name"}:
                records.append({"image": image_id, "caption": caption})

    if not records:
        raise ValueError(
            "Could not parse captions. Use an image/caption CSV, Flickr token text, "
            "or Flickr30k pipe-delimited annotation file."
        )
    return records

def group_references(records: Sequence[Dict[str, str]]) -> Dict[str, List[str]]:
    refs: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        refs[r["image"]].append(r["caption"])
    return dict(refs)


def split_by_image(
    records: Sequence[Dict[str, str]],
    train_ratio: float = 0.80,
    valid_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, List[str]]:
    total_ratio = train_ratio + valid_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train_ratio + valid_ratio + test_ratio must be 1.0")

    images = sorted({r["image"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(images)

    n = len(images)
    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))
    train_images = images[:n_train]
    valid_images = images[n_train : n_train + n_valid]
    test_images = images[n_train + n_valid :]

    return {"train": train_images, "valid": valid_images, "test": test_images}


def records_for_split(records: Sequence[Dict[str, str]], image_names: Sequence[str]) -> List[Dict[str, str]]:
    image_set = set(image_names)
    return [r for r in records if r["image"] in image_set]


def save_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ImagePathResolver:
    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        self.lookup = {name.lower(): name for name in os.listdir(image_dir)}

    def __call__(self, image_name: str) -> str:
        direct = os.path.join(self.image_dir, image_name)
        if os.path.isfile(direct):
            return direct
        alt = self.lookup.get(os.path.basename(image_name).lower())
        if alt is not None:
            return os.path.join(self.image_dir, alt)
        raise FileNotFoundError(f"Image not found under {self.image_dir}: {image_name}")


def _resize_short_side(img: Image.Image, short_side: int) -> Image.Image:
    w, h = img.size
    if min(w, h) == short_side:
        return img
    if w < h:
        new_w = short_side
        new_h = int(round(h * short_side / w))
    else:
        new_h = short_side
        new_w = int(round(w * short_side / h))
    return img.resize((new_w, new_h), Image.BICUBIC)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    return img.crop((left, top, left + size, top + size))


def _random_resized_crop(
    img: Image.Image,
    size: int,
    scale: Tuple[float, float] = (0.72, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> Image.Image:
    w, h = img.size
    area = w * h
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    for _ in range(10):
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect = math.exp(random.uniform(log_ratio[0], log_ratio[1]))
        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_w <= w and 0 < crop_h <= h:
            left = random.randint(0, w - crop_w)
            top = random.randint(0, h - crop_h)
            img = img.crop((left, top, left + crop_w, top + crop_h))
            return img.resize((size, size), Image.BICUBIC)

    img = _resize_short_side(img, size)
    img = _center_crop(img, size)
    return img.resize((size, size), Image.BICUBIC)


def _color_jitter(img: Image.Image, strength: float = 0.12) -> Image.Image:
    # Conservative jitter for captioning: avoid damaging object semantics.
    ops = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]
    random.shuffle(ops)
    for op in ops:
        factor = 1.0 + random.uniform(-strength, strength)
        img = op(img).enhance(factor)
    return img


def _to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


class CaptionTrainTransform:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        img = _random_resized_crop(img, self.image_size)
        if random.random() < 0.50:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.80:
            img = _color_jitter(img, strength=0.12)
        return _to_normalized_tensor(img)


class CaptionEvalTransform:
    def __init__(self, image_size: int = 224, resize_short_side: int = 256):
        self.image_size = image_size
        self.resize_short_side = resize_short_side

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        img = _resize_short_side(img, self.resize_short_side)
        img = _center_crop(img, self.image_size)
        return _to_normalized_tensor(img)


class FlickrCaptionDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, str]],
        image_dir: str,
        vocab: Vocabulary,
        max_len: int = 34,
        transform=None,
    ):
        self.records = list(records)
        self.resolver = ImagePathResolver(image_dir)
        self.vocab = vocab
        self.max_len = int(max_len)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]
        image = Image.open(self.resolver(r["image"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        ids = self.vocab.encode(r["caption"], self.max_len)
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        return {
            "image": image,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "image_name": r["image"],
            "caption": r["caption"],
        }


class FlickrEvalImageDataset(Dataset):
    """One item per image, used for generation metrics with multiple references."""

    def __init__(self, image_names: Sequence[str], image_dir: str, transform=None):
        self.image_names = list(image_names)
        self.resolver = ImagePathResolver(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict:
        image_name = self.image_names[idx]
        image = Image.open(self.resolver(image_name)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "image_name": image_name}


class CaptionCollateFn:
    """Pickle-safe collate function for Windows DataLoader workers.

    Windows uses multiprocessing spawn, so nested functions cannot be pickled.
    Keeping the collate callable as a top-level class allows num_workers > 0.
    """

    def __init__(self, pad_idx: int):
        self.pad_idx = int(pad_idx)

    def __call__(self, batch: Sequence[Dict]) -> Dict:
        images = torch.stack([item["image"] for item in batch], dim=0)
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_idx,
        )
        target_ids = pad_sequence(
            [item["target_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_idx,
        )
        return {
            "images": images,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "image_names": [item["image_name"] for item in batch],
            "captions": [item["caption"] for item in batch],
        }


def make_caption_collate_fn(pad_idx: int):
    return CaptionCollateFn(pad_idx)


def eval_image_collate(batch: Sequence[Dict]) -> Dict:
    images = torch.stack([item["image"] for item in batch], dim=0)
    return {"images": images, "image_names": [item["image_name"] for item in batch]}
