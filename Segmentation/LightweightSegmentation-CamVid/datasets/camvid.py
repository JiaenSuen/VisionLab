from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


CAMVID_CLASSES = (
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
)

# CamVid 常見 11-class palette。黑色 Unlabelled 不屬於訓練類別。
CAMVID_PALETTE = np.asarray(
    [
        [128, 128, 128],
        [128, 0, 0],
        [192, 192, 128],
        [128, 64, 128],
        [0, 0, 192],
        [128, 128, 0],
        [192, 128, 128],
        [64, 64, 128],
        [64, 0, 128],
        [64, 64, 0],
        [0, 128, 192],
    ],
    dtype=np.uint8,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def rgb_mask_to_class_ids(mask: Image.Image, ignore_index: int = 255) -> np.ndarray:
    """Convert a CamVid RGB palette mask or grayscale ID mask to HxW class IDs."""
    # Palette PNG stores color-table indices in a 2-D array; convert it to RGB
    # before matching colors. A true L-mode mask is treated as class IDs.
    if mask.mode == "P":
        mask = mask.convert("RGB")
    array = np.asarray(mask)
    if array.ndim == 2:
        result = array.astype(np.int64, copy=True)
        invalid = (result < 0) | (result >= len(CAMVID_CLASSES))
        result[invalid] = ignore_index
        return result

    rgb = array[..., :3]
    result = np.full(rgb.shape[:2], ignore_index, dtype=np.int64)
    for class_id, color in enumerate(CAMVID_PALETTE):
        result[np.all(rgb == color, axis=-1)] = class_id
    return result


class CamVidDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform=None,
        ignore_index: int = 255,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val or test; got {split!r}")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.ignore_index = ignore_index
        self.image_dir = self.root / split
        self.mask_dir = self.root / f"{split}_labels"

        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected directories {self.image_dir} and {self.mask_dir}. "
                "See README.md for the CamVid layout."
            )

        self.samples: list[tuple[Path, Path]] = []
        for image_path in sorted(self.image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = self.mask_dir / image_path.name
            if not mask_path.exists():
                # CamVid releases commonly append `_L` to label filenames,
                # e.g. 0001TP_009210.png -> 0001TP_009210_L.png.
                patterns = (
                    f"{image_path.stem}_L.*",
                    f"{image_path.stem}_label.*",
                    f"{image_path.stem}.*",
                )
                candidates = sorted(
                    {
                        candidate
                        for pattern in patterns
                        for candidate in self.mask_dir.glob(pattern)
                        if candidate.is_file()
                        and candidate.suffix.lower() in IMAGE_EXTENSIONS
                    }
                )
                if len(candidates) != 1:
                    candidate_names = [candidate.name for candidate in candidates]
                    raise FileNotFoundError(
                        f"Cannot uniquely match mask for {image_path.name}; "
                        f"candidates={candidate_names}"
                    )
                mask_path = candidates[0]
            self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        raw_mask = Image.open(mask_path)
        mask = Image.fromarray(
            rgb_mask_to_class_ids(raw_mask, self.ignore_index).astype(np.uint8),
            mode="L",
        )

        if image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch for {image_path.name}: {image.size} vs {mask.size}"
            )

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {"image": image, "mask": mask, "filename": image_path.name}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a CamVid split")
    parser.add_argument("--root", default="data/CamVid")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    args = parser.parse_args()

    dataset = CamVidDataset(args.root, args.split)
    _, mask_path = dataset.samples[0]
    ids = np.unique(rgb_mask_to_class_ids(Image.open(mask_path)))
    print(f"split={args.split}, samples={len(dataset)}")
    print(f"first_mask={mask_path}")
    print(f"unique_ids={ids.tolist()}")
    invalid = [int(x) for x in ids if x != 255 and not 0 <= x < len(CAMVID_CLASSES)]
    if invalid:
        raise SystemExit(f"Invalid class IDs: {invalid}")


if __name__ == "__main__":
    main()
