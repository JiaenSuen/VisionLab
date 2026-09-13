"""Smoke test for the current Tesseract end-to-end baseline."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.red_components import find_component_boxes, red_filter  # noqa: E402
from modules.sequence_groups import expand_boxes_for_ocr, group_nearby_boxes  # noqa: E402
from modules.tesseract_ocr import recognize_grouped_text  # noqa: E402

EXPECTED = {
    "1.jpg": "A234567890",
    "2.png": "BTS4ever J-hope",
}


def run_one(path: Path) -> str:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)

    mask, offset = red_filter(image)
    boxes, _ = find_component_boxes(mask, offset=offset)
    groups = group_nearby_boxes(boxes, max_gap=8, adaptive_gap=True, gap_height_ratio=1.0)
    crop_boxes = expand_boxes_for_ocr(groups, image.shape, mode="adaptive")
    results = recognize_grouped_text(
        image,
        groups,
        crop_boxes,
        preprocess_mode="gray",
        scale=4.0,
        min_group_aspect_ratio=3.0,
    )
    return " | ".join(result.text for result in results if result.text)


def main() -> None:
    for name, expected in EXPECTED.items():
        prediction = run_one(ROOT / "examples" / name)
        if prediction != expected:
            raise AssertionError(f"{name}: expected {expected!r}, got {prediction!r}")
        print(f"PASS {name}: {prediction}")


if __name__ == "__main__":
    main()
