"""Regression check against the successful pre-modular RedCharacterDetect logic."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.red_components import find_component_boxes, red_filter  # noqa: E402


def original_reference(image: np.ndarray):
    h, w = image.shape[:2]
    x1, y1 = int(w * 0.63), int(h * 0.70)
    crop = image[y1:h, x1:w]

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    red_axis = lab[:, :, 1]
    blur = cv2.GaussianBlur(red_axis, (0, 0), 3)
    diff = cv2.subtract(red_axis, blur)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    detect_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(detect_mask, connectivity=8)

    boxes = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < 8:
            continue
        if height < 5 or width < 2:
            continue
        boxes.append((int(x + x1), int(y + y1), int(width), int(height)))
    return mask, detect_mask, boxes


def main() -> None:
    for image_path in (ROOT / "examples" / "1.jpg", ROOT / "examples" / "2.png"):
        image = cv2.imread(str(image_path))
        reference_mask, reference_detect, reference_boxes = original_reference(image)
        mask, offset = red_filter(image)
        boxes, detect_mask = find_component_boxes(mask, offset)

        assert np.array_equal(reference_mask, mask), f"Raw mask regression: {image_path.name}"
        assert np.array_equal(reference_detect, detect_mask), f"Detect-mask regression: {image_path.name}"
        assert reference_boxes == boxes, f"BBox regression: {image_path.name}"
        print(f"PASS {image_path.name}: mask, detect mask, and {len(boxes)} boxes match reference.")


if __name__ == "__main__":
    main()
