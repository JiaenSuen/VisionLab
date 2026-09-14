"""Red-text localization for document/passport-card images.

The current task is intentionally color-driven: find red printed text anywhere in
an image.  HSV hue thresholding is therefore the default detector.  The earlier
Lab local-contrast detector is retained as an optional mode for ablation and for
very low-saturation red print.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]
NormalizedROI = Tuple[float, float, float, float]
Offset = Tuple[int, int]


def red_filter(
    image: np.ndarray,
    roi: NormalizedROI = (0.0, 0.0, 1.0, 1.0),
    *,
    mode: str = "hsv",
    blur_sigma: float = 3.0,
    min_saturation: int = 90,
    min_value: int = 40,
) -> tuple[np.ndarray, Offset]:
    """Return a binary mask of red candidates inside ``roi``.

    ``hsv`` is the default because the revised task is document-wide red-text
    detection rather than one fixed field.  ``lab`` reproduces the older local
    red-contrast method.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if mode not in {"hsv", "lab"}:
        raise ValueError("mode must be 'hsv' or 'lab'")

    x1n, y1n, x2n, y2n = roi
    if not (0.0 <= x1n < x2n <= 1.0 and 0.0 <= y1n < y2n <= 1.0):
        raise ValueError("ROI must be normalized within [0, 1].")

    h, w = image.shape[:2]
    x1, y1 = int(w * x1n), int(h * y1n)
    x2, y2 = int(w * x2n), int(h * y2n)
    crop = image[y1:y2, x1:x2]

    if mode == "hsv":
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        low_red = cv2.inRange(hsv, (0, min_saturation, min_value), (15, 255, 255))
        high_red = cv2.inRange(hsv, (165, min_saturation, min_value), (180, 255, 255))
        mask = cv2.bitwise_or(low_red, high_red)
    else:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        red_axis = lab[:, :, 1]
        local_baseline = cv2.GaussianBlur(red_axis, (0, 0), blur_sigma)
        red_contrast = cv2.subtract(red_axis, local_baseline)
        _, mask = cv2.threshold(
            red_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    return mask, (x1, y1)


def find_component_boxes(
    mask: np.ndarray,
    offset: Offset = (0, 0),
    min_area: int = 5,
    min_width: int = 2,
    min_height: int = 4,
    close_kernel: tuple[int, int] = (2, 2),
    max_area_ratio: float = 0.035,
) -> tuple[list[Box], np.ndarray]:
    """Convert the red mask into connected-component character proposals.

    Very large components are removed because document portraits, stamps, and
    large graphics can share a red hue but are not useful OCR glyph proposals.
    """
    if mask is None or mask.size == 0:
        raise ValueError("Mask is empty.")

    kernel = np.ones(close_kernel, np.uint8)
    detect_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(detect_mask, 8)

    mask_area = float(mask.shape[0] * mask.shape[1])
    max_area = mask_area * max_area_ratio
    ox, oy = offset
    boxes: list[Box] = []

    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < min_area or area > max_area:
            continue
        if width < min_width or height < min_height:
            continue
        boxes.append((int(x + ox), int(y + oy), int(width), int(height)))

    return boxes, detect_mask


def draw_boxes(image: np.ndarray, boxes: Iterable[Box], thickness: int = 1) -> np.ndarray:
    """Draw component proposals for debugging."""
    output = image.copy()
    for x, y, width, height in boxes:
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), thickness)
    return output
