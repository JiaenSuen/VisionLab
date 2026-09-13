"""Red-text localization using the same behavior as the original prototype."""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, width, height
NormalizedROI = Tuple[float, float, float, float]  # x1, y1, x2, y2
Offset = Tuple[int, int]  # x, y


def red_filter(
    image: np.ndarray,
    roi: NormalizedROI = (0.63, 0.70, 1.0, 1.0),
    blur_sigma: float = 3.0,
) -> tuple[np.ndarray, Offset]:
    """Extract a local-red binary mask from an ROI.

    Important: the ROI is cropped *before* Lab conversion and Otsu thresholding.
    This intentionally preserves the behavior of the original
    ``RedCharacterDetect`` prototype.

    Returns:
        mask: ROI-sized binary mask before morphology.
        offset: (x, y) location of the ROI in the original image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    x1n, y1n, x2n, y2n = roi
    if not (0.0 <= x1n < x2n <= 1.0 and 0.0 <= y1n < y2n <= 1.0):
        raise ValueError("ROI must be normalized as (x1, y1, x2, y2) within [0, 1].")

    image_h, image_w = image.shape[:2]
    x1 = int(image_w * x1n)
    y1 = int(image_h * y1n)
    x2 = int(image_w * x2n)
    y2 = int(image_h * y2n)

    crop = image[y1:y2, x1:x2]

    # Lab a-channel is sensitive to the green-red chromatic axis.
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    red_axis = lab[:, :, 1]

    # Highlight pixels that are locally redder than their surroundings.
    local_baseline = cv2.GaussianBlur(red_axis, (0, 0), blur_sigma)
    red_contrast = cv2.subtract(red_axis, local_baseline)

    # Adaptive threshold estimated only from the target ROI.
    _, mask = cv2.threshold(
        red_contrast,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return mask, (x1, y1)


def find_component_boxes(
    mask: np.ndarray,
    offset: Offset = (0, 0),
    min_area: int = 8,
    min_width: int = 2,
    min_height: int = 5,
    close_kernel: tuple[int, int] = (2, 2),
) -> tuple[list[Box], np.ndarray]:
    """Find connected-component boxes using the original prototype settings.

    A light 2x2 morphological closing is applied only to the detection mask.
    The returned boxes are translated back to original-image coordinates.

    Returns:
        boxes: component boxes in original-image coordinates.
        detect_mask: the ROI-sized mask after morphological closing.
    """
    if mask is None or mask.size == 0:
        raise ValueError("Mask is empty.")

    kernel = np.ones(close_kernel, np.uint8)
    detect_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        detect_mask,
        connectivity=8,
    )

    ox, oy = offset
    boxes: list[Box] = []

    for label in range(1, count):  # Label 0 is background.
        x, y, width, height, area = stats[label]

        if area < min_area:
            continue
        if height < min_height or width < min_width:
            continue

        boxes.append((
            int(x + ox),
            int(y + oy),
            int(width),
            int(height),
        ))

    return boxes, detect_mask


def draw_boxes(
    image: np.ndarray,
    boxes: Iterable[Box],
    thickness: int = 1,
) -> np.ndarray:
    """Draw component boxes for visualization only."""
    output = image.copy()
    for x, y, width, height in boxes:
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            thickness,
        )
    return output
