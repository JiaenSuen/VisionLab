"""Group nearby character boxes into sequence-OCR-ready text-line crops."""

from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, width, height


def _vertical_overlap_ratio(a: Box, b: Box) -> float:
    ay1, ay2 = a[1], a[1] + a[3]
    by1, by2 = b[1], b[1] + b[3]
    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    return overlap / max(1, min(a[3], b[3]))


def _same_text_line(a: Box, b: Box, min_vertical_overlap: float) -> bool:
    """Check whether two components are likely to belong to the same text line."""
    overlap_ok = _vertical_overlap_ratio(a, b) >= min_vertical_overlap

    a_center = a[1] + a[3] / 2.0
    b_center = b[1] + b[3] / 2.0
    center_tolerance = 0.55 * max(a[3], b[3])
    center_ok = abs(a_center - b_center) <= center_tolerance

    return overlap_ok or center_ok


def _horizontal_gap(a: Box, b: Box) -> int:
    """Return the free horizontal distance between two boxes; overlap gives zero."""
    a_left, a_right = a[0], a[0] + a[2]
    b_left, b_right = b[0], b[0] + b[2]

    if a_right < b_left:
        return b_left - a_right
    if b_right < a_left:
        return a_left - b_right
    return 0


def estimate_grouping_gap(
    boxes: Sequence[Box],
    minimum_gap: int = 8,
    gap_height_ratio: float = 1.0,
) -> int:
    """Estimate a scale-aware horizontal grouping distance.

    A fixed pixel threshold is brittle when document images are captured at
    different resolutions. Character height provides a simple scale proxy.
    The returned value is never smaller than ``minimum_gap`` so the behavior
    remains compatible with the earlier hand-tuned setting on small images.
    """
    if not boxes:
        return minimum_gap
    median_height = float(median(box[3] for box in boxes))
    return max(minimum_gap, int(round(median_height * gap_height_ratio)))


def group_nearby_boxes(
    boxes: Sequence[Box],
    max_gap: int = 8,
    min_vertical_overlap: float = 0.25,
    adaptive_gap: bool = False,
    gap_height_ratio: float = 1.0,
) -> list[Box]:
    """Transitively merge horizontally close boxes into text-sequence boxes.

    Boxes form a graph. Two boxes are connected when they are on approximately
    the same text line and their horizontal gap is no larger than the effective
    gap threshold. Connected graph components are then merged. With
    ``adaptive_gap=True``, the threshold is at least ``max_gap`` and may increase
    according to the median detected character height.
    """
    if not boxes:
        return []

    effective_gap = (
        estimate_grouping_gap(boxes, max_gap, gap_height_ratio)
        if adaptive_gap
        else max_gap
    )

    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if not _same_text_line(boxes[i], boxes[j], min_vertical_overlap):
                continue
            if _horizontal_gap(boxes[i], boxes[j]) <= effective_gap:
                union(i, j)

    members: dict[int, list[Box]] = {}
    for i, box in enumerate(boxes):
        members.setdefault(find(i), []).append(box)

    groups: list[Box] = []
    for group in members.values():
        x1 = min(box[0] for box in group)
        y1 = min(box[1] for box in group)
        x2 = max(box[0] + box[2] for box in group)
        y2 = max(box[1] + box[3] for box in group)
        groups.append((x1, y1, x2 - x1, y2 - y1))

    return sorted(groups, key=lambda box: (box[1], box[0]))


def expand_box(
    box: Box,
    image_shape: tuple[int, ...],
    padding: int = 4,
) -> Box:
    """Expand a box by a fixed number of pixels inside image boundaries."""
    x, y, width, height = box
    image_h, image_w = image_shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_w, x + width + padding)
    y2 = min(image_h, y + height + padding)

    return x1, y1, x2 - x1, y2 - y1


def expand_box_adaptive(
    box: Box,
    image_shape: tuple[int, ...],
    min_padding: int = 4,
    horizontal_height_ratio: float = 0.35,
    vertical_height_ratio: float = 0.20,
) -> Box:
    """Expand an OCR crop using font-scale-aware padding.

    The detector is intentionally kept tight for localization. OCR crops are a
    different representation and should include weak anti-aliased stroke edges
    and a small amount of surrounding context. Padding is therefore derived from
    detected text height rather than from the absolute image resolution.
    """
    x, y, width, height = box
    image_h, image_w = image_shape[:2]

    pad_x = max(min_padding, int(round(height * horizontal_height_ratio)))
    pad_y = max(min_padding, int(round(height * vertical_height_ratio)))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_w, x + width + pad_x)
    y2 = min(image_h, y + height + pad_y)
    return x1, y1, x2 - x1, y2 - y1


def expand_boxes_for_ocr(
    boxes: Sequence[Box],
    image_shape: tuple[int, ...],
    mode: str = "adaptive",
    padding: int = 4,
    min_padding: int = 4,
    horizontal_height_ratio: float = 0.35,
    vertical_height_ratio: float = 0.20,
) -> list[Box]:
    """Create OCR crop boxes without changing detector geometry."""
    if mode not in {"fixed", "adaptive"}:
        raise ValueError("mode must be 'fixed' or 'adaptive'.")

    expanded: list[Box] = []
    for box in boxes:
        if mode == "fixed":
            expanded.append(expand_box(box, image_shape, padding))
        else:
            expanded.append(
                expand_box_adaptive(
                    box,
                    image_shape,
                    min_padding=min_padding,
                    horizontal_height_ratio=horizontal_height_ratio,
                    vertical_height_ratio=vertical_height_ratio,
                )
            )
    return expanded


def resize_for_sequence_ocr(
    image: np.ndarray,
    target_height: int = 32,
) -> np.ndarray:
    """Resize a text crop to a fixed height while preserving its aspect ratio."""
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot resize an empty crop.")

    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def export_group_crops(
    image: np.ndarray,
    group_boxes: Sequence[Box],
    output_dir: str | Path,
    padding: int = 4,
    target_height: int | None = 32,
    padding_mode: str = "adaptive",
    min_padding: int = 4,
    horizontal_height_ratio: float = 0.35,
    vertical_height_ratio: float = 0.20,
) -> list[Path]:
    """Save grouped text crops that can be passed to sequence OCR tools.

    Detection boxes are never altered. Only the exported recognition crop is
    expanded. This separation prevents attempts to capture faint character edges
    from destabilizing the segmentation stage.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    crop_boxes = expand_boxes_for_ocr(
        group_boxes,
        image.shape,
        mode=padding_mode,
        padding=padding,
        min_padding=min_padding,
        horizontal_height_ratio=horizontal_height_ratio,
        vertical_height_ratio=vertical_height_ratio,
    )

    saved: list[Path] = []
    for index, (x, y, width, height) in enumerate(crop_boxes, start=1):
        crop = image[y : y + height, x : x + width]

        if target_height is not None:
            crop = resize_for_sequence_ocr(crop, target_height)

        path = output_path / f"text_group_{index:03d}.png"
        cv2.imwrite(str(path), crop)
        saved.append(path)

    return saved


def draw_group_boxes(
    image: np.ndarray,
    group_boxes: Iterable[Box],
    thickness: int = 2,
) -> np.ndarray:
    """Draw merged text-sequence boxes for debugging."""
    output = image.copy()
    for x, y, width, height in group_boxes:
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (255, 0, 255),
            thickness,
        )
    return output


def _horizontal_overlap_ratio(a: Box, b: Box) -> float:
    ax1, ax2 = a[0], a[0] + a[2]
    bx1, bx2 = b[0], b[0] + b[2]
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    return overlap / max(1, min(a[2], b[2]))


def _vertical_gap(a: Box, b: Box) -> int:
    a_top, a_bottom = a[1], a[1] + a[3]
    b_top, b_bottom = b[1], b[1] + b[3]
    if a_bottom < b_top:
        return b_top - a_bottom
    if b_bottom < a_top:
        return a_top - b_bottom
    return 0


def _merge_box_set(items: Sequence[Box]) -> Box:
    x1 = min(b[0] for b in items)
    y1 = min(b[1] for b in items)
    x2 = max(b[0] + b[2] for b in items)
    y2 = max(b[1] + b[3] for b in items)
    return (x1, y1, x2 - x1, y2 - y1)


def group_red_text_regions(
    boxes: Sequence[Box],
    *,
    horizontal_gap: int = 12,
    vertical_gap: int = 12,
    min_alignment: float = 0.25,
    min_vertical_members: int = 3,
) -> list[tuple[Box, str]]:
    """Create both horizontal and vertical red-text sequence proposals.

    Horizontal grouping preserves the previous sequence-OCR behavior.  Vertical
    grouping links components with overlapping x-ranges and small y-gaps, which
    is necessary for passport/document numbers printed from top to bottom.
    Elongated singleton components are also retained because several printed
    fields are already connected before grouping.
    """
    if not boxes:
        return []

    # Horizontal groups: reuse the established graph grouping.
    hgroups = group_nearby_boxes(
        boxes,
        max_gap=horizontal_gap,
        min_vertical_overlap=min_alignment,
        adaptive_gap=True,
        gap_height_ratio=0.9,
    )

    # Vertical graph over original connected components.
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    median_w = float(median([b[2] for b in boxes])) if boxes else 1.0
    effective_vgap = max(vertical_gap, int(round(median_w * 1.25)))

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            x_overlap = _horizontal_overlap_ratio(a, b)
            ac = a[0] + a[2] / 2.0
            bc = b[0] + b[2] / 2.0
            x_center_ok = abs(ac - bc) <= 0.65 * max(a[2], b[2])
            if not (x_overlap >= min_alignment or x_center_ok):
                continue
            if _vertical_gap(a, b) <= effective_vgap:
                union(i, j)

    members: dict[int, list[Box]] = {}
    for i, box in enumerate(boxes):
        members.setdefault(find(i), []).append(box)

    vgroups: list[Box] = []
    for group in members.values():
        if len(group) < min_vertical_members:
            continue
        merged = _merge_box_set(group)
        if merged[3] >= 1.6 * merged[2]:
            vgroups.append(merged)

    candidates: list[tuple[Box, str]] = []
    for b in vgroups:
        candidates.append((b, "vertical"))

    def mostly_inside(inner: Box, outer: Box) -> bool:
        ix1 = max(inner[0], outer[0])
        iy1 = max(inner[1], outer[1])
        ix2 = min(inner[0] + inner[2], outer[0] + outer[2])
        iy2 = min(inner[1] + inner[3], outer[1] + outer[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return inter / max(1, inner[2] * inner[3]) >= 0.70

    for b in hgroups:
        # Individual glyphs from a vertical sequence can themselves look like
        # short horizontal boxes.  Suppress them once a vertical text region
        # already explains the same pixels.
        if any(mostly_inside(b, vb) for vb in vgroups):
            continue
        if b[2] >= 1.5 * b[3]:
            candidates.append((b, "horizontal"))

    # Suppress near-duplicate boxes of the same orientation.
    unique: list[tuple[Box, str]] = []
    for box, orientation in sorted(candidates, key=lambda z: (z[0][1], z[0][0])):
        x, y, w, h = box
        duplicate = False
        for kept, kept_orientation in unique:
            if orientation != kept_orientation:
                continue
            kx, ky, kw, kh = kept
            ix = max(0, min(x + w, kx + kw) - max(x, kx))
            iy = max(0, min(y + h, ky + kh) - max(y, ky))
            inter = ix * iy
            union_area = w * h + kw * kh - inter
            if union_area and inter / union_area > 0.75:
                duplicate = True
                break
        if not duplicate:
            unique.append((box, orientation))
    return unique
