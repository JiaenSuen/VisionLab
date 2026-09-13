"""Tesseract OCR utilities for grouped red-text candidates.

The localization pipeline intentionally remains independent from OCR. Tight
connected-component boxes are grouped first, then expanded recognition crops are
passed to Tesseract. This separation preserves the stable detector while giving
OCR enough contextual pixels around weak anti-aliased character edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pytesseract

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class TesseractResult:
    """Recognition result associated with one grouped text proposal."""

    group_index: int
    text: str
    group_box: Box
    crop_box: Box
    aspect_ratio: float
    status: str


def configure_tesseract(tesseract_cmd: str | None = None) -> None:
    """Optionally configure the path to the Tesseract executable."""
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def prepare_tesseract_crop(
    crop: np.ndarray,
    mode: str = "gray",
    scale: float = 4.0,
) -> np.ndarray:
    """Prepare a crop for Tesseract while preserving character-edge evidence.

    ``gray`` is the default because experiments on the repository examples show
    that simple grayscale upsampling preserves anti-aliased stroke information
    better than aggressive binarization. ``otsu`` is retained as an ablation
    option, and ``raw`` keeps the color crop.
    """
    if crop is None or crop.size == 0:
        raise ValueError("Cannot OCR an empty crop.")
    if mode not in {"raw", "gray", "otsu"}:
        raise ValueError("mode must be one of: raw, gray, otsu")
    if scale <= 0:
        raise ValueError("scale must be positive")

    if mode == "raw":
        prepared = crop.copy()
    else:
        prepared = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if scale != 1.0:
        prepared = cv2.resize(
            prepared,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    if mode == "otsu":
        _, prepared = cv2.threshold(
            prepared,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    return prepared


def recognize_grouped_text(
    image: np.ndarray,
    group_boxes: Sequence[Box],
    crop_boxes: Sequence[Box],
    *,
    language: str = "eng",
    psm: int = 7,
    oem: int = 3,
    whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-",
    preprocess_mode: str = "gray",
    scale: float = 4.0,
    min_group_aspect_ratio: float = 3.0,
    processed_dir: str | Path | None = None,
) -> list[TesseractResult]:
    """Recognize grouped text candidates using Tesseract.

    The optional aspect-ratio filter suppresses short red labels that are not
    expected to be alphanumeric sequence fields. Set it to ``0`` to OCR every
    group. Recognition always uses the expanded crop box, not the tight detector
    geometry.
    """
    if len(group_boxes) != len(crop_boxes):
        raise ValueError("group_boxes and crop_boxes must have the same length")

    processed_path = Path(processed_dir) if processed_dir is not None else None
    if processed_path is not None:
        processed_path.mkdir(parents=True, exist_ok=True)

    config_parts = [
        f"--oem {oem}",
        f"--psm {psm}",
        "-c preserve_interword_spaces=1",
    ]
    if whitelist:
        config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
    config = " ".join(config_parts)

    results: list[TesseractResult] = []

    for index, (group_box, crop_box) in enumerate(zip(group_boxes, crop_boxes), start=1):
        gx, gy, gw, gh = group_box
        aspect_ratio = gw / max(1.0, float(gh))

        if min_group_aspect_ratio > 0 and aspect_ratio < min_group_aspect_ratio:
            results.append(
                TesseractResult(
                    group_index=index,
                    text="",
                    group_box=group_box,
                    crop_box=crop_box,
                    aspect_ratio=aspect_ratio,
                    status="skipped_geometry",
                )
            )
            continue

        x, y, width, height = crop_box
        crop = image[y : y + height, x : x + width]
        prepared = prepare_tesseract_crop(crop, mode=preprocess_mode, scale=scale)

        if processed_path is not None:
            cv2.imwrite(str(processed_path / f"text_group_{index:03d}_prepared.png"), prepared)

        text = pytesseract.image_to_string(
            prepared,
            lang=language,
            config=config,
        )
        text = " ".join(text.strip().split())

        results.append(
            TesseractResult(
                group_index=index,
                text=text,
                group_box=group_box,
                crop_box=crop_box,
                aspect_ratio=aspect_ratio,
                status="recognized" if text else "empty",
            )
        )

    return results


def annotate_ocr_results(
    image: np.ndarray,
    results: Sequence[TesseractResult],
    *,
    prefix: str = "OCR",
    draw_crop_box: bool = True,
) -> np.ndarray:
    """Return the full image with OCR boxes plus a non-destructive result panel.

    The document itself is kept visually intact except for thin localization
    rectangles. Recognized strings are rendered in a panel appended below the
    original image so labels do not cover document fields.
    """
    recognized = [result for result in results if result.text]
    image_h, image_w = image.shape[:2]

    line_height = max(22, int(round(image_h * 0.085)))
    panel_height = max(42, 12 + line_height * max(1, len(recognized)))
    output = cv2.copyMakeBorder(
        image,
        0,
        panel_height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    for result in recognized:
        x, y, width, height = result.crop_box if draw_crop_box else result.group_box
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 200, 0),
            1,
        )

    cv2.line(output, (0, image_h), (image_w - 1, image_h), (0, 0, 0), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, min(0.72, image_w / 900.0))
    thickness = 1 if image_w < 900 else 2

    if not recognized:
        cv2.putText(
            output,
            f"{prefix}: no alphanumeric sequence recognized",
            (8, image_h + line_height),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        return output

    for row, result in enumerate(recognized):
        label = f"{prefix} {row + 1}: {result.text}"
        text_y = image_h + 8 + line_height * (row + 1) - 5
        cv2.putText(
            output,
            label,
            (8, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return output
