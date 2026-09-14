"""Tesseract OCR helpers for horizontal and vertical red-text regions."""

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
    group_index: int
    text: str
    group_box: Box
    crop_box: Box
    orientation: str
    status: str


def configure_tesseract(tesseract_cmd: str | None = None) -> None:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def prepare_tesseract_crop(crop: np.ndarray, mode: str = "gray", scale: float = 4.0) -> np.ndarray:
    if crop is None or crop.size == 0:
        raise ValueError("Cannot OCR an empty crop.")
    if mode not in {"raw", "gray", "otsu", "redmask"}:
        raise ValueError("mode must be one of: raw, gray, otsu, redmask")

    if mode == "raw":
        prepared = crop.copy()
    elif mode == "redmask":
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        low = cv2.inRange(hsv, (0, 90, 30), (15, 255, 255))
        high = cv2.inRange(hsv, (165, 90, 30), (180, 255, 255))
        red = cv2.bitwise_or(low, high)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        prepared = 255 - red
    else:
        prepared = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if scale != 1.0:
        prepared = cv2.resize(prepared, None, fx=scale, fy=scale, interpolation=(cv2.INTER_NEAREST if mode == "redmask" else cv2.INTER_CUBIC))
    if mode == "otsu":
        _, prepared = cv2.threshold(prepared, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return prepared


def _ocr_once(image: np.ndarray, language: str, psm: int, whitelist: str) -> tuple[str, float]:
    config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    data = pytesseract.image_to_data(
        image,
        lang=language,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    words = []
    confidences = []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        try:
            conf_value = float(conf)
        except (TypeError, ValueError):
            conf_value = -1.0
        if text:
            words.append(text)
            if conf_value >= 0:
                confidences.append(conf_value)
    return " ".join(words), (sum(confidences) / len(confidences) if confidences else -1.0)


def recognize_red_text(
    image: np.ndarray,
    regions: Sequence[tuple[Box, str]],
    crop_boxes: Sequence[Box],
    *,
    language: str = "eng",
    psm: int = 7,
    whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-",
    preprocess_mode: str = "auto",
    scale: float = 4.0,
    processed_dir: str | Path | None = None,
) -> list[TesseractResult]:
    """OCR all red-text regions, automatically handling vertical text.

    Vertical regions are rotated in both directions; the candidate with the
    higher Tesseract confidence (then longer text) is selected.
    """
    if len(regions) != len(crop_boxes):
        raise ValueError("regions and crop_boxes must have the same length")

    processed_path = Path(processed_dir) if processed_dir is not None else None
    if processed_path:
        processed_path.mkdir(parents=True, exist_ok=True)

    results: list[TesseractResult] = []
    for index, ((group_box, orientation), crop_box) in enumerate(zip(regions, crop_boxes), start=1):
        x, y, w, h = crop_box
        crop = image[y:y+h, x:x+w]

        candidates: list[tuple[str, float, np.ndarray]] = []
        if orientation == "vertical":
            # Vertical printing is much more stable after suppressing the document
            # background and rotating the red-only text line into reading order.
            for rotated in (
                cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
            ):
                prepared = prepare_tesseract_crop(rotated, "redmask", scale)
                text, confidence = _ocr_once(prepared, language, psm, whitelist)
                candidates.append((text, confidence, prepared))
        else:
            modes = ("gray", "redmask") if preprocess_mode == "auto" else (preprocess_mode,)
            for mode in modes:
                prepared = prepare_tesseract_crop(crop, mode, scale)
                text, confidence = _ocr_once(prepared, language, psm, whitelist)
                candidates.append((text, confidence, prepared))

        best_text, best_conf, best_prepared = max(
            candidates,
            key=lambda item: (item[1], len(item[0].replace(" ", ""))),
        )

        if processed_path is not None:
            cv2.imwrite(str(processed_path / f"red_text_{index:03d}_{orientation}.png"), best_prepared)

        results.append(
            TesseractResult(
                group_index=index,
                text=" ".join(best_text.split()),
                group_box=group_box,
                crop_box=crop_box,
                orientation=orientation,
                status="recognized" if best_text else "empty",
            )
        )
    return results


def annotate_ocr_results(image: np.ndarray, results: Sequence[TesseractResult]) -> np.ndarray:
    """Draw all recognized red-text regions and append one consolidated result panel."""
    visible = [r for r in results if r.text]
    h, w = image.shape[:2]
    line_h = max(24, int(h * 0.065))
    panel_h = max(50, 16 + line_h * max(1, len(visible)))
    output = cv2.copyMakeBorder(image, 0, panel_h, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))

    for r in visible:
        x, y, bw, bh = r.crop_box
        color = (0, 180, 0) if r.orientation == "horizontal" else (255, 0, 255)
        cv2.rectangle(output, (x,y), (x+bw,y+bh), color, 2)

    cv2.line(output, (0,h), (w-1,h), (0,0,0), 1)
    if not visible:
        cv2.putText(output, "OCR: no red text recognized", (8,h+line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
        return output

    for row, r in enumerate(visible, start=1):
        label = f"{row}. [{r.orientation}] {r.text}"
        cv2.putText(output, label, (8, h + row*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return output
