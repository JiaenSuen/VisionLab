"""Reusable modules for red-text localization, grouping, and OCR."""

from .red_components import draw_boxes, find_component_boxes, red_filter
from .sequence_groups import (
    draw_group_boxes,
    expand_boxes_for_ocr,
    export_group_crops,
    group_nearby_boxes,
    group_red_text_regions,
)
from .tesseract_ocr import (
    TesseractResult,
    annotate_ocr_results,
    prepare_tesseract_crop,
    recognize_red_text,
)

__all__ = [
    "red_filter",
    "find_component_boxes",
    "draw_boxes",
    "group_nearby_boxes",
    "group_red_text_regions",
    "expand_boxes_for_ocr",
    "export_group_crops",
    "draw_group_boxes",
    "TesseractResult",
    "prepare_tesseract_crop",
    "recognize_red_text",
    "annotate_ocr_results",
]
