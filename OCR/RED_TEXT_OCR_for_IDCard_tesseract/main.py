"""Run red-text localization, sequence grouping, and Tesseract OCR."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pytesseract

from modules.red_components import draw_boxes, find_component_boxes, red_filter
from modules.sequence_groups import (
    draw_group_boxes,
    estimate_grouping_gap,
    expand_boxes_for_ocr,
    export_group_crops,
    group_nearby_boxes,
)
from modules.tesseract_ocr import (
    annotate_ocr_results,
    configure_tesseract,
    recognize_grouped_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Red-text localization + Tesseract OCR pipeline")
    parser.add_argument(
        "image",
        nargs="?",
        default="examples/PERU_ID_04.bmp",
        help="Path to the input image (default: examples/1.jpg)",
    )
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        default=(0.63, 0.70, 1.0, 1.0),
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Normalized ROI. Default reproduces the successful prototype.",
    )
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--min-width", type=int, default=2)
    parser.add_argument("--min-height", type=int, default=5)

    parser.add_argument("--max-gap", type=int, default=8)
    parser.add_argument(
        "--adaptive-gap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale grouping distance by median component height (default: enabled).",
    )
    parser.add_argument("--gap-height-ratio", type=float, default=1.0)
    parser.add_argument("--vertical-overlap", type=float, default=0.25)

    parser.add_argument(
        "--padding-mode",
        choices=("fixed", "adaptive"),
        default="adaptive",
        help="Recognition-crop padding strategy.",
    )
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--min-padding", type=int, default=4)
    parser.add_argument("--pad-x-height-ratio", type=float, default=0.35)
    parser.add_argument("--pad-y-height-ratio", type=float, default=0.20)

    # Tesseract settings. The defaults target one red alphanumeric text line.
    parser.add_argument("--tesseract-cmd", default=None, help="Optional path to tesseract.exe")
    parser.add_argument("--ocr-lang", default="eng")
    parser.add_argument("--ocr-psm", type=int, default=7, help="Tesseract page segmentation mode")
    parser.add_argument(
        "--ocr-preprocess",
        choices=("raw", "gray", "otsu"),
        default="gray",
        help="Input representation passed to Tesseract.",
    )
    parser.add_argument("--ocr-scale", type=float, default=4.0, help="Upsampling factor before OCR")
    parser.add_argument(
        "--ocr-min-aspect",
        type=float,
        default=3.0,
        help="Skip short groups below this width/height ratio. Use 0 to OCR every group.",
    )
    parser.add_argument(
        "--ocr-whitelist",
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-",
        help="Allowed Tesseract characters. Use an empty string to disable the whitelist.",
    )
    return parser.parse_args()


def _write_predictions_csv(path: Path, results) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group",
                "status",
                "prediction",
                "aspect_ratio",
                "group_box_x",
                "group_box_y",
                "group_box_w",
                "group_box_h",
                "crop_box_x",
                "crop_box_y",
                "crop_box_w",
                "crop_box_h",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.group_index,
                    result.status,
                    result.text,
                    f"{result.aspect_ratio:.4f}",
                    *result.group_box,
                    *result.crop_box,
                ]
            )


def main() -> None:
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    output_dir = Path(args.output)
    crop_dir = output_dir / "ocr_inputs"
    prepared_dir = output_dir / "ocr_preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    roi = tuple(args.roi)

    # Stage 1: preserve the successful RedCharacterDetect localization baseline.
    mask, offset = red_filter(image, roi=roi)
    boxes, detect_mask = find_component_boxes(
        mask,
        offset=offset,
        min_area=args.min_area,
        min_width=args.min_width,
        min_height=args.min_height,
    )

    # Stage 2: convert imperfect components into sequence-level OCR proposals.
    group_boxes = group_nearby_boxes(
        boxes,
        max_gap=args.max_gap,
        min_vertical_overlap=args.vertical_overlap,
        adaptive_gap=args.adaptive_gap,
        gap_height_ratio=args.gap_height_ratio,
    )

    # Detection geometry remains tight; recognition crops are expanded separately.
    crop_boxes = expand_boxes_for_ocr(
        group_boxes,
        image.shape,
        mode=args.padding_mode,
        padding=args.padding,
        min_padding=args.min_padding,
        horizontal_height_ratio=args.pad_x_height_ratio,
        vertical_height_ratio=args.pad_y_height_ratio,
    )

    # Debug outputs keep each stage observable.
    cv2.imwrite(str(output_dir / "character_mask.png"), mask)
    cv2.imwrite(str(output_dir / "detect_mask.png"), detect_mask)
    cv2.imwrite(str(output_dir / "component_boxes.png"), draw_boxes(image, boxes))
    cv2.imwrite(str(output_dir / "group_boxes.png"), draw_group_boxes(image, group_boxes))
    cv2.imwrite(str(output_dir / "ocr_crop_boxes.png"), draw_group_boxes(image, crop_boxes))

    export_group_crops(
        image,
        group_boxes,
        crop_dir,
        padding=args.padding,
        target_height=None,
        padding_mode=args.padding_mode,
        min_padding=args.min_padding,
        horizontal_height_ratio=args.pad_x_height_ratio,
        vertical_height_ratio=args.pad_y_height_ratio,
    )

    # Stage 3: Tesseract OCR on expanded crops from the original image.
    configure_tesseract(args.tesseract_cmd)
    try:
        results = recognize_grouped_text(
            image,
            group_boxes,
            crop_boxes,
            language=args.ocr_lang,
            psm=args.ocr_psm,
            whitelist=args.ocr_whitelist,
            preprocess_mode=args.ocr_preprocess,
            scale=args.ocr_scale,
            min_group_aspect_ratio=args.ocr_min_aspect,
            processed_dir=prepared_dir,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR executable was not found. Install Tesseract and add it to PATH, "
            "or provide --tesseract-cmd with the full path to tesseract.exe."
        ) from exc

    csv_path = output_dir / "tesseract_predictions.csv"
    _write_predictions_csv(csv_path, results)

    annotated = annotate_ocr_results(image, results)
    result_path = output_dir / "ocr_result.png"
    cv2.imwrite(str(result_path), annotated)

    effective_gap = (
        estimate_grouping_gap(
            boxes,
            minimum_gap=args.max_gap,
            gap_height_ratio=args.gap_height_ratio,
        )
        if args.adaptive_gap
        else args.max_gap
    )

    print(f"Character/component boxes: {len(boxes)}")
    print(f"Grouping gap used:          {effective_gap} px")
    print(f"Merged text groups:         {len(group_boxes)}")
    print(f"Tesseract predictions:      {csv_path}")
    for result in results:
        if result.status == "skipped_geometry":
            print(f"  group {result.group_index}: skipped (aspect={result.aspect_ratio:.2f})")
        else:
            print(f"  group {result.group_index}: {result.text!r}")
    print(f"Annotated full image:       {result_path}")


if __name__ == "__main__":
    main()
