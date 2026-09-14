"""Detect and recognize red text in passport/document-card images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pytesseract

from modules.red_components import draw_boxes, find_component_boxes, red_filter
from modules.sequence_groups import draw_group_boxes, expand_boxes_for_ocr, group_red_text_regions
from modules.tesseract_ocr import annotate_ocr_results, configure_tesseract, recognize_red_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Document-wide red-text OCR (horizontal + vertical)")
    p.add_argument("image", nargs="?", default="examples/passport_card_sample.bmp")
    p.add_argument("--output", default="outputs")
    p.add_argument("--red-mode", choices=("hsv", "lab"), default="hsv")
    p.add_argument("--roi", nargs=4, type=float, default=(0.0,0.0,1.0,1.0), metavar=("X1","Y1","X2","Y2"))
    p.add_argument("--min-area", type=int, default=5)
    p.add_argument("--max-area-ratio", type=float, default=0.035)
    p.add_argument("--horizontal-gap", type=int, default=18)
    p.add_argument("--vertical-gap", type=int, default=24)
    p.add_argument("--padding", type=int, default=5)
    p.add_argument("--region-min-red-saturation", type=float, default=110.0, help="Reject skin/background candidates whose red pixels are weakly saturated.")
    p.add_argument("--tesseract-cmd", default=None)
    p.add_argument("--ocr-lang", default="eng")
    p.add_argument("--ocr-psm", type=int, default=7)
    p.add_argument("--ocr-scale", type=float, default=4.0)
    p.add_argument("--ocr-preprocess", choices=("auto","raw","gray","otsu","redmask"), default="auto")
    p.add_argument("--ocr-whitelist", default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prepared = out / "ocr_preprocessed"

    mask, offset = red_filter(image, tuple(args.roi), mode=args.red_mode)
    boxes, detect_mask = find_component_boxes(
        mask,
        offset=offset,
        min_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
    )

    regions = group_red_text_regions(
        boxes,
        horizontal_gap=args.horizontal_gap,
        vertical_gap=args.vertical_gap,
    )
    # Region-level color validation removes reddish skin/background fragments
    # while retaining strongly printed red text.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    filtered_regions = []
    for box, orientation in regions:
        x, y, w, h = box
        patch = hsv[y:y+h, x:x+w]
        hue, sat, _ = cv2.split(patch)
        red_pixels = ((hue <= 15) | (hue >= 165)) & (sat >= 90)
        values = sat[red_pixels]
        median_sat = float(__import__("numpy").median(values)) if values.size else 0.0
        if median_sat >= args.region_min_red_saturation:
            filtered_regions.append((box, orientation))
    regions = filtered_regions

    region_boxes = [b for b, _ in regions]
    crop_boxes = expand_boxes_for_ocr(
        region_boxes,
        image.shape,
        mode="fixed",
        padding=args.padding,
    )

    cv2.imwrite(str(out / "red_mask.png"), mask)
    cv2.imwrite(str(out / "component_boxes.png"), draw_boxes(image, boxes))
    cv2.imwrite(str(out / "group_boxes.png"), draw_group_boxes(image, region_boxes))

    configure_tesseract(args.tesseract_cmd)
    try:
        results = recognize_red_text(
            image,
            regions,
            crop_boxes,
            language=args.ocr_lang,
            psm=args.ocr_psm,
            whitelist=args.ocr_whitelist,
            preprocess_mode=args.ocr_preprocess,
            scale=args.ocr_scale,
            processed_dir=prepared,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract not found. Install it or pass --tesseract-cmd.") from exc

    with (out / "tesseract_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index","orientation","prediction","x","y","w","h"])
        for r in results:
            writer.writerow([r.group_index, r.orientation, r.text, *r.group_box])

    annotated = annotate_ocr_results(image, results)
    cv2.imwrite(str(out / "ocr_result.png"), annotated)

    print(f"Red components: {len(boxes)}")
    print(f"Text regions:   {len(regions)}")
    for r in results:
        print(f"  {r.group_index:02d} [{r.orientation}] {r.text!r}")
    print(f"Final result:   {out / 'ocr_result.png'}")


if __name__ == "__main__":
    main()
