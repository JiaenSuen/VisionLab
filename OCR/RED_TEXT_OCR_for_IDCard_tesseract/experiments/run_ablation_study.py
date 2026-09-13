"""Controlled ablation study for localization, crop padding, grouping, and Tesseract OCR.

The repository examples form a small deterministic engineering case study. The
results support implementation decisions but are not presented as population-
level statistical evidence.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.red_components import find_component_boxes, red_filter  # noqa: E402
from modules.sequence_groups import (  # noqa: E402
    draw_group_boxes,
    estimate_grouping_gap,
    expand_box,
    expand_box_adaptive,
    expand_boxes_for_ocr,
    group_nearby_boxes,
)
from modules.tesseract_ocr import annotate_ocr_results, recognize_grouped_text  # noqa: E402

EXAMPLES = [ROOT / "examples" / "1.jpg", ROOT / "examples" / "2.png"]
EXPECTED_TEXT = {
    "1.jpg": "A234567890",
    "2.png": "BTS4ever J-hope",
}
ROI = (0.63, 0.70, 1.0, 1.0)


def _local_red_diagnostics(image: np.ndarray):
    image_h, image_w = image.shape[:2]
    x1 = int(image_w * ROI[0])
    y1 = int(image_h * ROI[1])
    x2 = int(image_w * ROI[2])
    y2 = int(image_h * ROI[3])
    crop = image[y1:y2, x1:x2]
    red_axis = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 1]
    baseline = cv2.GaussianBlur(red_axis, (0, 0), 3.0)
    contrast = cv2.subtract(red_axis, baseline)
    threshold, mask = cv2.threshold(
        contrast,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return contrast, float(threshold), mask, (x1, y1)


def _soft_support_coverage(
    image: np.ndarray,
    group_box,
    detect_mask_roi: np.ndarray,
    contrast_roi: np.ndarray,
    otsu_threshold: float,
    offset,
    crop_box,
    low_threshold_ratio: float = 0.5,
) -> float:
    image_h, image_w = image.shape[:2]
    ox, oy = offset

    hard = np.zeros((image_h, image_w), dtype=np.uint8)
    hard[oy : oy + detect_mask_roi.shape[0], ox : ox + detect_mask_roi.shape[1]] = detect_mask_roi

    low_threshold = max(1.0, otsu_threshold * low_threshold_ratio)
    low_roi = (contrast_roi >= low_threshold).astype(np.uint8) * 255
    low = np.zeros_like(hard)
    low[oy : oy + low_roi.shape[0], ox : ox + low_roi.shape[1]] = low_roi

    gx, gy, gw, gh = group_box
    seed = np.zeros_like(hard)
    seed[gy : gy + gh, gx : gx + gw] = hard[gy : gy + gh, gx : gx + gw]
    vicinity = cv2.dilate(seed, np.ones((7, 7), np.uint8), iterations=1)
    support = (low > 0) & (vicinity > 0)

    total = int(support.sum())
    if total == 0:
        return 1.0

    cx, cy, cw, ch = crop_box
    inside = np.zeros_like(support)
    inside[cy : cy + ch, cx : cx + cw] = True
    return float((support & inside).sum() / total)


def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def _cer(reference: str, prediction: str) -> float:
    return _levenshtein(reference, prediction) / max(1, len(reference))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    padding_rows: list[dict] = []
    threshold_rows: list[dict] = []
    grouping_rows: list[dict] = []
    ocr_rows: list[dict] = []

    for image_path in EXAMPLES:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)

        raw_mask, offset = red_filter(image, roi=ROI)
        boxes, detect_mask = find_component_boxes(raw_mask, offset=offset)
        contrast, otsu_threshold, _, diagnostic_offset = _local_red_diagnostics(image)
        assert offset == diagnostic_offset

        adaptive_gap = estimate_grouping_gap(boxes, minimum_gap=8, gap_height_ratio=1.0)
        groups = group_nearby_boxes(
            boxes,
            max_gap=8,
            adaptive_gap=True,
            gap_height_ratio=1.0,
        )

        # Padding ablation: preserve detector, vary recognition crop only.
        strategies = {
            "tight_0px": lambda box: expand_box(box, image.shape, 0),
            "fixed_2px": lambda box: expand_box(box, image.shape, 2),
            "fixed_4px": lambda box: expand_box(box, image.shape, 4),
            "adaptive": lambda box: expand_box_adaptive(box, image.shape),
        }
        for group_index, group in enumerate(groups, start=1):
            for strategy_name, strategy in strategies.items():
                crop_box = strategy(group)
                coverage = _soft_support_coverage(
                    image,
                    group,
                    detect_mask,
                    contrast,
                    otsu_threshold,
                    offset,
                    crop_box,
                )
                padding_rows.append(
                    {
                        "image": image_path.name,
                        "group": group_index,
                        "strategy": strategy_name,
                        "soft_support_coverage": f"{coverage:.4f}",
                        "crop_width": crop_box[2],
                        "crop_height": crop_box[3],
                    }
                )

        # Threshold ablation: relax Otsu and observe component/group topology.
        for factor in (1.0, 0.85, 0.75, 0.60):
            threshold = max(1.0, otsu_threshold * factor)
            _, relaxed_mask = cv2.threshold(contrast, threshold, 255, cv2.THRESH_BINARY)
            relaxed_boxes, _ = find_component_boxes(relaxed_mask, offset=offset)
            relaxed_groups = group_nearby_boxes(
                relaxed_boxes,
                max_gap=8,
                adaptive_gap=True,
                gap_height_ratio=1.0,
            )
            threshold_rows.append(
                {
                    "image": image_path.name,
                    "otsu_threshold": f"{otsu_threshold:.2f}",
                    "threshold_factor": f"{factor:.2f}",
                    "effective_threshold": f"{threshold:.2f}",
                    "foreground_pixels": int((relaxed_mask > 0).sum()),
                    "component_boxes": len(relaxed_boxes),
                    "sequence_groups": len(relaxed_groups),
                }
            )

        # Grouping ablation: fixed versus scale-aware horizontal distance.
        grouping_variants = [
            ("fixed_8", dict(max_gap=8, adaptive_gap=False)),
            ("fixed_10", dict(max_gap=10, adaptive_gap=False)),
            ("adaptive", dict(max_gap=8, adaptive_gap=True, gap_height_ratio=1.0)),
        ]
        for name, kwargs in grouping_variants:
            variant_groups = group_nearby_boxes(boxes, **kwargs)
            grouping_rows.append(
                {
                    "image": image_path.name,
                    "strategy": name,
                    "component_boxes": len(boxes),
                    "effective_gap": adaptive_gap if name == "adaptive" else kwargs["max_gap"],
                    "sequence_groups": len(variant_groups),
                    "largest_group_width": max((box[2] for box in variant_groups), default=0),
                }
            )

        # Visualization: tight detector geometry versus expanded recognition crops.
        adaptive_crop_boxes = [expand_box_adaptive(box, image.shape) for box in groups]
        visualization = draw_group_boxes(image, groups, thickness=1)
        visualization = draw_group_boxes(visualization, adaptive_crop_boxes, thickness=2)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_tight_vs_adaptive.png"), visualization)

        # Tesseract preprocessing ablation on the same grouped/crop geometry.
        crop_boxes = expand_boxes_for_ocr(groups, image.shape, mode="adaptive")
        reference = EXPECTED_TEXT[image_path.name]
        for preprocess in ("raw", "gray", "otsu"):
            results = recognize_grouped_text(
                image,
                groups,
                crop_boxes,
                preprocess_mode=preprocess,
                scale=4.0,
                min_group_aspect_ratio=3.0,
            )
            prediction = " | ".join(result.text for result in results if result.text)
            ocr_rows.append(
                {
                    "image": image_path.name,
                    "preprocess": preprocess,
                    "reference": reference,
                    "prediction": prediction,
                    "exact_match": int(prediction == reference),
                    "cer": f"{_cer(reference, prediction):.4f}",
                }
            )

            if preprocess == "gray":
                annotated = annotate_ocr_results(image, results)
                cv2.imwrite(str(output_dir / f"{image_path.stem}_tesseract_result.png"), annotated)

    _write_csv(output_dir / "padding_ablation.csv", padding_rows)
    _write_csv(output_dir / "threshold_ablation.csv", threshold_rows)
    _write_csv(output_dir / "grouping_ablation.csv", grouping_rows)
    _write_csv(output_dir / "tesseract_ocr_ablation.csv", ocr_rows)

    summary_by_strategy: dict[str, list[float]] = {}
    for row in padding_rows:
        summary_by_strategy.setdefault(row["strategy"], []).append(float(row["soft_support_coverage"]))

    ocr_by_mode: dict[str, list[dict]] = {}
    for row in ocr_rows:
        ocr_by_mode.setdefault(row["preprocess"], []).append(row)

    lines = [
        "# Ablation Study Summary",
        "",
        "This report was generated by `experiments/run_ablation_study.py` using the two repository examples. It is a controlled engineering case study, not a dataset-level benchmark.",
        "",
        "## Recognition-Crop Padding",
        "",
        "The detector is intentionally kept tight. Recognition crops are expanded separately so faint anti-aliased stroke evidence can be recovered without changing connected-component topology.",
        "",
        "| Strategy | Mean soft-support coverage | Minimum coverage |",
        "|---|---:|---:|",
    ]
    for strategy, values in summary_by_strategy.items():
        lines.append(f"| {strategy} | {sum(values)/len(values):.4f} | {min(values):.4f} |")

    lines += [
        "",
        "## Threshold Relaxation",
        "",
        "Lower segmentation thresholds recover additional red-contrast pixels but also alter component topology. The current evidence supports preserving the stable Otsu-based detector and solving crop truncation at the recognition-crop stage.",
        "",
        "## Grouping Distance",
        "",
        "A fixed pixel gap is resolution-sensitive. Scaling the grouping distance with median component height preserves the earlier small-image behavior while forming a coherent long sequence on the larger example.",
        "",
        "## Tesseract OCR Baseline",
        "",
        "Tesseract is evaluated only after the localization/grouping pipeline. The red detector therefore remains interpretable and independent from recognition. The table reports exact sequence accuracy and character error rate (CER) on the two controlled examples.",
        "",
        "| Preprocess | Exact matches | Mean CER |",
        "|---|---:|---:|",
    ]
    for mode, rows in ocr_by_mode.items():
        exact = sum(int(row["exact_match"]) for row in rows)
        mean_cer = sum(float(row["cer"]) for row in rows) / len(rows)
        lines.append(f"| {mode} | {exact}/{len(rows)} | {mean_cer:.4f} |")

    lines += [
        "",
        "In the present examples, grayscale upsampling is the strongest default because it preserves anti-aliased edge intensity while increasing the effective character size seen by Tesseract. Hard Otsu binarization can remove or reshape weak strokes, while raw color crops leave unnecessary background variation. This finding is intentionally limited to the current examples and should be re-evaluated on a larger labeled set.",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run(ROOT / "experiments" / "results")
