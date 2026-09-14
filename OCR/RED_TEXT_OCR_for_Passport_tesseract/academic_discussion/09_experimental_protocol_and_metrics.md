# Experimental Protocol and Metrics

## 1. Why a Protocol Is Needed

The current project has produced meaningful qualitative findings, but a research-quality result requires a repeatable dataset, labels, metrics, and controlled ablations. The next phase should preserve the successful baseline and evaluate one design factor at a time. The goal is not to maximize one screenshot but to determine which choices remain beneficial across variation in image quality, scale, compression, and document instances.

## 2. Dataset Construction

Create a small benchmark with at least 50–100 cropped document images if available. Include multiple acquisition conditions: original resolution, downsampled images, JPEG-compressed images, mild brightness/color shifts, and small geometric perturbations. Each sample should have the ground-truth target string and, for a subset, character or line bounding boxes. If manual pixel masks are too expensive, line-level bounding boxes plus text labels are sufficient to evaluate grouping and OCR. Split the data into development and test sets before tuning thresholds.

## 3. Segmentation/Localization Metrics

For pixel masks, use foreground precision, recall, F1, and IoU. Because OCR is sensitive to missing strokes, recall should be considered separately from precision. For component boxes, use detection precision/recall at an IoU threshold, but also report **character coverage**: the fraction of ground-truth character area contained by at least one accepted proposal after padding. A high-precision detector that truncates character edges may be worse for OCR than a slightly larger crop.

Useful structural error metrics are **fragmentation rate** (average number of predicted components per ground-truth character) and **merge rate** (fraction of predicted components overlapping more than one ground-truth character). These directly quantify the failure modes that motivated sequence grouping.

## 4. Grouping Metrics

For merged sequence crops, report **group completeness** (fraction of characters from one target sequence included in the correct group), **group purity** (fraction of pixels/characters in a group that belong to the intended sequence), and the number of groups per target field. The ideal value is often one group per text line, but this depends on the document layout. Record crop width/height statistics because extremely wide groups can indicate over-merging.

## 5. Recognition Metrics

Once OCR is attached, use **Character Error Rate (CER)** as the primary sequence metric:

\[
CER = \frac{S + D + I}{N},
\]

where `S`, `D`, and `I` are substitutions, deletions, and insertions relative to a ground-truth string of length `N`. Also report exact-string accuracy. For a per-character CNN baseline, report classification accuracy only on correctly segmented single-character crops, and separately report end-to-end string accuracy so segmentation errors are not hidden.

## 6. Controlled Ablation Table

A recommended first table is:

| Experiment | ROI before Otsu | Feature | Morphology | Component filters | Grouping | Primary metric |
|---|---:|---|---|---|---|---|
| A | Yes | HSV hard threshold | 3x3 open+close | baseline | No | character coverage |
| B | Yes | HSV hard threshold | 2x2 close | baseline | No | character coverage |
| C | Yes | Lab local contrast | none | baseline | No | fragmentation rate |
| D (baseline) | Yes | Lab local contrast | 2x2 close | 8/2/5 | No | detection F1 |
| E | No (late ROI) | Lab local contrast | 2x2 close | 8/2/5 | No | false positives |
| F | Yes | Lab local contrast | 2x2 close | 8/2/5 | Yes | CER / group purity |

The table isolates one main causal factor at a time. Experiment E is especially important because it directly tests the regression mechanism observed during modularization.

## 7. Reporting Concrete Findings

A future academic result should use wording such as: “Applying ROI cropping before Otsu reduced false-positive components by X% relative to late ROI masking across N test images,” or “2x2 closing reduced fragmentation from A to B components per character while increasing merge rate by only C.” These statements are stronger than “the image looked cleaner” because they identify the measured behavior, effect size, and sample size.

## Diagnostic Metric Added: Soft Stroke-Support Coverage

For the bbox-padding study, a diagnostic coverage proxy was introduced to estimate whether a recognition crop retains faint red evidence near the hard segmentation boundary. The local-red contrast image is thresholded at a lower level than Otsu, but candidate pixels are restricted to a small dilation neighborhood around the accepted hard mask. Coverage is then the fraction of this local support contained by the proposed recognition crop. The metric is deliberately described as a proxy rather than ground truth because it is derived from the same image signal.

The current two-image study showed 0.8262 mean coverage for tight crops, 0.9853 for fixed two-pixel padding, and 1.0000 for both fixed four-pixel and adaptive padding. These values should not be generalized beyond the present examples, but they are sufficient to justify the next controlled design choice: crop expansion is preferable to weakening segmentation when the immediate problem is edge truncation.

## Regression Testing as an Experimental Control

Software regression checks are now part of the experimental protocol. Before an ablation is interpreted, the upstream baseline can be verified with `tests/baseline_regression.py`. This is especially important for adaptive image processing, where changes in ROI, threshold population, morphology, interpolation, or coordinate translation can silently alter results. A valid downstream experiment should first demonstrate that the frozen upstream masks and boxes have not changed.
