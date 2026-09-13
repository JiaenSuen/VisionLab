# Ablation Plan and Future Experiments

## Experiment 1 — ROI Before vs. After Otsu

**Question:** Was the modular regression primarily caused by changing the statistical scope of Otsu? Freeze every other parameter. In condition A, crop the ROI before Lab/local-contrast/Otsu. In condition B, compute local contrast and Otsu on the full image and apply the ROI afterward. Record the Otsu threshold value, raw component count, accepted component count, false-positive count, and character coverage. This is the highest-priority experiment because the mechanism is theoretically clear and directly connected to an observed failure.

**Expected result:** late ROI masking should increase threshold variability and false positives when the full document contains strong chromatic structures unrelated to the target field.

## Experiment 2 — HSV vs. Lab Local Contrast

**Question:** Does the local relative-color representation preserve weak edges better than fixed HSV ranges? Compare a tuned HSV baseline against Lab local contrast under original resolution, 0.75x/0.5x downsampling, and several JPEG quality levels. Measure pixel/box recall and character coverage. Avoid changing morphology simultaneously.

**Expected result:** HSV may perform competitively on clean, saturated text, while Lab local contrast should degrade more gracefully when edge pixels shift in absolute hue/saturation but remain locally redder than the background.

## Experiment 3 — Morphology Ablation

Compare no morphology, 2x2 closing, 3x3 closing, 2x2 opening, and the original 3x3 open+close. Measure fragmentation rate, merge rate, and character coverage. This experiment should demonstrate why the smallest closing was beneficial and quantify the trade-off between reconnecting strokes and accidentally joining neighboring characters.

## Experiment 4 — Scale-Normalized Component Filtering

Replace fixed thresholds with values relative to ROI height or median candidate height. Compare stability across rescaled images. The hypothesis is that fixed `area >= 8`, `width >= 2`, `height >= 5` works because the current images have a consistent scale, not because those numbers are intrinsically optimal. A scale-aware rule should generalize better.

## Experiment 5 — Grouping Threshold Sensitivity

Sweep `max_gap` and vertical-overlap criteria. Record group purity, completeness, and number of output groups. Also test a normalized gap such as `gap <= alpha * median_character_height`. This should reduce dependence on input resolution. Visualize failure cases where a chain of components causes transitive over-merging; graph connectivity can propagate through intermediate boxes even if the endpoints are far apart.

## Experiment 6 — Character CNN vs. Sequence OCR

Create two downstream baselines. **Character baseline:** padded single-component crops -> small CNN trained on alphanumeric data. **Sequence baseline:** merged crops -> Tesseract first, followed later by a learned sequence recognizer under the same crop protocol. Compare end-to-end CER and exact-string accuracy rather than only classifier accuracy. This experiment directly tests the project hypothesis that sequence recognition is more tolerant of imperfect segmentation.

## Experiment 7 — Degradation Robustness

Generate controlled variants using downsampling, blur, JPEG compression, brightness shifts, and mild color shifts. Plot localization recall and OCR CER against degradation severity. This turns the informal observation “pixel quality may be the problem” into a measurable robustness curve and reveals whether the dominant bottleneck is acquisition quality, segmentation, or recognition.
