# Reproducibility and Debugging Protocol

## 1. Freeze a Baseline Before Refactoring

Before modularizing or optimizing a working prototype, save its exact input image, parameter values, output masks, component boxes, and counts. The corrected baseline for this project is: normalized ROI `(0.63, 0.70, 1.0, 1.0)` -> Lab `a` channel -> Gaussian blur with sigma 3 -> positive local difference -> Otsu -> 2x2 closing -> 8-connected components -> `area >= 8`, `width >= 2`, `height >= 5`. Any refactor should first be verified to reproduce these outputs before additional improvements are introduced.

## 2. Use Stage-by-Stage Equivalence Tests

When code structure changes, compare intermediate arrays rather than only final images. Useful checks include exact equality of ROI coordinates, feature-map dimensions, threshold masks, morphology masks, box lists, and group lists. If floating-point processing prevents exact equality, compare summary statistics and a numerical tolerance. This protocol would have detected the earlier modular regression immediately because the raw mask already differed before component labeling.

## 3. Debug at the First Divergent Stage

When a new version fails, do not tune downstream parameters immediately. Find the earliest stage where it diverges from the baseline:

1. Is the same ROI cropped?
2. Is the same channel/feature map produced?
3. Is the same thresholding population used?
4. Does morphology alter topology as expected?
5. Are component filters identical?
6. Are grouping rules the only remaining difference?

Fixing the first divergence usually prevents wasted tuning later.

## 4. Record Adaptive Parameters

For automatic algorithms, log their estimated values. Otsu returns a threshold; record it. If a future method estimates illumination, scale, or normalization statistics, record those as well. Sudden changes in an automatically estimated parameter often explain downstream regressions more directly than visual inspection.

## 5. Keep Detection and Recognition Artifacts Separate

Store at least three classes of output: diagnostic masks, localization visualizations, and recognition crops. Do not overwrite them with one image. This makes it possible to answer whether OCR failed because the target was not localized, because the crop was truncated, or because the recognizer misclassified a visually complete crop.

## 6. One-Variable Experiment Rule

Unless the purpose is an integrated system comparison, change only one causal factor relative to the frozen baseline. A commit or experiment label should state that factor explicitly: `roi_order_ablation`, `morphology_2x2_close`, or `gap_normalization`. If multiple changes are unavoidable, add intermediate runs so their effects remain attributable.

## 7. Minimum Experiment Record

For every run, save: experiment ID, date, code version, input set, preprocessing parameters, threshold values, number of raw/accepted components, number of groups, metrics, representative success/failure images, and a short interpretation. The interpretation should distinguish **observation** (“false components increased”) from **hypothesis** (“full-image Otsu changed the threshold due to unrelated chromatic structures”).
