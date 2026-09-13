# Failure Analysis and Root Causes

## 1. Failure Taxonomy

| Observed symptom | Immediate mechanism | Root cause | Corrective strategy |
|---|---|---|---|
| Thin strokes disappear | Foreground pixels removed | Anti-aliasing + hard threshold + destructive morphology | Use local contrast; avoid opening; crop original image for OCR |
| Too many tiny boxes | Fragmented foreground accepted | No/light closing and thresholds too permissive | 2x2 closing + stricter area/size filters |
| Large irrelevant red regions | Color filter accepts true red non-text | Color is not a semantic cue | Apply document ROI and geometric constraints |
| Successful prototype becomes noisy after modularization | Adaptive threshold changes | ROI applied after histogram-dependent Otsu | Crop ROI before feature extraction/threshold estimation |
| Two characters become one box | Foreground pixels touch | Low resolution, thresholding, morphology | Do not force split; use sequence grouping + sequence OCR |
| One character becomes several boxes | Weak/disconnected strokes | Threshold removes connecting pixels | Small closing; transitive grouping; original-image crop |
| OCR crop loses useful edge information | Binary mask used as recognition image | Localization and recognition roles conflated | Use mask for localization, original pixels for recognition |

## 2. The Most Important Regression: ROI Order

The largest unexpected failure occurred when the project was refactored into modules. The high-level operations appeared unchanged, but the ROI was treated as a late mask instead of the first operation. This caused Otsu thresholding to observe the entire document. Because Otsu estimates a threshold from the input histogram, unrelated structures changed the estimated separation between background-like and foreground-like responses. The later ROI could remove outside pixels spatially, but it could not reverse the threshold chosen from contaminated statistics. This explains why false boxes appeared *inside* the desired ROI even though the outside region was ultimately masked.

This case demonstrates a general debugging principle: when an operator estimates parameters from data, its **scope** is part of its definition. Moving a crop, normalization, or histogram calculation can change the algorithm even if every named function call remains present.

## 3. Secondary Regression: Removing the 2x2 Closing

The successful prototype used a small closing after thresholding and before connected-component labeling. During an attempt to preserve thin characters, this step was removed. The motivation was reasonable—morphology can destroy information—but the specific operation mattered. Closing is not equivalent to opening. In the observed images, the small closing repaired gaps created by thresholding and reduced the number of tiny disconnected pieces. Removing it therefore increased fragmentation and caused extra boxes. The correct conclusion is not “morphology is bad” but rather “morphology must match the topological error being corrected.”

## 4. Secondary Regression: Looser Geometric Filters

The prototype rejected components with area below 8 pixels, height below 5 pixels, or width below 2 pixels. A modular version used weaker defaults. This admitted fragments that the original code intentionally removed. Because connected components can create many small regions, even a small change in acceptance thresholds can noticeably increase false positives. Geometric filtering is therefore part of the baseline and must be kept constant during comparisons unless it is the explicit experimental variable.

## 5. Failure of the “One Component = One Character” Assumption

Even the successful preprocessing cannot guarantee perfect character boxes. Two touching letters can form one component, and one weak character can form several. This is a structural limitation of connected-component segmentation, not merely a parameter-tuning failure. The appropriate solution is to change the downstream formulation rather than indefinitely tuning morphology. Sequence-level recognition treats the crop as an ordered visual signal and allows the model/decoder to infer the number of characters.

## 6. What Should Be Logged in Future Failures

Every failed experiment should record the input image, exact commit/version, ROI coordinates, feature representation, threshold method and returned threshold value if available, morphology type/kernel/iterations, component filters, number of raw components, number of accepted boxes, number of grouped sequences, and representative debug images. Without these records, a visually better or worse result cannot be causally attributed to a specific design change.

## New Failure Mode: Treating a Tight Detector Box as the Recognition Crop

A detector box is mathematically defined by the accepted binary support. If anti-aliased character edges fall below the threshold, the box can appear to cut the visible glyph even when the thresholded localization is internally correct. Attempting to solve this by lowering the threshold changes connected-component topology. In the current larger example, relaxing the Otsu threshold increased foreground pixels from 953 to 1606 and changed the grouped output from three sequences to six. The root cause is therefore not simply "insufficient bbox size"; it is a mismatch between the purpose of localization geometry and the purpose of recognition input.

The current solution is to preserve the tight detector as an interpretable proposal and generate a second expanded crop for OCR. This resolves information retention without injecting uncertain edge pixels into connected-component construction.

## Historical Failure: Pretrained CRNN Produced Blank Predictions

The historical CAPTCHA checkpoint loaded and executed correctly but returned mostly CTC blank predictions on document crops. This established a useful distinction between software integration and recognition transfer. The source model was trained on different visual statistics and a restricted vocabulary, so the result was classified as domain shift rather than a localization failure. The CRNN has now been removed from the active runtime, while the full analysis is retained under `archive/previous_crnn_iteration/`.

## Current OCR Failure Mode: Over-Processing the Recognition Crop

Tesseract experiments show that recognition can degrade even when localization and grouping are unchanged. On the two current examples, raw color produced 0/2 exact matches and Otsu-binarized input produced 1/2, while grayscale 4x upsampling produced 2/2. The key cause is representation mismatch: raw crops retain distracting color/background variation, while a second hard threshold can reshape weak anti-aliased strokes. The current corrective strategy is therefore to keep the binary mask confined to localization and preserve grayscale edge evidence for recognition.
