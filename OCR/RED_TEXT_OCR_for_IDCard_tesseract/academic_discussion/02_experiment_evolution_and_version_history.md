# Experiment Evolution and Version History

## Stage 1 — Fixed HSV Red Thresholding with Opening and Closing

The first implementation converted the input image to HSV and used two hue intervals around red, followed by a 3x3 morphological opening and closing. This design was intuitive: HSV explicitly separates hue from brightness, two intervals are required because red wraps around the hue axis, opening removes isolated noise, and closing fills small gaps. In practice, however, the result damaged many character strokes while retaining unrelated red graphics. The reason was not simply a “bad threshold.” Thin text at low resolution contains mixed edge pixels with lower saturation, shifted hue, and background contamination. Hard thresholding rejected some of those pixels, and the subsequent opening further eroded already-thin strokes. At the same time, large red objects such as flags or stamps were valid red regions and therefore could not be rejected using color alone.

**Finding:** color identity is not equivalent to text identity. A color mask can constrain the search space, but geometry or spatial priors are still required. Aggressive morphology is especially risky when the target structures are only a few pixels thick.

## Stage 2 — Reduced Morphological Destruction

The next revision removed morphological opening and reduced the structural element to a small 2x2 closing. This change was motivated by information preservation. Opening is erosion followed by dilation; once a one- or two-pixel stroke is removed by erosion, later dilation cannot reliably reconstruct its original topology. Closing, by contrast, can bridge small holes or discontinuities without first shrinking the foreground. The revision reduced stroke destruction, but it did not solve the deeper problem that hard HSV thresholds still treated anti-aliased edge pixels inconsistently and could not distinguish text from other red structures.

**Finding:** when OCR is downstream, segmentation should be conservative. It is usually preferable to keep questionable foreground pixels and reject candidates later using ROI/geometry than to destroy character structure early.

## Stage 3 — Spatial ROI as a Strong Prior

A normalized lower-right ROI was introduced to focus only on the region where the target alphanumeric field was expected. This was a major conceptual improvement because it used document layout rather than forcing color segmentation to solve an impossible semantic problem. The ROI suppressed large irrelevant red regions elsewhere in the document without manipulating the characters themselves. At this stage, the system still used absolute color filtering, but the experiment established an important principle: when the document template is fixed or semi-structured, spatial priors are often more reliable and less destructive than stronger morphological filtering.

**Finding:** exploit known task structure before increasing algorithmic complexity. A reliable ROI can remove entire classes of false positives at almost zero computational cost.

## Stage 4 — Lab Local Red Contrast and Otsu Thresholding

The most successful prototype changed the feature representation. The ROI was cropped first, converted to CIE Lab, and the `a` channel was used as a red-green chromatic axis. A Gaussian-blurred local baseline was subtracted from the `a` channel, producing a response that emphasized pixels locally redder than their surroundings. Otsu thresholding then binarized this local-contrast image automatically. A 2x2 closing was applied only to the detection mask, after which 8-connected components were filtered by minimum area, width, and height.

This version produced substantially cleaner boxes. Its strength came from the interaction of several design choices rather than one parameter: a restricted ROI stabilized the image statistics, local contrast emphasized relative color differences, Otsu adapted to the current ROI, light closing repaired tiny discontinuities, and geometric filtering removed fragments.

## Stage 5 — Modularization Regression

A later modular rewrite unexpectedly produced many false boxes even though the high-level pipeline still appeared to be “Lab + Otsu + connected components.” The regression exposed a reproducibility problem: the implementation no longer preserved the exact computational order. In one version, local contrast and Otsu were evaluated over the whole document before applying the ROI; the 2x2 closing was omitted; and minimum component thresholds were loosened. Because Otsu depends on the distribution of its input values, changing the region used for threshold estimation changed the decision boundary. Removing closing increased fragmentation, and weaker geometric thresholds admitted fragments that the original prototype rejected.

**Finding:** preprocessing order is part of the algorithm. Two pipelines with the same named operations are not equivalent when an adaptive/statistical operator observes a different input distribution.

## Stage 6 — Corrected Modular Baseline

The corrected project restores the successful prototype exactly: crop ROI first, compute Lab local red contrast, apply Otsu inside that ROI, perform 2x2 closing on a detection copy, run 8-connected components, and apply the original minimum area/width/height constraints. The code is now modular without changing the baseline behavior. This stage establishes a stable point from which future experiments should change only one factor at a time.

## Stage 7 — Sequence-Oriented OCR Preparation

The final current stage recognizes that connected components are not reliable semantic character boundaries. Nearby components on approximately the same line are represented as a graph and transitively grouped according to horizontal gap and vertical compatibility. Each group is converted to an enclosing rectangle, expanded, cropped from the original image, and resized to a fixed height while preserving aspect ratio. These crops are designed for later CRNN/CTC or general sequence OCR. The key idea is to move segmentation responsibility away from fragile per-character heuristics and toward a recognizer that can infer a variable-length label sequence from one image region.

## Iteration: Modular CRNN Integration and Recognition-Crop Decoupling

The latest iteration intentionally froze the successful red-localization baseline and moved experimentation downstream. A regression script now reproduces the original `RedCharacterDetect` logic and confirms byte-identical raw masks, identical post-closing masks, and identical accepted component boxes on both repository examples. This step was introduced because an earlier modularization had accidentally changed the statistical population used by Otsu thresholding and therefore invalidated direct comparison.

Three controlled changes were then evaluated. First, OCR crop padding was separated from detection geometry. The tight localization box is still used to describe the thresholded component, but the crop passed to recognition is expanded. Second, the fixed eight-pixel sequence-grouping threshold was augmented with a scale-aware rule based on median component height. Third, the historical CAPTCHA CRNN was refactored into a reusable package and connected to the grouped-crop output. The CRNN integration executed correctly but showed negative transfer, with predictions dominated by the CTC blank class. This result is recorded as a model-domain mismatch rather than a preprocessing regression.


## Iteration: Tesseract Baseline and Full-Document Annotation

The historical CRNN was removed from the active runtime after its negative-transfer behavior was established. Tesseract was introduced as a training-free sequence OCR baseline so recognition could be evaluated without adding a new target-domain learning problem. The grouped text proposal and adaptive recognition crop are now passed to Tesseract using PSM 7 and an alphanumeric whitelist. Three input representations were compared while keeping localization constant: raw color, grayscale 4x upsampling, and Otsu-binarized 4x input. On the two controlled examples, grayscale achieved 2/2 exact matches with CER 0.0000, Otsu achieved 1/2, and raw color achieved 0/2. The final output now includes the full document, OCR localization boxes, and a separate result panel so recognized text does not obscure source fields.
