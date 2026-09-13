# Research and Design Decision Log

## D1 — Use a Hand-Crafted Front End Before Deep Learning

**Decision:** Begin with color/geometry-based localization rather than training a detector immediately. **Reason:** the target field has strong red-color and layout priors, the dataset is initially small, and interpretability is valuable for understanding failure causes. **Status:** retained.

## D2 — Stop Treating HSV as the Final Red Representation

**Decision:** Replace the main HSV hard-threshold experiment with Lab `a`-channel local contrast. **Reason:** HSV removed anti-aliased/weak edge pixels and could not distinguish locally meaningful red strokes from other red structures. **Status:** Lab local contrast is the current baseline; HSV remains an ablation baseline.

## D3 — Apply ROI Before Adaptive Thresholding

**Decision:** Crop the known target region before Lab conversion/local contrast/Otsu. **Reason:** Otsu is histogram-dependent, so the ROI must define the data distribution used to estimate the threshold. **Status:** critical baseline requirement.

## D4 — Use 2x2 Closing, Not Aggressive Opening

**Decision:** Apply a minimal 2x2 closing to the detection mask. **Reason:** the dominant observed topological error is small stroke gaps, while opening destroys thin characters. **Status:** retained; should be revalidated across scales.

## D5 — Keep Connected Components but Change Their Interpretation

**Decision:** Use 8-connected components as proposal regions, not guaranteed semantic characters. **Reason:** the method is fast and interpretable but cannot resolve touching/fragmented glyphs reliably. **Status:** retained as localization evidence.

## D6 — Separate Localization Mask from Recognition Image

**Decision:** Crop OCR input from the original image with padding rather than feeding the binary detection mask by default. **Reason:** the binary mask discards weak edges, gradients, and texture that may help recognition. **Status:** retained.

## D7 — Add Sequence Grouping Before CRNN/OCR

**Decision:** Transitively merge horizontally nearby components with vertical compatibility, export N sequence crops, and delay semantic segmentation to a sequence recognizer. **Reason:** increases tolerance to one-to-many and many-to-one component/character mappings. **Status:** implemented as preprocessing; model evaluation pending.

## D8 — Treat Refactoring as an Experimentally Sensitive Operation

**Decision:** require stage-by-stage equivalence testing whenever a working preprocessing pipeline is modularized. **Reason:** a prior refactor changed ROI order, morphology, and filters and therefore changed behavior despite similar high-level logic. **Status:** adopted as a reproducibility rule.

## Decision: Separate Detection Boxes from Recognition Crop Boxes

**Decision:** Keep connected-component/group geometry tight, and expand only the image region passed to OCR.

**Evidence:** Tight crops retained 82.62% mean estimated faint-stroke support; four-pixel/adaptive crops retained 100% in the current two-image study. Lowering the segmentation threshold to capture more edge pixels destabilized topology on the larger example, increasing sequence groups from three to six.

**Consequence:** `component_boxes.png` and `group_boxes.png` remain localization diagnostics. `ocr_crop_boxes.png` visualizes the actual expanded regions used for recognition.

## Decision: Introduce Scale-Aware Grouping

**Decision:** Use `max(8 px, median component height x ratio)` as the effective horizontal gap when adaptive grouping is enabled.

**Evidence:** Fixed eight pixels produced five groups on the larger example; ten pixels/adaptive grouping produced three while preserving the smaller example.

## Historical Decision (Superseded): Integrate but Do Not Trust the Historical CRNN as Final OCR

**Decision:** Preserve the old checkpoint as a transfer baseline and software-integration test.

**Evidence:** The checkpoint loads and runs, but target-domain crops are almost entirely CTC-blank. The source model was trained on CAPTCHA appearance and supports only `0-9A-Z`.

**Consequence:** This decision was later superseded by removing the CRNN from the active runtime and adopting Tesseract as the baseline.

## Decision: Remove CRNN from the Active Runtime and Use Tesseract as the OCR Baseline

**Decision:** Remove the historical CRNN package and checkpoint from the executable project. Preserve its negative-transfer analysis only in the academic archive, and use Tesseract as the active sequence recognizer.

**Evidence:** The historical model executed correctly but was dominated by CTC blank outputs because its CAPTCHA training distribution and vocabulary did not match the document domain. Tesseract, using the same grouped recognition crops, provides an immediately testable baseline without retraining.

**Consequence:** Recognition experiments now isolate image-processing and crop-design effects from neural-network domain adaptation. A learned recognizer can be reintroduced later as a fair comparison after a labeled target-domain dataset exists.

## Decision: Use Grayscale 4x Upsampling Before Tesseract

**Decision:** Use grayscale cubic upsampling by a factor of four as the default Tesseract representation, without a second hard threshold.

**Evidence:** On the two current examples, grayscale achieved 2/2 exact matches with mean CER 0.0000. Raw color achieved 0/2 exact matches, and Otsu-binarized input achieved 1/2.

**Consequence:** The OCR stage preserves intermediate edge intensities instead of converting every pixel to a second binary decision. Otsu and raw modes remain available for controlled ablations.
