# Bounding-Box Tightness versus Recognition-Crop Padding

## Research Question

The latest qualitative inspection raised an important ambiguity: several connected-component boxes visually touched or cut through the edge of the rendered glyph. Two competing solutions were considered. The first was to modify red segmentation so that the detected foreground more accurately included faint anti-aliased edge pixels. The second was to preserve the successful detector and enlarge only the crop passed to the recognizer. These alternatives are not equivalent. The first changes the topology from which connected components are computed; the second changes only information retained for recognition.

## Experimental Design

A small ablation was implemented in `experiments/run_ablation_study.py`. For each grouped text proposal, the experiment estimated a faint-stroke support region by lowering the local-red contrast threshold to 50% of the Otsu value but restricting those pixels to a narrow dilation neighborhood around the accepted hard mask. This is not ground-truth character segmentation; it is a diagnostic proxy for weak red edge evidence that the hard threshold may omit. Four crop policies were compared without changing the detector: zero padding, fixed two-pixel padding, fixed four-pixel padding, and height-adaptive padding.

Across the six groups in the two repository examples, the tight box retained 82.62% mean support and fell to 57.97% on the worst group. Two pixels improved mean support to 98.53% but still missed weak evidence in some cases. Four pixels and the adaptive policy reached 100% coverage on this proxy for every current group. The result explains why a visually tight bounding box can be unsuitable as an OCR crop even when it is a valid localization box.

## Why Not Simply Lower the Segmentation Threshold?

The threshold-ablation results provide a direct answer. On the larger example, the Otsu baseline produced 953 foreground pixels, 15 accepted components, and 3 sequence groups. Reducing the threshold below Otsu increased foreground to 1606 pixels and 16 accepted components, but the grouped result deteriorated to 6 sequence groups. In other words, additional foreground changed component connectivity and introduced unstable structure rather than merely extending the correct strokes. This is a key image-processing lesson: maximizing foreground recall is not identical to maximizing useful segmentation topology.

## Decision

The detector remains unchanged. Tight component and group boxes are preserved as localization evidence, while a separate recognition-crop box is expanded. The default OCR crop now uses height-adaptive padding with a minimum four-pixel margin. This design is preferable because it is information-preserving but does not feed uncertain low-contrast pixels back into connected-component construction. If a future labeled dataset demonstrates systematic clipping beyond this margin, the detector can be revisited with explicit coverage ground truth rather than by visual trial-and-error.
