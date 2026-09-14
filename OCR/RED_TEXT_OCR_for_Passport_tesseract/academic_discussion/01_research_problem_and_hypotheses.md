# Research Problem and Hypotheses

## 1. Problem Definition

The practical objective is to extract a small red alphanumeric field from a low-resolution document image and prepare it for robust OCR. The task is narrower than generic scene-text detection because the target text has two useful priors: it is red and appears in a known or approximately known region. At the same time, the image is difficult because characters are only a few pixels wide, anti-aliasing blends stroke boundaries with the background, JPEG or resizing artifacts perturb color values, and neighboring characters can either fragment or merge after thresholding. A direct OCR model could be applied to the entire image, but the research question here is whether a lightweight, interpretable preprocessing stage can substantially reduce the search space before recognition.

The pipeline therefore separates three questions that should not be conflated: **where is red text likely to exist, what connected structures are produced by segmentation, and how should those structures be packaged for recognition?** This separation is academically useful because a failure at localization can be diagnosed independently from a recognition error. It also supports low-compute applications in which classical image processing is used as a front end and a small learned model is reserved for semantic recognition.

## 2. Research Questions

The current study is organized around five research questions. **RQ1:** Is hard red-color thresholding sufficient for low-resolution red text, or is a local chromatic-contrast representation more stable? **RQ2:** How strongly does the placement of the ROI relative to adaptive threshold estimation affect false-positive components? **RQ3:** Can lightweight morphology improve component continuity without destroying thin strokes? **RQ4:** Under what conditions does connected-component analysis remain useful despite the fact that one component is not always one character? **RQ5:** Can proximity-based grouping convert imperfect component detections into sequence images that are more tolerant of segmentation errors and therefore better suited to sequence-level OCR than per-character classification?

These questions deliberately focus on causes rather than only final accuracy. The aim is to identify which preprocessing assumptions are responsible for success or failure so that future projects can choose methods from image statistics rather than by uncontrolled trial-and-error.

## 3. Working Hypotheses

**H1 — Local chromatic contrast.** A Lab `a`-channel local-contrast representation should be more robust than a fixed HSV red range when the target strokes are anti-aliased or weakly saturated, because the decision is based on being *redder than the local neighborhood* rather than satisfying one absolute hue/saturation interval.

**H2 — ROI-before-threshold.** Cropping the target ROI before Otsu thresholding should reduce false components because Otsu is histogram-dependent. Irrelevant structures outside the ROI change the class distribution and can therefore move the threshold even if they are masked out afterward.

**H3 — Light closing, not aggressive opening.** A very small morphological closing can reconnect tiny gaps in strokes, whereas opening can erase narrow foreground structures because its erosion stage acts before reconstruction.

**H4 — Components as evidence, not characters.** Connected components should be interpreted as localization primitives rather than guaranteed characters. Sequence-level grouping should therefore be more robust than forcing every component into an independent character classifier.

**H5 — Recognition from original pixels.** Detection masks should localize text, while OCR crops should be taken from the original image with padding. This preserves edge pixels and texture that may have been discarded by thresholding.
