# Image-Processing Knowledge Base for This Project

## 1. Color Space Is a Representation Choice, Not a Guarantee

RGB, HSV, and Lab emphasize different structures. HSV is convenient for absolute color-range rules, but low-saturation or anti-aliased pixels may not remain inside a clean hue interval. Lab is useful here because its `a` channel approximately isolates red-green chromatic variation from luminance. The practical rule is to inspect channel distributions in the actual ROI before deciding which color space is “best.” A representation should be chosen because it makes foreground/background separation simpler, not because it is commonly used for color filtering.

## 2. Local Contrast Can Be More Stable Than Absolute Thresholds

If the background changes slowly while the target differs locally, subtracting a smoothed local baseline can suppress global variation. The current Gaussian subtraction is a simple form of high-pass filtering on the chromatic channel. Similar reasoning appears in background subtraction, unsharp masking, local normalization, and top-hat morphology. The important design question is whether the signal is defined by an absolute value or by a difference relative to its neighborhood.

## 3. Adaptive Thresholds Depend on Their Statistical Scope

Otsu, histogram equalization, CLAHE, and many automatic normalization procedures estimate parameters from data. Their output can change when unrelated pixels are included. Always define which spatial population should determine the statistics. In this project, the ROI must be applied before Otsu because the threshold should model the target field, not the entire document.

## 4. Morphology Must Match the Error Topology

Opening removes small foreground objects and narrow protrusions because it erodes first. Closing fills small gaps/holes and connects nearby foreground because it dilates first. Neither operation is universally “denoising.” If the dominant error is fragmented characters, opening is often counterproductive; a small closing may help. Kernel size and iteration count must be interpreted relative to character stroke width and image scale.

## 5. Connectivity Is Not Semantics

Connected-component labeling answers whether foreground pixels are topologically connected under 4- or 8-neighborhood rules. It does not know where one character ends and another begins. Therefore, “one component = one character” should always be treated as a testable assumption. Touching characters and fragmented characters are expected failure modes, not anomalies.

## 6. Preserve the Rich Image for Recognition

Binary masks are lossy. They remove intensity gradients, weak edges, color differences, and anti-aliasing. A strong design pattern is to use a simple representation for localization and then crop the recognition input from the original image. Padding should compensate for under-segmentation. This principle appears broadly in detect-then-recognize systems.

## 7. Resolution Changes the Meaning of Every Pixel Parameter

A 2-pixel gap, 8-pixel area threshold, or 4-pixel padding has no invariant physical meaning. If the input scale changes, fixed pixel parameters change their relative strength. For more general deployment, normalize quantities by ROI height, estimated character height, or median component statistics. Before changing algorithms, verify whether a failure is simply caused by a different image scale.

## 8. Debug Intermediate Images, Not Only the Final Boxes

A minimum debug set should include: original ROI, selected color channel, local-contrast response, raw threshold mask, post-morphology mask, all connected components, filtered components, grouped boxes, and final OCR crops. Comparing these stages localizes the first point at which information is lost. This is much more efficient than changing several parameters based only on the final visualization.

## 9. Separate Detection Metrics from Recognition Metrics

A visually clean mask is not necessarily the mask that produces the best OCR. Detection should be evaluated using coverage/precision or bounding-box metrics; recognition should be evaluated using character error rate or exact-string accuracy. A slightly noisy localization may be preferable if it preserves all characters and the recognizer can ignore the noise.

## 10. Change One Causal Factor at a Time

The modularization regression demonstrates why uncontrolled tuning is misleading. ROI order, morphology, and geometric filters changed simultaneously, so the source of the degradation was initially unclear. Future experiments should use ablation tables in which only one factor changes from a frozen baseline. This transforms trial-and-error into hypothesis testing.
