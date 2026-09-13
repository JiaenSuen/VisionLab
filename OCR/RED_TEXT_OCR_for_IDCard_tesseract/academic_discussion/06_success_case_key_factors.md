# Why the Successful Version Worked

## 1. Success Was Produced by an Interaction of Choices

The strongest version should not be reduced to a single phrase such as “Lab works better.” Its performance came from an ordered combination of priors and operators. First, the known document layout restricted the search to a lower-right ROI. Second, the Lab `a` channel provided a chromatic axis aligned with red-green variation. Third, subtracting a Gaussian local baseline emphasized *relative* redness rather than requiring a globally fixed red value. Fourth, Otsu selected a data-dependent threshold from the already restricted response distribution. Fifth, a 2x2 closing repaired minor discontinuities before connectivity was analyzed. Finally, minimum area/width/height filters removed residual fragments. Each stage addressed a different failure mode, and their order preserved the assumptions of the next stage.

The key academic interpretation is that this is a **structured prior pipeline**: spatial prior -> chromatic prior -> local contrast -> adaptive binarization -> topology repair -> geometric filtering. It succeeds because the task is constrained enough for these priors to be informative.

## 2. The ROI Stabilized Statistics, Not Only Space

The ROI had two roles. The obvious role was spatial suppression of unrelated red structures. The less obvious and more important role was statistical stabilization. Because Otsu depends on the histogram of its input, cropping before threshold estimation prevented unrelated content from changing the threshold. This explains why the modular regression produced different detections even after a late ROI mask. In future experiments, ROI placement should therefore be considered an ablation variable rather than a cosmetic implementation detail.

## 3. Local Contrast Matched the Visual Cue Actually Present

The human observation that the characters remained visible because their boundaries differed in tone from the immediate background motivated the local-contrast design. The Gaussian subtraction formalized that observation. This is a good example of turning qualitative inspection into an explicit signal-processing hypothesis. Instead of asking “what exact red value should I threshold?” the method asks “where is the red-green chromatic response locally higher than the surrounding paper?” That reformulation made the method less dependent on absolute color calibration.

## 4. Light Closing Corrected the Right Topological Error

The successful version did not use morphology for generic denoising. It used a very small closing for a specific purpose: reconnecting gaps caused by thresholding. This distinction matters. Opening would first erode the already-thin foreground and therefore worsen the dominant error. The successful operation was chosen according to the observed topology, not because morphology is generally beneficial.

## 5. Conservative Geometry Reduced Fragment Noise

The final component filters rejected tiny regions that were unlikely to be useful characters. Their values were small enough to preserve legitimate alphanumeric structures in the current scale but strict enough to remove many isolated artifacts. When these thresholds were loosened in the modular version, false detections increased. The result demonstrates that post-segmentation geometry can be an effective low-cost prior, but it should be normalized or revalidated if the image resolution changes.

## 6. The Most Transferable Finding

The most transferable finding is not a specific threshold or kernel size. It is the experimental structure: **preserve information early, constrain the domain with strong priors, inspect intermediate representations, and keep adaptive operators inside the region whose statistics they are intended to model.** This principle is applicable to color segmentation, defect inspection, document analysis, and many other classical computer-vision tasks.
