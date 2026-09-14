# Red Segmentation: Detailed Analysis

## 1. Why Hard HSV Thresholding Was an Attractive First Choice

HSV thresholding is a reasonable first baseline when the desired foreground has a known color. Hue approximately represents color category, saturation measures chromatic strength, and value captures brightness. Red requires two hue ranges because the OpenCV hue axis wraps around at its endpoints. This approach is fast, deterministic, interpretable, and requires no training data. It is therefore appropriate as an initial experiment. Its limitation is that printed “red” text does not occupy a single stable point in HSV space. Camera processing, JPEG compression, resampling, illumination, anti-aliasing, and background color all perturb individual pixel values. A stroke center can remain strongly red while its boundary becomes less saturated or shifts in hue. Consequently, strict thresholds often produce a skeleton-like interior rather than the complete printed character.

The critical methodological lesson is that a threshold must be chosen according to the *distribution created by the imaging process*, not the nominal ink color. When each stroke is only a few pixels wide, losing one boundary pixel can remove a large fraction of the character geometry.

## 2. Why Local Lab `a` Contrast Worked Better

CIE Lab separates luminance (`L`) from two chromatic axes. The `a` channel increases from green toward red, making it useful when the target differs from the background primarily in red-green chromaticity. The successful detector does not classify a pixel as red by comparing it to a fixed absolute interval. Instead, it computes a local baseline using Gaussian blur and evaluates the positive difference

\[
D(x,y)=\max\{a(x,y)-G_\sigma(a)(x,y),0\},
\]

where `a(x,y)` is the Lab `a` value and `G_sigma` is a Gaussian-smoothed neighborhood. This acts as a simple local chromatic high-pass operation. A character pixel can be useful even if it is not globally “very red,” provided that it is redder than nearby paper/background pixels. That relative criterion matches the visual observation that the target characters remain visible because of local tonal contrast even when their absolute color varies.

This design should not be described as universally superior to HSV. It is superior under the current assumptions: locally smooth background, target red text, restricted ROI, and relatively small illumination/color shifts.

## 3. Why Otsu Must See the Correct Region

Otsu thresholding selects a threshold from the observed histogram by maximizing between-class separability (equivalently minimizing within-class variance under the classical formulation). It is therefore not a fixed operator whose output depends only on each pixel independently. If the histogram contains unrelated structures from a flag, portrait, stamp, or background ornamentation, the class distribution changes and so can the selected threshold. Applying the ROI only *after* Otsu cannot undo this statistical contamination because the threshold has already been estimated from the wrong population.

The successful pipeline performs ROI cropping first. This is equivalent to defining the population of pixels that the threshold is allowed to model. The regression observed during modularization is therefore mechanistically explainable rather than mysterious: the code changed the statistical sample on which Otsu operated. This is a central lesson for all adaptive preprocessing methods—CLAHE, adaptive thresholding, histogram normalization, and automatic exposure-like operations can also depend strongly on the spatial region over which their statistics are computed.

## 4. Edge Loss Is Not Only a Resolution Problem

Low spatial resolution is one source of lost edges, but several mechanisms interact. **Anti-aliasing** intentionally mixes foreground and background values at boundaries. **JPEG compression** introduces block and ringing artifacts. **Resizing** changes pixel values through interpolation. **Color subsampling** can reduce chromatic spatial resolution. **Hard thresholding** then converts gradual transitions into binary decisions, and **morphological erosion/opening** can remove any remaining narrow structures. Therefore, increasing resolution can help, but changing the algorithmic order and avoiding destructive operations are equally important.

A useful diagnostic is to inspect the original crop, color-feature map, raw binary mask, morphology result, and component boxes separately. If the edge is already weak in the original image, the problem is acquisition. If it exists in the feature map but disappears after thresholding, the decision boundary is too strict. If it survives thresholding but disappears after morphology, the structural element or operation is responsible.

## 5. Recognition Should Use Original Pixels

The binary segmentation mask is best treated as a localization instrument. It discards intensity, color gradients, anti-aliasing, and weak edge pixels; those cues may still help a learned recognizer. For this reason, after a component or group box is found, the project expands the box by a small padding margin and crops from the original BGR image rather than from the binary mask. This separation is one of the most transferable findings of the project: **a detection representation and a recognition representation do not need to be identical.** A sparse or approximate mask can be sufficient to locate text while the recognition model receives a richer crop.
