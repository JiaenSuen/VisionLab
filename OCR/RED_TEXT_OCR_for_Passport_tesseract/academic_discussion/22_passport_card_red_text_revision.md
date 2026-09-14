# Passport/Document-Card Red-Text Revision

## Task Redefinition

The original pipeline was optimized around a fixed red alphanumeric field in a lower-right region. That formulation made a strong spatial prior useful: crop the ROI first, compute Lab local-red contrast, estimate an Otsu threshold inside that limited population, and then use connected components. The new task is materially different. The objective is now to detect **all red printed text on a passport-style or document-card image**, and red sequences may be horizontal or vertical. Consequently, a fixed ROI is no longer a valid general assumption.

## Why the Previous Successful Method Was Not Reused Unchanged

The earlier ROI-first Lab/Otsu method succeeded because threshold statistics were estimated from a small target region. Applying the same adaptive threshold over the full document changes the statistical population to include portraits, security textures, borders, stamps, and illumination variation. This can shift the threshold and generate unstable connected components. The revision therefore uses explicit HSV red hue as the default segmentation prior. This is not a claim that HSV is universally superior; it is a consequence of the revised problem containing a strong color cue across unknown spatial locations.

## Portrait False Positives

A full-image red detector initially produced many false components from skin and reddish photo regions. The important observation was that hue alone was insufficient. True printed red text in the test document showed substantially stronger saturation than most false portrait regions. The revised implementation therefore raises the pixel-level saturation requirement and adds a grouped-region median-saturation validation stage. This two-stage color check is a practical example of combining low-level pixel evidence with region-level statistics rather than relying on a single threshold.

## Vertical Text

The previous grouping algorithm assumed a horizontal reading direction. In the revised document, one number is printed vertically. Treating each vertical glyph independently causes many short horizontal boxes and prevents sequence OCR. The new grouping stage therefore builds vertical chains using horizontal overlap / x-center consistency plus a maximum vertical gap. Once a vertical chain explains a set of pixels, overlapping short horizontal proposals are suppressed. This avoids duplicate interpretations of the same glyphs.

## Orientation-Aware OCR

Tesseract expects a conventional reading direction. Vertical candidate regions are therefore rotated in both directions and evaluated after red-only background suppression. Experiments showed that grayscale crops retained too much portrait and security-pattern structure, whereas a red-only mask produced a recognizable vertical sequence. The OCR stage selects the stronger orientation candidate. Horizontal regions retain both grayscale and red-mask candidates because small low-resolution labels may benefit from gray anti-aliasing information.

## Current Limitations

The localization stage now captures the dominant horizontal and vertical red sequences reliably on the supplied example. The remaining errors mainly occur on very small, low-resolution red labels. Their bounding boxes can be correct while Tesseract still confuses individual characters. This distinction is important: further loosening segmentation is unlikely to solve a recognition-resolution problem. Future improvement should evaluate super-resolution, higher-resolution acquisition, language-specific OCR, or a trained sequence recognizer rather than repeatedly changing the red threshold.

## Reusable Image-Processing Lesson

The main methodological lesson is to re-evaluate priors whenever the task definition changes. A fixed ROI was previously an effective prior, but became a source of systematic misses when red text could occur anywhere. Likewise, horizontal-only grouping was appropriate for one field but invalid for vertically printed sequences. A robust workflow should therefore separate assumptions into spatial, chromatic, geometric, and recognition priors and test whether each assumption still holds before carrying a successful earlier pipeline into a new domain.
