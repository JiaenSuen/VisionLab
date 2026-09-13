# Sequence Grouping and the Transition to OCR

## 1. Motivation

The initial downstream idea was to enlarge each connected-component box and classify it as one character. This is useful as a simple baseline, but it depends on a fragile assumption: one connected component must equal one semantic character. The experiments already show both violations of this assumption. A weak character can fragment into several components, while adjacent characters can touch and become a single component. Once either error occurs, a per-character classifier receives the wrong semantic unit and cannot reconstruct the information that was discarded by segmentation.

The pipeline therefore changes its objective from **perfect character segmentation** to **reliable text-sequence localization**. Neighboring components are merged into a wider line-like proposal. The recognition backend then receives a complete sequence crop and decides how many characters are present. This design is compatible with Tesseract today and with learned sequence recognizers in later work.

## 2. Current Grouping Algorithm

Each accepted connected-component box is represented as a node. Two nodes are connected when their horizontal free gap is below an effective threshold and they are vertically compatible according to overlap or center alignment. Union-find constructs transitive connected sets, so A can merge with C through B even when A and C are not directly adjacent. Each connected set is converted into one enclosing rectangle.

The effective horizontal threshold can scale with median component height. This is important because an eight-pixel gap has different meaning at different image resolutions. On the current larger example, fixed eight-pixel grouping produced five groups, while ten pixels and the adaptive rule produced three coherent groups without changing the smaller example.

## 3. Detection Box Versus Recognition Crop

The merged group rectangle remains a localization diagnostic. It is not used directly as the final OCR crop. A second crop is expanded around the group and sampled from the original image. This distinction addresses a recurring low-resolution problem: anti-aliased edge pixels can be visible to a human but absent from the hard binary mask. Lowering the segmentation threshold to capture them changes connected-component topology, whereas crop expansion preserves the stable detector and simply gives OCR more context.

The padding ablation supports this separation. Tight boxes retained 82.62% mean estimated faint-stroke support with a minimum of 57.97%, while four-pixel and adaptive crops retained 100% of the current support proxy.

## 4. Tesseract as the Current Sequence Recognizer

The active backend is now Tesseract rather than the previous CRNN experiment. Tesseract receives the expanded sequence crop, not individual character boxes. The default configuration uses PSM 7 because each proposal is treated as one text line, an alphanumeric whitelist, and 4x cubic upsampling. A preprocessing ablation showed that grayscale upsampling produced exact recognition on both controlled examples, while raw color failed on both and an additional Otsu threshold failed on one.

This result reinforces the same information-preservation principle used elsewhere in the project. Binary localization is useful for geometry, but the recognizer benefits from gray-level structure. A second hard threshold can reshape or delete weak strokes that still carry recognition evidence.

## 5. Remaining Risks

Grouping can over-merge nearby fields or under-merge widely spaced characters. Small punctuation can be rejected before grouping. The current aspect-ratio filter also encodes an application prior: the target is a long alphanumeric field, so short red labels are skipped by default. Future evaluation should therefore report group completeness, group purity, CER, and exact-string accuracy over a larger labeled dataset. The current two-image success is evidence that the interface works, not proof of general OCR robustness.
