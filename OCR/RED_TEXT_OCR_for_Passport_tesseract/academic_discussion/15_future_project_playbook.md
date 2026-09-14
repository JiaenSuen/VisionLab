# From Zero to a Reliable Image-Processing Pipeline: A Future-Project Playbook

## Step 1 — Define the Signal and the Nuisance Variables

Before writing filters, state what visually distinguishes the target from its local background and what can vary. For this project, useful signals were red chromaticity and known location; nuisance variables were low resolution, anti-aliasing, compression, background texture, and unrelated red graphics. This prevents the common mistake of choosing an algorithm first and then forcing the data to fit it.

## Step 2 — Build the Simplest Observable Baseline

Create a baseline that exposes every intermediate image. For color tasks, inspect RGB/HSV/Lab channels and histograms inside the actual ROI. Save the channel image, threshold mask, and overlays. The first baseline can be crude; its purpose is to reveal the failure mechanism. Do not add morphology, component filters, and OCR simultaneously.

## Step 3 — Classify the Failure Before Changing Parameters

Ask whether the error is **acquisition**, **representation**, **threshold**, **topology**, **geometry**, or **recognition**. If character edges are absent from the original image, preprocessing cannot recover them reliably. If they exist in the color feature but disappear after thresholding, change the decision rule. If the mask is correct but components fragment, use topology repair. If the crop is good but the predicted character is wrong, the recognizer is the bottleneck.

## Step 4 — Use Priors Before Complexity

If the task has a fixed region, known color, expected scale, or restricted alphabet, exploit those priors explicitly. In this project, ROI cropping solved a class of false positives more safely than stronger morphology. Priors reduce the problem size and allow simpler algorithms to work.

## Step 5 — Preserve Information Until It Is Safe to Discard

Avoid early irreversible operations. Binary thresholding and erosion throw away information. Use them for localization if needed, but keep the original image for later recognition. Add padding to proposal boxes. In general, let later stages discard irrelevant context rather than forcing early stages to reconstruct lost detail.

## Step 6 — Freeze a Successful Baseline

Once a configuration works, save the exact code, parameters, debug images, and numerical outputs. Refactoring is not allowed to change behavior until equivalence is verified. This protects against silent algorithm changes and creates a stable reference for ablation studies.

## Step 7 — Convert Tuning Into Ablation

Change one factor at a time and record metrics. Examples: ROI before/after Otsu, HSV/Lab, no morphology/closing/opening, or fixed/normalized gap threshold. If two changes are made together, their effects cannot be attributed. A short ablation table is more informative than many undocumented visual experiments.

## Step 8 — Choose Metrics That Match the Final Task

For OCR preprocessing, a clean-looking mask is not the final goal. Track missing-character coverage, fragmentation, merging, group completeness, and eventually CER. A detector with slightly more background can be preferable if it preserves all useful stroke information.

## Step 9 — Escalate to Deep Learning Only at the Right Boundary

Use learning where hand-crafted assumptions become fragile. Here, local red segmentation is still interpretable and effective, but exact character segmentation is not. Therefore, the appropriate boundary is to keep classical localization and delegate variable-length semantic decoding to a sequence recognizer. This is more principled than replacing the entire pipeline merely because some components touch.

## Step 10 — Preserve Negative Results

Record failed settings and why they failed. Negative results prevent repeated work and often reveal the strongest scientific insight. In this project, the failed modular version demonstrated that statistical scope and operation order are part of the algorithm. That insight is more general and valuable than the fact that one set of boxes looked better on one image.
