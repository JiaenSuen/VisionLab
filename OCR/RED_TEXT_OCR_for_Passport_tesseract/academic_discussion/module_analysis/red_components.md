# Module Analysis: `modules/red_components.py`

## Responsibility

`red_components.py` implements the localization baseline in two explicit stages: `red_filter()` produces an ROI-sized binary local-red mask and the ROI offset; `find_component_boxes()` performs a small closing, connected-component labeling, geometric filtering, and conversion back to original-image coordinates. The separation is intentional because it prevents detection logic from becoming entangled with visualization or sequence recognition.

## `red_filter()` — Algorithmic Rationale

The function crops the normalized ROI before any histogram-dependent processing. This is a critical reproducibility property, not merely an optimization. The crop is converted to Lab, the `a` channel is extracted, a Gaussian-smoothed baseline is computed, and `cv2.subtract` keeps positive local red contrast. Otsu then binarizes the response. The output is deliberately the **raw** pre-morphology mask so that the effect of morphology can be inspected independently.

The main research parameters are ROI coordinates and Gaussian sigma. The current sigma of 3 pixels defines the spatial scale of the local baseline. If characters or input resolution change substantially, this value should be revalidated or normalized by estimated character size.

## `find_component_boxes()` — Algorithmic Rationale

This function applies a 2x2 closing because the observed dominant error is small threshold-induced gaps. It then uses 8-connectivity to label foreground structures and rejects regions below minimum area/width/height thresholds. The returned boxes are proposals, not semantic labels. The detection mask is returned alongside boxes because it is an essential debugging artifact.

## Known Limitations

Fixed pixel thresholds are resolution-dependent. Small punctuation may be rejected. Closing can merge adjacent characters if the gap is already very small. Otsu assumes a histogram that can be meaningfully separated into classes; difficult backgrounds can violate this assumption. These limitations should be measured through the ablation protocol rather than hidden by additional heuristics.
