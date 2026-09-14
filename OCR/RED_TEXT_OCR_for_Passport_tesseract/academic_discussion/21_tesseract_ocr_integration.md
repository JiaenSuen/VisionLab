# Tesseract OCR Integration and End-to-End Annotation

## Motivation

The previous iteration demonstrated that the historical CAPTCHA CRNN could be modularized and executed, but its learned representation did not transfer to the document domain. The next research decision was therefore to remove the CRNN from the active runtime and introduce Tesseract as a non-trained OCR baseline. This change reduces experimental confounding: localization, grouping, and crop construction can now be evaluated independently from neural-network retraining. Tesseract also provides a practical reference point before a new learned recognizer is justified.

## Recognition Interface

The successful red-text detector is left unchanged. Connected components are grouped into text-line proposals and then expanded only at the recognition stage. Tesseract receives crops from the original document rather than from the binary segmentation mask. This distinction is important because weak anti-aliased stroke edges may disappear in the hard mask while remaining visible in the source image. The recognizer therefore benefits from information that is intentionally not required for connected-component topology.

## Preprocessing Experiment

Three Tesseract inputs were compared using the same groups and crop geometry: raw color, grayscale with 4x cubic upsampling, and grayscale followed by Otsu binarization. On the two controlled examples, raw color achieved 0/2 exact matches with mean CER 0.5666, grayscale upsampling achieved 2/2 exact matches with CER 0.0000, and Otsu achieved 1/2 exact matches with mean CER 0.0333. The current evidence therefore favors grayscale upsampling. The result is consistent with the observation that low-resolution character boundaries are encoded partly as intermediate intensity values; a second hard threshold can convert those uncertain but informative pixels into irreversible foreground/background decisions.

## Candidate Filtering

The ROI contains both the target alphanumeric field and short red labels. Because the active Tesseract configuration uses an English alphanumeric whitelist, sending small non-target red labels to OCR can create meaningless Latin-character substitutions. A simple width-to-height ratio filter is therefore applied before recognition. The default threshold of 3.0 retains the long sequence field in both controlled examples while suppressing the short label fragments. This is an application prior rather than a universal text detector rule, so the threshold is exposed as a command-line parameter and can be disabled.

## Full-Document Annotation

The end-to-end output is `ocr_result.png`. Thin rectangles are drawn on the full original document at the expanded OCR crop locations. Recognized strings are not rendered directly over document fields; instead, a result panel is appended below the source image. This design preserves the complete visual document while making the link between localized sequence and OCR output explicit. A CSV file records the group index, recognition status, text, aspect ratio, tight group geometry, and expanded crop geometry for reproducible analysis.

## Current Finding

The main finding of this iteration is not merely that Tesseract can read the two examples. It is that **recognition quality improved when the pipeline preserved gray-level edge evidence and decoupled OCR crop construction from binary localization**. This supports the broader project principle that localization and recognition should use different representations when their information requirements differ. A larger labeled dataset is still required before claiming general OCR robustness.
