# Module Analysis: `main.py`

`main.py` is the orchestration layer for the complete classical-localization + Tesseract OCR pipeline. It reads the input image, applies the fixed ROI-aware red localization baseline, extracts connected-component proposals, groups nearby components into sequence candidates, constructs expanded recognition crops, runs Tesseract, writes structured prediction metadata, and exports the final annotated full-document image.

The orchestration intentionally exposes the major experimental parameters rather than hiding them in module globals. Localization thresholds and geometry filters remain separable from grouping distance, crop padding, and OCR preprocessing. This makes one-variable ablations possible and protects the successful detector from accidental changes caused by recognition experiments.

The main end-to-end artifact is `ocr_result.png`. Thin rectangles indicate the regions actually passed to recognition, while recognized strings are placed in a panel below the document so original fields are not obscured. `tesseract_predictions.csv` stores both tight group geometry and expanded crop geometry, making it possible to reconstruct which representation produced each OCR result.
