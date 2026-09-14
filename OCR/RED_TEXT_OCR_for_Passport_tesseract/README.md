# Red-Text OCR Pipeline for Passport / Document Cards

This project detects and recognizes **red printed text anywhere on a passport-style or document-card image**, including both horizontal and vertical text. The runtime is intentionally simple: red-color segmentation -> connected components -> orientation-aware grouping -> enlarged OCR crop -> Tesseract -> one annotated output image.

## Current Pipeline

```text
Input passport/document card
  -> full-image HSV red segmentation
  -> light 2x2 closing
  -> connected-component proposals
  -> reject very large red graphics
  -> horizontal + vertical text grouping
  -> suppress horizontal fragments already explained by a vertical line
  -> region-level red-saturation validation
  -> padded crop from original image
  -> horizontal: gray/red-mask OCR candidates
  -> vertical: rotate both directions + red-mask OCR
  -> choose best Tesseract candidate
  -> annotate all recognized red text on one full-image result
```

The previous fixed lower-right ROI is no longer the default. The task is now **document-wide red-text recognition**, so the default ROI is the full image `(0, 0, 1, 1)`.

## Quick Start

Install Python packages:

```bash
pip install -r requirements.txt
```

Install Tesseract OCR separately and add it to `PATH`, or provide its executable path.

Run the included passport/document-card example:

```bash
python main.py
```

Run another image:

```bash
python main.py path/to/card.png
```

Windows example when Tesseract is not in PATH:

```bash
python main.py path/to/card.png --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Main Output

The important file is:

```text
outputs/ocr_result.png
```

It contains the complete input card, bounding boxes for recognized red text, and one result panel listing all OCR strings. Horizontal regions are shown in green; vertical regions are shown in magenta.

Additional debug files:

```text
outputs/
├── red_mask.png
├── component_boxes.png
├── group_boxes.png
├── ocr_preprocessed/
├── tesseract_predictions.csv
└── ocr_result.png
```

## Important Changes in This Revision

### 1. Fixed ROI removed

The earlier implementation assumed one red alphanumeric field in the lower-right corner of an ID-card example. That assumption is incompatible with passport/document cards where red text can appear at the top, left edge, or elsewhere. The default is now full-image processing.

### 2. HSV absolute-red segmentation is now the default

The earlier Lab local-contrast method was effective for one fixed low-resolution field, but full-document Otsu statistics are unstable and can create many false components. For the revised task, explicit red hue is a stronger task prior. The old Lab mode is still available with `--red-mode lab` for comparison.

### 3. Vertical red text is supported

Connected components are grouped in both directions. Vertical lines are formed from components with similar x-position and small vertical gaps. Before OCR, vertical crops are rotated both clockwise and counter-clockwise; Tesseract results are compared and the stronger candidate is retained.

### 4. Reddish skin/background suppression added

Passport photographs contain skin tones that can enter a loose red mask. The pipeline therefore uses two levels of color validation: a stronger HSV saturation threshold during pixel segmentation and a second median-red-saturation check at the grouped-region level. This keeps strongly printed red text while removing most weak reddish portrait fragments.

### 5. OCR preprocessing became orientation-aware

For horizontal red text, the pipeline can compare grayscale and red-mask representations. For vertical text, a red-only mask is used after rotation because suppressing the portrait/security background substantially improves sequence OCR.

## Default Example

The runtime example is now:

```text
examples/passport_card_sample.bmp
```

The earlier Taiwan ID-card examples are retained only under:

```text
examples/archive_legacy_id/
```

They are no longer the primary task definition.

## Current Engineering Finding

The key lesson from this task change is that **the most successful preprocessing prior must match the revised task definition**. ROI-first Lab/Otsu was a good solution for a known fixed field. Once the target became all red text across a complete document, that same prior became restrictive. Full-image HSV red segmentation, orientation-aware grouping, and background suppression are more appropriate for the new problem.

The current default example recognizes the major horizontal red number and the vertical red number, while also attempting other red fields. Low-resolution small red labels remain the main OCR error source and should be treated as recognition-resolution limitations rather than localization failures.

## Academic Notes

See `academic_discussion/22_passport_card_red_text_revision.md` for the task-change analysis, design decisions, failure causes, and the reason vertical OCR required a different preprocessing path.
