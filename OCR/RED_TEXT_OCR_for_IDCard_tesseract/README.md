# Red-Text OCR Research Pipeline

A lightweight and interpretable OCR pipeline for localizing low-resolution red alphanumeric text in semi-structured document images. The current version deliberately separates **localization**, **sequence grouping**, **recognition crop construction**, and **OCR**. Tesseract is used as the active recognizer; the earlier CRNN implementation has been removed from the executable project and retained only as historical discussion material.

## Current Pipeline

```text
Input document
  -> target ROI crop
  -> CIE Lab a-channel
  -> Gaussian local chromatic baseline
  -> positive local-red contrast
  -> Otsu thresholding inside the ROI
  -> 2x2 morphological closing
  -> 8-connected components
  -> conservative geometric filtering
  -> scale-aware transitive component grouping
  -> expanded crop from the original image
  -> grayscale 4x upsampling
  -> Tesseract OCR (PSM 7)
  -> bounding box + recognized text rendered on the full document
```

The detector and OCR crop are intentionally different representations. The detector remains tight because stable connected-component topology is more important than visually covering every anti-aliased edge pixel. The recognition crop is expanded afterward so OCR receives the complete character context without weakening segmentation.

## Project Structure

```text
red_text_ocr_pipeline_tesseract/
├── main.py
├── modules/
│   ├── red_components.py
│   ├── sequence_groups.py
│   └── tesseract_ocr.py
├── experiments/
│   ├── run_ablation_study.py
│   └── results/
├── tests/
│   └── baseline_regression.py
├── academic_discussion/
├── examples/
├── outputs/
├── requirements.txt
└── README.md
```

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Tesseract is an external executable and must also be installed on the system. After installation, either add it to `PATH` or provide the executable explicitly:

```bash
python main.py examples/1.jpg --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

The default OCR language is English because the target field in the current experiments is alphanumeric. Additional Tesseract language data can be selected with `--ocr-lang` when needed.

## Basic Usage

Run the default example:

```bash
python main.py
```

Run another image:

```bash
python main.py path/to/image.png
```

A typical command using the current research defaults is:

```bash
python main.py path/to/image.png \
  --roi 0.63 0.70 1.0 1.0 \
  --adaptive-gap \
  --padding-mode adaptive \
  --ocr-preprocess gray \
  --ocr-scale 4 \
  --ocr-min-aspect 3.0
```

`--ocr-min-aspect 3.0` suppresses short red labels that are unlikely to be the target alphanumeric sequence. Set it to `0` if every grouped red region should be sent to Tesseract.

## Output

Each run exports:

```text
outputs/
├── character_mask.png
├── detect_mask.png
├── component_boxes.png
├── group_boxes.png
├── ocr_crop_boxes.png
├── ocr_inputs/
│   └── text_group_*.png
├── ocr_preprocessed/
│   └── text_group_*_prepared.png
├── tesseract_predictions.csv
└── ocr_result.png
```

`ocr_result.png` is the main end-to-end output. It preserves the complete document image, draws a thin box around each recognized sequence, and appends a result panel below the document so OCR text does not cover original fields.

## Why the OCR Box Is Larger Than the Detection Box

The connected-component box is a localization estimate derived from a binary mask. At low resolution, anti-aliasing and compression can cause valid stroke-edge pixels to fall below the segmentation threshold. Expanding the detector itself by lowering the threshold can increase false foreground connections and change the connected-component topology. The current pipeline therefore keeps the detector unchanged and expands only the crop passed to OCR.

The controlled padding experiment on the two repository examples found:

| Crop strategy | Mean estimated weak-stroke coverage | Minimum coverage |
|---|---:|---:|
| Tight box | 0.8262 | 0.5797 |
| Fixed +2 px | 0.9853 | 0.9565 |
| Fixed +4 px | 1.0000 | 1.0000 |
| Adaptive | 1.0000 | 1.0000 |

This is the main reason the current implementation uses adaptive recognition padding rather than modifying the successful segmentation baseline.

## Tesseract Baseline Experiment

Run the controlled experiment suite:

```bash
python experiments/run_ablation_study.py
```

On the two current examples, the Tesseract preprocessing ablation produced:

| Tesseract input | Exact matches | Mean CER |
|---|---:|---:|
| Raw color crop | 0/2 | 0.5666 |
| Grayscale + 4x upsampling | 2/2 | 0.0000 |
| Otsu binary + 4x upsampling | 1/2 | 0.0333 |

The current best default is therefore **grayscale upsampling without a second hard threshold**. This retains anti-aliased intensity information that Tesseract can use, whereas binary thresholding can reshape weak strokes. These results are a small engineering case study and should not be interpreted as a large-scale benchmark.

## Reproducibility Guard

The earlier project suffered a regression when modularization changed the order of ROI cropping and Otsu threshold estimation. The current detector is protected by:

```bash
python tests/baseline_regression.py
```

The test compares the modular implementation against the successful historical `RedCharacterDetect` reference logic and verifies identical masks and component boxes on the repository examples.

## Main Findings

1. **ROI-before-Otsu changes the threshold statistics and is essential to the successful baseline.**
2. **Local Lab red contrast is more robust than strict absolute-red thresholding for the observed images.**
3. **A small 2x2 closing repairs fragmentation; aggressive morphology can erase thin strokes.**
4. **Connected components should be treated as localization proposals, not guaranteed characters.**
5. **Detection geometry and recognition coverage should be optimized separately.**
6. **Grouping distance should scale with character size rather than remain a fixed pixel constant.**
7. **For the present examples, grayscale 4x upsampling gives Tesseract a substantially better input than raw color or hard binarization.**

## Academic Discussion

Start with [`academic_discussion/README.md`](academic_discussion/README.md). The discussion archive documents successful and failed red-segmentation variants, the ROI/Otsu regression, connected-component limitations, crop-padding ablations, scale-aware grouping, the previous CRNN transfer attempt, and the current Tesseract baseline. Historical CRNN material is kept under `academic_discussion/archive/` for research traceability but is no longer part of the runtime code.

## Current Research Direction

The current pipeline should now be treated as a reproducible classical-localization + OCR baseline. The next useful step is to build a labeled set of document sequences and evaluate robustness across blur, compression, scale, illumination, and font variation. Recognition should be reported with Character Error Rate and exact-string accuracy while localization/grouping remains evaluated separately. Only after this baseline is quantified on a larger dataset is it useful to reintroduce a learned sequence recognizer for a fair comparison.
