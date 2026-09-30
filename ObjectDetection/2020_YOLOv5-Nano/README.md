# YOLOv5-Nano Reimplementation

A plain-PyTorch reproduction study of a **YOLOv5-Nano-style** object detector. The project recreates the main architectural ideas—Conv/C3 blocks, SPPF, FPN/PAN feature fusion, and three-scale detection heads—without depending on the Ultralytics package.

> This is an independent reimplementation for architecture study. It is not weight-compatible with Ultralytics YOLOv5 and does not reproduce the full official training recipe.

## Highlights

- YOLOv5-Nano-style backbone and neck implemented from scratch in PyTorch
- C3, SPPF, FPN/PAN, and three detection scales
- ~**1.75M parameters** with 20 classes
- Pascal VOC-style CSV + YOLO-format labels
- Mixed-precision training, checkpoint resume, NMS, and mAP@0.5 evaluation

## Architecture

```text
Input 320×320
   ↓
Conv + C3 backbone
   ↓
SPPF
   ↓
FPN top-down fusion
   ↓
PAN bottom-up fusion
   ↓
P3 / P4 / P5 detection heads
40×40   20×20   10×10
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a forward-pass check:

```bash
python quick_sanity.py
```

Training settings are defined near the top of `train.py`. The expected dataset layout is:

```text
data/
├── images/
├── labels/
├── mini_train.csv
└── mini_test.csv
```

Each CSV uses `img,label` columns, and each label file follows standard YOLO text format:

```text
class_id x_center y_center width height
```

Then run:

```bash
python train.py
```

## Reproduction Note

A reduced Pascal VOC-style pilot run is preserved under `results/pilot_voc_subset/`. It verifies the end-to-end pipeline, but should not be interpreted as an official YOLOv5 benchmark because the dataset and training objective differ from the original implementation.

## Reference

- Ultralytics YOLOv5: https://github.com/ultralytics/yolov5
