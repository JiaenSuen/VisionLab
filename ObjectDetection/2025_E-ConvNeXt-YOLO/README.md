# E-ConvNeXt + YOLOv10-Style Object Detector

A PyTorch research implementation that combines an **E-ConvNeXt backbone** with a lightweight **YOLOv10-style PAFPN and dual assignment head** for object detection experiments on Pascal VOC / YOLO-format annotations.

It includes the model, loss, data pipeline, training/evaluation code, single-image inference, and a compact record of the reference VOC0712 experiment.


> This is an independent research implementation inspired by YOLOv10-style one-to-many / one-to-one training. It is not the official YOLOv10 implementation.

## Highlights

- E-ConvNeXt `mini`, `tiny`, and `small` backbone variants.
- Lightweight PAFPN with depthwise separable convolution blocks.
- Objectness-free decoupled detection heads.
- Dual training branches: one-to-many and one-to-one assignment.
- CIoU-based box regression and quality-aware classification loss.
- AMP, EMA, AdamW, cosine learning-rate decay, gradient accumulation, and channels-last training.
- VOC-style `mAP@0.5` evaluation with per-class AP reporting.
- Standalone sanity check and single-image inference script.

## Reference Experiment

The included experiment logs correspond to a from-scratch VOC0712 run using the `mini` model.

| Item | Value |
| --- | ---: |
| Dataset | VOC0712 |
| Training images | 16,551 |
| Test images | 4,952 |
| Number of classes | 20 |
| Input resolution | 448 × 448 |
| Parameters | 5.66 M |
| Epochs | 100 |
| Batch size | 8 |
| Gradient accumulation | 2 |
| Best full test mAP@0.5 | **57.79%** |
| Best epoch | **80** |
| Final full test mAP@0.5 | 57.13% |

The full training history and per-class AP values are stored in [`results/voc0712_mini/`](results/voc0712_mini/).

## Project Structure

```text
EConvNeXt-YOLOv10-VOC/
├── README.md
├── requirements.txt
├── .gitignore
├── __init__.py
├── yolo.py                 # E-ConvNeXt backbone, PAFPN, and dual detection head
├── loss.py                 # One-to-many + one-to-one detection losses
├── dataset.py              # VOC / YOLO-format dataset and augmentations
├── eval.py                 # Decoding, NMS, and VOC mAP evaluation
├── train.py                # Training entry point
├── predict.py              # Single-image inference
├── quick_sanity.py         # Forward/loss/backward smoke test
└── results/
    └── voc0712_mini/       # Compact experiment logs; no large checkpoints
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

For CUDA training, install the PyTorch build appropriate for your CUDA environment before or together with the remaining requirements.

## Quick Sanity Check

Run this before preparing a dataset:

```bash
python quick_sanity.py
```

A successful run performs model construction, forward propagation, loss computation, and backward propagation.

## Dataset Format

The default directory layout is:

```text
data/
├── images/
│   ├── 2007_000033.jpg
│   └── ...
├── labels/
│   ├── 2007_000033.txt
│   └── ...
├── train.csv
└── test.csv
```

Each CSV contains two columns:

```csv
img,label
2007_000033.jpg,2007_000033.txt
```

Each label file follows normalized YOLO `class x_center y_center width height` format:

```text
14 0.512 0.487 0.238 0.411
6 0.271 0.318 0.164 0.207
```

Coordinates must be normalized to `[0, 1]`.

## Training

A reproduction-oriented command for the included `mini` experiment is:

```bash
python train.py \
  --variant mini \
  --model-name EConvNeXtMini_YOLOv10n_VOC \
  --image-size 448 \
  --batch-size 8 \
  --accum-steps 2 \
  --epochs 100 \
  --train-csv data/train.csv \
  --test-csv data/test.csv \
  --img-dir data/images/ \
  --label-dir data/labels/
```

For the `small` variant on limited VRAM, a practical starting point is:

```bash
python train.py \
  --variant small \
  --batch-size 4 \
  --accum-steps 4
```

All configuration values in `train.py` are exposed through the command line. Use:

```bash
python train.py --help
```

### Optional Backbone Weights

A classifier-style E-ConvNeXt checkpoint can be loaded into matching backbone layers:

```bash
python train.py \
  --variant mini \
  --backbone-weights path/to/econvnext_mini_classifier.pth
```

Classifier heads and incompatible parameter shapes are skipped automatically.

### Resume Training

```bash
python train.py \
  --resume record/EConvNeXtMini_YOLOv10n_VOC/last.pth.tar
```

## Inference

Training checkpoints contain enough configuration metadata for single-image inference:

```bash
python predict.py \
  --weights weights/best.pth.tar \
  --image path/to/image.jpg \
  --output outputs/prediction.jpg \
  --score-threshold 0.25
```

EMA weights are used automatically when available. Add `--raw-model` to use the non-EMA model state.

## Outputs

Training runs are written to:

```text
record/<model-name>/
```

Typical outputs include:

```text
best.pth.tar
last.pth.tar
training_metrics.csv
test_metrics.csv
per_class_ap.csv
dataset_record.csv
experiment_report.txt
train_snapshot.py
```

Large checkpoints are intentionally excluded from version control. For a public repository, publish pretrained weights through **GitHub Releases** or **Git LFS** rather than committing them directly into the Git history.

## Model Design

The detector is composed of three stages:

1. **E-ConvNeXt backbone** — hierarchical feature extraction with depthwise `7×7` convolution, pointwise MLP-style projections, ESE attention, residual connections, and CSP-style stage partitioning.
2. **Lightweight PAFPN** — multi-scale top-down and bottom-up feature aggregation using lightweight C2f-style blocks.
3. **Dual objectness-free head** — separate one-to-many and one-to-one branches. Training supervises both branches, while evaluation and inference use the one-to-one branch by default.

This implementation is intentionally compact and experiment-oriented. It is suitable for architecture ablations, training studies, lightweight detector prototyping, and educational reproduction.

## Reproducibility Notes

- The reference run was trained from scratch; no backbone checkpoint was loaded.
- The saved experiment used `seed=123`, AdamW, EMA, AMP, cosine decay, and `drop_path_rate=0.05`.
- Quick evaluations use a subset of test batches and are not directly comparable to the full test-set mAP values.
- The reported **57.79% mAP@0.5** is the best full-test result recorded in the supplied experiment logs.

## License

No open-source license is included in this package. Before publishing the project for unrestricted reuse, add a license that matches your intended distribution policy, such as MIT or Apache-2.0.
