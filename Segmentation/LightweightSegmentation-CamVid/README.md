# LightweightSegmentation

A lightweight semantic segmentation project for fine-tuning `LR-ASPP + MobileNetV3-Large` on the 11-class CamVid dataset.

## 1. Installation

Python 3.10–3.12 and a CUDA-enabled version of PyTorch are recommended:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Adjust the official PyTorch installation command according to your CUDA and PyTorch environment. PyTorch does not need to be reinstalled if a compatible version is already available.

## 2. Dataset Preparation

Organize the CamVid dataset using the following directory structure. Label masks may either have the same filename as their corresponding images or use the common CamVid `_L` suffix.

For example, `0001TP_009210.png` may correspond to `0001TP_009210_L.png`.

```text
data/CamVid/
├── train/          # Training images
├── train_labels/   # Training masks
├── val/            # Validation images
├── val_labels/     # Validation masks
├── test/           # Test images
└── test_labels/    # Test masks
```

The dataset loader supports both RGB palette masks and grayscale class-ID masks with pixel values ranging from `0` to `10`. Unknown colors and invalid class IDs are converted to `ignore_index=255`.

Inspect the dataset and mask encoding before training:

```bash
python -m datasets.camvid --root data/CamVid --split train
```

## 3. Training

Start the full training process with:

```bash
python train.py --config configs/lraspp.yaml
```

For a quick smoke test:

```bash
python train.py --config configs/lraspp.yaml --epochs 1 --max-train-batches 2 --max-val-batches 2
```

## 4. Evaluation, Prediction, and Benchmarking

Evaluate the trained model and save its segmentation predictions:

```bash
python evaluate.py --config configs/lraspp.yaml \
  --checkpoint outputs/lraspp_experiment/checkpoints/best.pt \
  --save-predictions
```

Run semantic segmentation on a single image:

```bash
python predict.py --config configs/lraspp.yaml \
  --checkpoint outputs/lraspp_experiment/checkpoints/best.pt \
  --image example.png
```

Measure inference latency, FPS, parameter count, and peak memory usage:

```bash
python benchmark.py --config configs/lraspp.yaml \
  --checkpoint outputs/lraspp_experiment/checkpoints/best.pt
```

## 5. First-Run Checklist

1. Verify that `python -m datasets.camvid` reports only mask IDs from `0` to `10`, with an optional `255` ignore label.
2. Confirm that the model can overfit a small number of training batches before starting full training.
3. Never resize segmentation masks using bilinear interpolation. This project uses nearest-neighbor interpolation for all masks.
4. When comparing different models, keep the input resolution, dataset split, loss function, and benchmarking protocol fixed.

## 6. Experimental Setup

The following table records the primary experimental configuration. Replace the placeholder values if the final configuration differs.

| Setting | Configuration |
|---|---|
| Dataset | CamVid |
| Number of classes | 11 |
| Training samples | — |
| Validation samples | — |
| Test samples | — |
| Input resolution | 512 × 512 |
| Model | LR-ASPP + MobileNetV3-Large |
| Pretrained weights | — |
| Loss function | Cross-Entropy + Dice Loss |
| Optimizer | AdamW |
| Initial learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Batch size | 8 |
| Number of epochs | 50 |
| Learning-rate scheduler | Polynomial decay |
| Mixed-precision training | Enabled |
| Training device | — |
| Random seed | 42 |

## 7. Overall Segmentation Results

Report the results from the test set using the checkpoint with the highest validation mIoU.

| Model | Pretraining | Loss | Input Size | Pixel Accuracy (%) | Mean Accuracy (%) | mIoU (%) |
|---|---|---|---:|---:|---:|---:|
| LR-ASPP + MobileNetV3-Large | — | Cross-Entropy | 512 × 512 | — | — | — |
| LR-ASPP + MobileNetV3-Large | — | Cross-Entropy + Dice | 512 × 512 | — | — | — |

The best result should be highlighted in bold after all experiments are complete.

## 8. Per-Class IoU

| Model | Sky | Building | Pole | Road | Pavement | Tree | Sign | Fence | Car | Pedestrian | Bicyclist | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR-ASPP (CE) | — | — | — | — | — | — | — | — | — | — | — | — |
| LR-ASPP (CE + Dice) | — | — | — | — | — | — | — | — | — | — | — | — |

All values should be reported as percentages. In addition to the overall mIoU, particular attention should be paid to small and thin object classes such as `Pole`, `Sign`, `Pedestrian`, and `Bicyclist`.

## 9. Ablation Study

This experiment evaluates whether Dice Loss improves minority-class segmentation without changing the model architecture or inference cost.

| Experiment | Cross-Entropy | Dice Loss | Data Augmentation | mIoU (%) | Small-Class mIoU (%) |
|---|:---:|:---:|:---:|---:|---:|
| A: Baseline | ✓ |  | ✓ | — | — |
| B: Combined loss | ✓ | ✓ | ✓ | — | — |
| C: No augmentation | ✓ | ✓ |  | — | — |

`Small-Class mIoU` may be calculated as the mean IoU of `Pole`, `Sign`, `Pedestrian`, and `Bicyclist`.

## 10. Computational Efficiency

All latency measurements should use batch size 1 and a fixed input resolution. Run at least 50 warm-up iterations followed by 200 measured iterations. GPU timing should be synchronized before and after measurement.

| Model | Precision | Parameters (M) | Model Size (MB) | Peak Memory (MB) | Latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|
| LR-ASPP + MobileNetV3-Large | FP32 | — | — | — | — | — |
| LR-ASPP + MobileNetV3-Large | FP16 | — | — | — | — | — |

### Hardware Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU memory | — |
| CPU | — |
| System memory | — |
| Operating system | Windows 11 |
| CUDA version | — |
| PyTorch version | — |
| Torchvision version | — |

## 11. Qualitative Results

Example segmentation results can be placed in the following format:

| Input Image | Ground Truth | Prediction |
|---|---|---|
| To be added | To be added | To be added |
| To be added | To be added | To be added |

The qualitative analysis should discuss:

- Segmentation quality around object boundaries
- Confusion between `Road` and `Pavement`
- Recognition of small objects such as pedestrians and traffic signs
- Failure cases caused by shadows, low contrast, or object occlusion
- Over-smoothed predictions around thin structures

## 12. Experimental Notes

Record any important observations from the experiments:

- **Best validation epoch:** —
- **Best validation mIoU:** —
- **Final test mIoU:** —
- **Average training time per epoch:** —
- **Total training time:** —
- **Observed failure cases:** —
- **Additional remarks:** —