# Pilot reproduction run

This folder preserves a development run used to verify the training and evaluation pipeline on a reduced Pascal VOC-style subset.

- Train images: 3,000
- Test images: 500
- Input size: 320 × 320
- Epochs: 100
- Best test mAP@0.5: 0.007224 at epoch 50

These numbers are **not** comparable to official YOLOv5 benchmarks. The run used a reduced dataset and a simplified anchor-based training objective, and is retained only as a reproduction/debugging record.
