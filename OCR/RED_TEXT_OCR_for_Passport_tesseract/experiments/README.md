# Experiments

`run_ablation_study.py` reproduces the current controlled engineering experiments on the two repository examples. It evaluates four questions independently: recognition-crop padding, segmentation-threshold relaxation, fixed versus scale-aware grouping distance, and Tesseract input preprocessing (`raw`, `gray`, `otsu`). Results are written to `experiments/results/` as CSV files, annotated OCR images, and a generated Markdown summary.

Run from the repository root:

```bash
python experiments/run_ablation_study.py
```

The examples are a small case study rather than a benchmark. Their purpose is to make causal implementation changes reproducible before a larger labeled dataset is introduced.
