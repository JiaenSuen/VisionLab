# E-ConvNeXt + Light Transformer Image Captioning

PyTorch implementation of an image-captioning model that combines an **E-ConvNeXt visual encoder**, **multi-scale visual features**, **learned-query resampling**, and an **autoregressive Transformer decoder**. The training pipeline supports both **Flickr8k** and **Flickr30k**.

## Highlights

- E-ConvNeXt encoder with `mini`, `tiny`, and `small` variants.
- Multi-scale visual features extracted from the C4 and C5 stages.
- Learned-query resampling before language decoding.
- Transformer decoder with causal self-attention and visual cross-attention.
- Greedy and beam-search caption generation.
- Flickr8k and Flickr30k annotation parsing.
- Training and evaluation utilities for BLEU, METEOR, ROUGE-L, and CIDEr-style metrics.

## Architecture

```text
Input Image
    │
    ▼
E-ConvNeXt Encoder
    │
    ├── C4 Feature Map
    └── C5 Feature Map
            │
            ▼
1×1 Projection + Adaptive Pooling
            │
            ▼
Multi-Scale Visual Tokens
            │
            ▼
Learned-Query Visual Resampler
  Cross-Attention + Self-Attention
            │
            ▼
Resampled Visual Tokens
            │
            ▼
Autoregressive Transformer Decoder
            │
            ▼
Generated Caption
```

The visual resampler converts the multi-scale CNN representation into a fixed set of learned visual queries before autoregressive decoding. This reduces the number of visual tokens processed by decoder cross-attention while retaining information from multiple encoder stages.

## Results and Discussion

The E-ConvNeXt–Light Transformer was compared with a conventional **ResNet–Attention LSTM** baseline to study the trade-off between model size and caption quality.

| Model | Parameters | BLEU-1 | BLEU-4 | METEOR | CIDEr |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet–Attention LSTM | 14.055 M | 0.5735 | **0.1872** | **0.3572** | **0.4091** |
| E-ConvNeXt–Light Transformer | **9.429 M** | **0.5746** | 0.1717 | 0.3304 | 0.3522 |

The proposed model reduces the parameter count by approximately **32.9%** while maintaining essentially the same BLEU-1 performance. BLEU-1 slightly increases from 0.5735 to 0.5746, suggesting that the model preserves basic object and scene vocabulary despite using fewer parameters.

The difference becomes more visible on higher-order and semantic metrics. BLEU-4 decreases from 0.1872 to 0.1717, METEOR from 0.3572 to 0.3304, and CIDEr from 0.4091 to 0.3522. The result indicates that the reduced model retains word-level recognition more effectively than longer phrase structure and reference-level semantic agreement. The CIDEr gap is the largest among the reported metrics, suggesting weaker consistency with the descriptive patterns shared across multiple reference captions.

Overall, the experiment shows a clear **efficiency–quality trade-off**: the E-ConvNeXt–Light Transformer substantially reduces model size and preserves unigram-level caption quality, while higher-order language generation remains weaker than the larger recurrent baseline. This makes the architecture useful as a lightweight vision-to-text representation module and provides a practical baseline for further work on pretrained language decoders, knowledge distillation, and parameter-efficient multimodal adaptation.

> The table reports the comparative experiment associated with this project. Metric values should only be compared under the same dataset split and evaluation implementation.

## Supported Datasets

The data loader supports common Flickr8k and Flickr30k annotation formats, including CSV, Flickr8k `image#idx<TAB>caption`, and Flickr30k pipe-delimited annotations.

Example configurations are provided in:

```text
configs/flickr8k.json
configs/flickr30k.json
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train on Flickr8k:

```bash
python train.py --config configs/flickr8k.json
```

Train on Flickr30k:

```bash
python train.py --config configs/flickr30k.json
```

Generate a caption from a trained checkpoint:

```bash
python infer.py \
  --checkpoint runs/flickr8k/checkpoints/best.pt \
  --image path/to/example.jpg \
  --beam-size 3
```

Run the parser and forward/backward sanity check:

```bash
python quick_sanity.py
```

## Main Files

```text
econvnext.py       E-ConvNeXt encoder variants
model.py           visual projection, resampler, and Transformer decoder
dataset.py         Flickr8k/Flickr30k parsing and vocabulary utilities
loss.py            caption training loss and token metrics
train.py           training and evaluation pipeline
infer.py           single-image caption generation
configs/           Flickr8k and Flickr30k experiment configurations
results/           concise comparison summary
```

## License

Released under the MIT License. See `LICENSE`.
