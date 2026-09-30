"""Fast parser and model smoke tests for the public project package."""

import tempfile
from pathlib import Path

import torch

from dataset import Vocabulary, parse_caption_file
from loss import LabelSmoothingCrossEntropy
from model import build_caption_model


def test_caption_parsers():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        flickr8k = root / "captions.txt"
        flickr8k.write_text(
            "image,caption\nexample.jpg,A person walks near a river.\n",
            encoding="utf-8",
        )
        flickr30k = root / "results.csv"
        flickr30k.write_text(
            "image_name|comment_number|comment\nexample.jpg|0|A person walks near a river.\n",
            encoding="utf-8",
        )
        assert parse_caption_file(str(flickr8k))[0]["image"] == "example.jpg"
        assert parse_caption_file(str(flickr30k))[0]["caption"].startswith("A person")


def test_forward_backward():
    config = {
        "encoder_variant": "mini",
        "d_model": 64,
        "visual_pool_size": 4,
        "visual_queries": 8,
        "resampler_layers": 1,
        "decoder_layers": 1,
        "nhead": 4,
        "ffn_dim": 128,
        "dropout": 0.1,
        "max_len": 12,
    }
    vocab = Vocabulary(min_freq=1, max_size=64).build(
        ["a person walks near a river", "a dog runs through grass"]
    )
    model = build_caption_model(config, len(vocab), vocab.pad_idx, vocab.bos_idx, vocab.eos_idx)
    model.train()

    images = torch.randn(2, 3, 128, 128)
    input_ids = torch.tensor(
        [
            [vocab.bos_idx, vocab.stoi.get("a", vocab.unk_idx), vocab.stoi.get("person", vocab.unk_idx)],
            [vocab.bos_idx, vocab.stoi.get("a", vocab.unk_idx), vocab.stoi.get("dog", vocab.unk_idx)],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [vocab.stoi.get("a", vocab.unk_idx), vocab.stoi.get("person", vocab.unk_idx), vocab.eos_idx],
            [vocab.stoi.get("a", vocab.unk_idx), vocab.stoi.get("dog", vocab.unk_idx), vocab.eos_idx],
        ],
        dtype=torch.long,
    )
    logits = model(images, input_ids)
    loss = LabelSmoothingCrossEntropy(0.1, ignore_index=vocab.pad_idx)(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"Sanity check passed. logits={tuple(logits.shape)}, loss={loss.item():.4f}")


if __name__ == "__main__":
    test_caption_parsers()
    test_forward_backward()
