"""Generate an image caption from a trained E-ConvNeXt + Transformer checkpoint."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from dataset import CaptionEvalTransform, Vocabulary
from model import build_caption_model
from utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Caption one image with a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or another training checkpoint.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--vocab", default=None, help="Optional vocab.json for legacy checkpoints without embedded vocabulary.")
    parser.add_argument("--beam-size", type=int, default=3, help="Beam size. Use 1 for greedy decoding.")
    parser.add_argument("--max-len", type=int, default=None, help="Optional generation length override.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def clean_state_dict(state_dict):
    prefix = "_orig_mod."
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def main():
    args = parse_args()
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a model config.")

    if "vocab" in checkpoint:
        vocab = Vocabulary.from_dict(checkpoint["vocab"])
    elif args.vocab:
        vocab = Vocabulary.load(args.vocab)
    else:
        raise ValueError("Vocabulary is not embedded in this checkpoint. Pass --vocab path/to/vocab.json.")

    model = build_caption_model(config, len(vocab), vocab.pad_idx, vocab.bos_idx, vocab.eos_idx).to(device)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)
    model.eval()

    image = Image.open(Path(args.image)).convert("RGB")
    transform = CaptionEvalTransform(config.get("image_size", 224), resize_short_side=256)
    tensor = transform(image).unsqueeze(0).to(device)
    max_len = int(args.max_len or config.get("max_len", 36))

    with torch.inference_mode():
        if args.beam_size > 1:
            sequence = model.generate_beam(tensor, max_len=max_len, beam_size=args.beam_size)[0]
        else:
            sequence = model.generate_greedy(tensor, max_len=max_len)[0]

    print(vocab.decode(sequence.detach().cpu().tolist()))


if __name__ == "__main__":
    main()
