from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from models import build_model
from project_utils import extract_logits, load_config
from utils.checkpoint import load_model_checkpoint
from utils.transforms import ImageOnlyTransform
from utils.visualization import colorize_mask


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lraspp.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config["dataset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config["model"]["name"], int(dataset_config["num_classes"]), pretrained=False
    )
    load_model_checkpoint(args.checkpoint, model, device)
    model.to(device).eval()

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    tensor = ImageOnlyTransform(tuple(dataset_config["image_size"]))(image)
    prediction = extract_logits(model(tensor.unsqueeze(0).to(device))).argmax(1)[0]
    output = Path(args.output) if args.output else Path(config["output_dir"]) / "predictions" / image_path.name
    output.parent.mkdir(parents=True, exist_ok=True)
    colorize_mask(prediction.cpu().numpy()).resize(
        image.size, Image.Resampling.NEAREST
    ).save(output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

