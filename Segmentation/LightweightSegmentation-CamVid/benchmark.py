from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models import build_model
from project_utils import load_config
from utils.benchmark import benchmark_model
from utils.checkpoint import load_model_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lraspp.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config["dataset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config["model"]["name"], int(dataset_config["num_classes"]), pretrained=False
    )
    if args.checkpoint:
        load_model_checkpoint(args.checkpoint, model, device)
    height, width = dataset_config["image_size"]
    results = []
    for fp16 in (False, True):
        if fp16 and device.type != "cuda":
            continue
        results.append(
            benchmark_model(
                model,
                (1, 3, height, width),
                device,
                args.warmup,
                args.iterations,
                fp16,
            )
        )
    output = Path(config["output_dir"]) / "benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

