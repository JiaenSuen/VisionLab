from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.camvid import CAMVID_CLASSES, CamVidDataset
from metrics import SegmentationConfusionMatrix, format_metrics
from models import build_model
from project_utils import extract_logits, load_config
from utils.checkpoint import load_model_checkpoint
from utils.transforms import EvalTransform
from utils.visualization import save_prediction


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lraspp.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config["dataset"]
    num_classes = int(dataset_config["num_classes"])
    ignore_index = int(dataset_config.get("ignore_index", 255))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CamVidDataset(
        dataset_config["root"],
        args.split,
        EvalTransform(tuple(dataset_config["image_size"])),
        ignore_index,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(dataset_config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    model = build_model(config["model"]["name"], num_classes, pretrained=False)
    load_model_checkpoint(args.checkpoint, model, device)
    model.to(device).eval()
    confusion = SegmentationConfusionMatrix(num_classes, ignore_index)
    prediction_dir = Path(config["output_dir"]) / "predictions"

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        prediction = extract_logits(model(images)).argmax(dim=1)
        confusion.update(prediction, masks)
        if args.save_predictions:
            for mask, filename in zip(prediction.cpu().numpy(), batch["filename"]):
                save_prediction(mask, prediction_dir / filename)

    metrics = confusion.compute()
    metrics["class_names"] = list(CAMVID_CLASSES)
    metrics_path = Path(config["output_dir"]) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(format_metrics(metrics, CAMVID_CLASSES))
    print(f"saved: {metrics_path}")


if __name__ == "__main__":
    main()
