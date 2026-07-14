from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader

from datasets.camvid import CAMVID_CLASSES, CamVidDataset
from metrics import SegmentationConfusionMatrix, format_metrics
from models import build_model
from project_utils import build_loss, extract_logits, load_config, set_seed
from utils.checkpoint import save_checkpoint
from utils.transforms import EvalTransform, TrainTransform


def run_epoch(
    model,
    loader,
    criterion,
    device,
    num_classes,
    ignore_index,
    optimizer=None,
    scaler=None,
    use_amp=False,
    max_batches=None,
):
    training = optimizer is not None
    model.train(training)
    confusion = SegmentationConfusionMatrix(num_classes, ignore_index)
    total_loss = 0.0
    batches = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
                dtype=torch.float16,
            ):
                logits = extract_logits(model(images))
                loss = criterion(logits, masks)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach())
        batches += 1
        confusion.update(logits.argmax(dim=1), masks)

    metrics = confusion.compute()
    metrics["loss"] = total_loss / max(batches, 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lraspp.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    dataset_config = config["dataset"]
    train_config = config["training"]
    image_size = tuple(dataset_config["image_size"])
    ignore_index = int(dataset_config.get("ignore_index", 255))
    num_classes = int(dataset_config["num_classes"])
    epochs = args.epochs or int(train_config["epochs"])
    output_dir = Path(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_dataset = CamVidDataset(
        dataset_config["root"], "train", TrainTransform(image_size), ignore_index
    )
    val_dataset = CamVidDataset(
        dataset_config["root"], "val", EvalTransform(image_size), ignore_index
    )
    loader_args = {
        "batch_size": int(train_config["batch_size"]),
        "num_workers": int(dataset_config.get("num_workers", 4)),
        "pin_memory": pin_memory,
        "persistent_workers": int(dataset_config.get("num_workers", 4)) > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    model = build_model(
        config["model"]["name"],
        num_classes,
        bool(config["model"].get("pretrained", True)),
    ).to(device)
    criterion = build_loss(config["loss"], ignore_index)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = PolynomialLR(optimizer, total_iters=epochs, power=0.9)
    use_amp = bool(train_config.get("mixed_precision", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    log_path = output_dir / "train_log.csv"
    best_iou = -1.0
    print(f"device={device}, train={len(train_dataset)}, val={len(val_dataset)}")
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=("epoch", "lr", "train_loss", "train_miou", "val_loss", "val_miou"),
        )
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                num_classes,
                ignore_index,
                optimizer,
                scaler,
                use_amp,
                args.max_train_batches,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                num_classes,
                ignore_index,
                use_amp=use_amp,
                max_batches=args.max_val_batches,
            )
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            writer.writerow(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": train_metrics["loss"],
                    "train_miou": train_metrics["mean_iou"],
                    "val_loss": val_metrics["loss"],
                    "val_miou": val_metrics["mean_iou"],
                }
            )
            log_file.flush()
            print(
                f"epoch {epoch:03d}/{epochs} | "
                f"train loss={train_metrics['loss']:.4f} mIoU={train_metrics['mean_iou']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} mIoU={val_metrics['mean_iou']:.4f}"
            )
            save_checkpoint(
                checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch, val_metrics
            )
            if val_metrics["mean_iou"] > best_iou:
                best_iou = val_metrics["mean_iou"]
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_metrics,
                )

    print(f"best validation mIoU={best_iou:.4f}")
    print(format_metrics(val_metrics, CAMVID_CLASSES))


if __name__ == "__main__":
    main()
