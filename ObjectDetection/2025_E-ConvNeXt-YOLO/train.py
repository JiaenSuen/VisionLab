"""
train.py

Training entrypoint for E-ConvNeXt-mini/small + YOLOv10n-style detector on VOC.

Expected data layout by default:
  data/images/
  data/labels/
  data/train.csv
  data/test.csv

CSV format:
  img,label
  2007_000033.jpg,2007_000033.txt

Label format per row:
  class x_center y_center width height
where xywh are normalized to [0,1].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from copy import deepcopy
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from dataset import (
    VOC_CLASSES,
    RawVOCDataset,
    build_transforms,
    raw_voc_collate,
    scan_dataset,
)
from eval import get_bboxes, mean_average_precision, save_checkpoint
from loss import YOLOv10EConvNeXtLoss
from yolo import build_yolov10_econvnext, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_fast_torch() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def prepare_csv(path: str, header: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_csv(path: str, row: List[object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def build_loader(args, dataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=raw_voc_collate,
    )
    if args.num_workers > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
    return DataLoader(**kwargs)


def build_optimizer(model: torch.nn.Module, args) -> optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.base_lr,
        betas=(0.9, 0.999),
    )


def get_lr(args, epoch_float: float) -> float:
    if epoch_float < args.warmup_epochs:
        a = max(epoch_float / max(args.warmup_epochs, 1e-9), 0.0)
        return args.warmup_start_lr + a * (args.base_lr - args.warmup_start_lr)
    progress = (epoch_float - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + (args.base_lr - args.min_lr) * cosine


def set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


class ModelEMA:
    """EMA over the full detector for stable evaluation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9998):
        base = model.module if hasattr(model, "module") else model
        self.ema = deepcopy(base).eval()
        self.decay = float(decay)
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        msd = (model.module if hasattr(model, "module") else model).state_dict()
        for k, v in self.ema.state_dict().items():
            if not v.dtype.is_floating_point:
                v.copy_(msd[k])
            else:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)

    def state_dict(self):
        return {"updates": self.updates, "ema": self.ema.state_dict()}

    def load_state_dict(self, state) -> None:
        if not state:
            return
        self.updates = int(state.get("updates", 0))
        self.ema.load_state_dict(state["ema"], strict=True)


def model_for_eval(model: torch.nn.Module, ema: Optional[ModelEMA], use_ema: bool):
    return ema.ema if use_ema and ema is not None else model


def save_training_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler,
    ema: Optional[ModelEMA],
    epoch: int,
    best_map: float,
    best_epoch: int,
    args,
) -> None:
    base = model.module if hasattr(model, "module") else model
    state = {
        "epoch": epoch,
        "model": base.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "best_map": best_map,
        "best_epoch": best_epoch,
        "args": vars(args),
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_checkpoint(state, filename)


def load_resume(args, model, optimizer, scaler, ema: Optional[ModelEMA]):
    if not args.resume:
        return 0, 0.0, 0, False
    path = args.resume if isinstance(args.resume, str) else args.last_model_file
    if not path or not os.path.exists(path):
        return 0, 0.0, 0, False
    ckpt = torch.load(path, map_location="cpu")
    base = model.module if hasattr(model, "module") else model
    base.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_map", 0.0)), int(ckpt.get("best_epoch", 0)), True


def write_dataset_record(args, train_stats: Dict[str, object], test_stats: Dict[str, object]) -> None:
    path = os.path.join(args.record_dir, "dataset_record.csv")
    os.makedirs(args.record_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "split", "csv_path", "image_count", "label_files_found", "missing_label_files",
            "empty_label_files", "total_boxes", "class_id", "class_name", "class_box_count",
        ])
        for stats in (train_stats, test_stats):
            for cls in range(args.num_classes):
                w.writerow([
                    stats["split"], stats["csv_path"], stats["image_count"], stats["label_files_found"],
                    stats["missing_label_files"], stats["empty_label_files"], stats["total_boxes"],
                    cls, VOC_CLASSES.get(cls, f"class_{cls}"), stats["per_class_counts"].get(cls, 0),
                ])


def append_per_class_ap(args, epoch: int, split_name: str, ap_by_class: Optional[Dict[int, float]]) -> None:
    if ap_by_class is None:
        return
    path = os.path.join(args.record_dir, "per_class_ap.csv")
    prepare_csv(path, ["epoch", "split", "class_id", "class_name", "ap"])
    for cls in range(args.num_classes):
        append_csv(path, [epoch, split_name, cls, VOC_CLASSES.get(cls, f"class_{cls}"), f"{float(ap_by_class.get(cls, 0.0)):.6f}"])


def write_report(args, train_stats, test_stats, summary) -> None:
    lines = []
    lines.append("E-ConvNeXt + YOLOv10n-style VOC Experiment")
    lines.append("=" * 64)
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Architecture: E-ConvNeXt backbone + YOLOv10-style PAFPN + dual objectness-free head.")
    lines.append("Loss: one-to-many + one-to-one anchor-free CIoU and quality focal BCE.")
    lines.append("")
    for k, v in sorted(vars(args).items()):
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append(f"Training images: {train_stats['image_count']}, boxes: {train_stats['total_boxes']}")
    lines.append(f"Testing images: {test_stats['image_count']}, boxes: {test_stats['total_boxes']}")
    lines.append("")
    lines.append("Final summary:")
    for k, v in summary.items():
        lines.append(f"{k}: {v}")
    with open(os.path.join(args.record_dir, "experiment_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def train_one_epoch(args, loader, model, optimizer, loss_fn, scaler, ema: Optional[ModelEMA], epoch: int, device: str):
    model.train()
    if hasattr(loader.dataset.transform, "set_epoch"):
        loader.dataset.transform.set_epoch(epoch)

    mean_loss, mean_box, mean_cls, mean_pos = [], [], [], []
    skipped = 0
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)
    use_amp_now = bool(args.amp and device.startswith("cuda"))
    try:
        autocast_ctx = torch.amp.autocast
        autocast_kwargs = {"device_type": "cuda", "enabled": use_amp_now}
    except Exception:
        autocast_ctx = torch.cuda.amp.autocast
        autocast_kwargs = {"enabled": use_amp_now}

    pbar = tqdm(
        enumerate(loader, start=1),
        total=num_batches,
        desc=f"Epoch {epoch:03d}/{args.epochs}",
        dynamic_ncols=True,
        leave=True,
    )

    for batch_idx, (images, targets) in pbar:
        epoch_float = (epoch - 1) + (batch_idx - 1) / max(num_batches, 1)
        lr = get_lr(args, epoch_float)
        set_optimizer_lr(optimizer, lr)

        images = images.to(device, non_blocking=True)
        if args.channels_last and device.startswith("cuda"):
            images = images.contiguous(memory_format=torch.channels_last)
        targets = [t.to(device, non_blocking=True) for t in targets]

        with autocast_ctx(**autocast_kwargs):
            outputs = model(images, branch="both")
            loss, stats = loss_fn(outputs, targets)
            loss = loss / args.accum_steps

        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix({"skip": skipped, "lr": f"{lr:.2e}"})
            continue

        scaler.scale(loss).backward()

        if batch_idx % args.accum_steps == 0 or batch_idx == num_batches:
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        mean_loss.append(float(stats.get("loss", 0.0)))
        mean_box.append(float(stats.get("box", 0.0)))
        mean_cls.append(float(stats.get("cls", 0.0)))
        mean_pos.append(float(stats.get("pos", 0.0)))

        if batch_idx % args.log_every == 0 or batch_idx == num_batches:
            pbar.set_postfix({
                "loss": f"{np.mean(mean_loss):.4f}",
                "box": f"{np.mean(mean_box):.4f}",
                "cls": f"{np.mean(mean_cls):.4f}",
                "pos": f"{np.mean(mean_pos):.1f}",
                "lr": f"{lr:.2e}",
                "skip": skipped,
            })

    return {
        "loss": float(np.mean(mean_loss)) if mean_loss else float("nan"),
        "box": float(np.mean(mean_box)) if mean_box else 0.0,
        "cls": float(np.mean(mean_cls)) if mean_cls else 0.0,
        "pos": float(np.mean(mean_pos)) if mean_pos else 0.0,
        "lr": optimizer.param_groups[0]["lr"],
        "skipped": skipped,
    }


def evaluate_map(args, loader, model, ema: Optional[ModelEMA], device: str, desc: str, max_eval_batches: Optional[int], max_detections: int, return_per_class: bool = False):
    eval_model = model_for_eval(model, ema, args.eval_use_ema)
    pred, tgt = get_bboxes(
        loader=loader,
        model=eval_model,
        iou_threshold=args.nms_iou,
        threshold=args.map_conf,
        device=device,
        desc=desc,
        max_eval_batches=max_eval_batches,
        max_detections=max_detections,
        use_amp=args.amp,
        pre_nms_topk=args.pre_nms_topk,
        branch=args.eval_branch,
    )
    result = mean_average_precision(pred, tgt, iou_threshold=args.map_iou, num_classes=args.num_classes, return_per_class=return_per_class)
    if return_per_class:
        mAP, ap_by_class = result
        return mAP, len(pred), len(tgt), ap_by_class
    return result, len(pred), len(tgt), None


# --------------------
# config
# --------------------
 
CONFIG = dict(
    model_name="EConvNeXtMini_YOLOv10n_VOC",
    variant="mini",                 # "mini", "small", or "tiny"
    num_classes=20,
    image_size=448,
    img_dir="data/images/",
    label_dir="data/labels/",
    train_csv="data/train.csv",
    test_csv="data/test.csv",
    record_root="record",
    backbone_weights="",            # optional classifier/pretrained E-ConvNeXt checkpoint

    epochs=100,
    batch_size=8,                    # For the small variant on 8 GB VRAM, start with 4
    accum_steps=2,                   # For small + batch size 4, accumulation=4 is a practical starting point
    num_workers=4,
    pin_memory=True,
    close_aug_epochs=20,

    base_lr=3.0e-4,
    min_lr=1.0e-5,
    warmup_epochs=3.0,
    warmup_start_lr=1.0e-5,
    weight_decay=3.0e-4,
    max_grad_norm=5.0,
    drop_path_rate=0.05,
    one2one_gain=0.5,

    amp=True,
    channels_last=True,
    compile=False,
    seed=123,
    ema_decay=0.9998,
    eval_use_ema=True,
    eval_branch="one2one",           # "one2one" or "one2many"

    map_iou=0.5,
    nms_iou=0.45,
    map_conf=1e-4,
    pre_nms_topk=2500,

    quick_eval_every=5,
    quick_eval_start=5,
    quick_eval_batches=30,
    quick_max_detections=100,
    full_eval_every=20,
    full_eval_start=20,
    full_max_detections=300,

    resume="",                      # e.g. "record/EConvNeXtMini_YOLOv10n_VOC/last.pth.tar"
    torch_threads=0,                 # CPU only; 0 keeps PyTorch default
    log_every=10,                    # tqdm postfix refresh frequency
    save_every=10,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a CLI directly from CONFIG so command-line runs remain reproducible."""
    parser = argparse.ArgumentParser(
        description="Train E-ConvNeXt + YOLOv10-style detector on VOC/YOLO-format labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for key, default in CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, int):
            parser.add_argument(flag, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(flag, type=float, default=default)
        else:
            parser.add_argument(flag, type=str, default=default)
    return parser


def get_config(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.variant not in {"mini", "small", "tiny"}:
        raise ValueError("--variant must be 'mini', 'small', or 'tiny'")
    if args.eval_branch not in {"one2one", "one2many"}:
        raise ValueError("--eval-branch must be 'one2one' or 'one2many'")

    args.record_dir = os.path.join(args.record_root, args.model_name)
    args.metrics_csv = os.path.join(args.record_dir, "training_metrics.csv")
    args.test_metrics_csv = os.path.join(args.record_dir, "test_metrics.csv")
    args.best_model_file = os.path.join(args.record_dir, "best.pth.tar")
    args.last_model_file = os.path.join(args.record_dir, "last.pth.tar")
    return args


def main():
    args = get_config()
    seed_everything(args.seed)
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    enable_fast_torch()
    os.makedirs(args.record_dir, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(args.record_dir, "train_snapshot.py"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Model:", args.model_name)
    print("Variant:", args.variant)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if not os.path.exists(args.train_csv) or not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Expected CSV files:\n  {args.train_csv}\n  {args.test_csv}")

    train_stats = scan_dataset(args.train_csv, args.label_dir, "train", args.num_classes)
    test_stats = scan_dataset(args.test_csv, args.label_dir, "test", args.num_classes)
    write_dataset_record(args, train_stats, test_stats)
    print(f"Train images={train_stats['image_count']} boxes={train_stats['total_boxes']}")
    print(f"Test  images={test_stats['image_count']} boxes={test_stats['total_boxes']}")

    train_transform, eval_transform = build_transforms(args.image_size, args.epochs, args.close_aug_epochs)
    train_dataset = RawVOCDataset(args.train_csv, args.img_dir, args.label_dir, transform=train_transform)
    test_dataset = RawVOCDataset(args.test_csv, args.img_dir, args.label_dir, transform=eval_transform)
    train_loader = build_loader(args, train_dataset, args.batch_size, shuffle=True, drop_last=True)
    test_loader = build_loader(args, test_dataset, args.batch_size, shuffle=False, drop_last=False)

    model = build_yolov10_econvnext(
        num_classes=args.num_classes,
        variant=args.variant,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    if args.backbone_weights:
        missing, unexpected = model.load_backbone_weights(args.backbone_weights, strict=False)
        print(f"Loaded backbone weights: {args.backbone_weights}")
        print(f"Backbone missing={len(missing)} unexpected={len(unexpected)}")

    if args.channels_last and device.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as exc:
            print(f"torch.compile failed; continuing without compile: {exc}")

    print(f"Params total: {count_parameters(model, trainable_only=False):.3f}M")
    print(f"Params trainable: {count_parameters(model, trainable_only=True):.3f}M")

    optimizer = build_optimizer(model, args)
    loss_fn = YOLOv10EConvNeXtLoss(num_classes=args.num_classes, one2one_gain=args.one2one_gain)
    scaler_enabled = bool(args.amp and device.startswith("cuda"))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # Compatibility with older PyTorch versions.
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    ema = ModelEMA(model, args.ema_decay) if args.eval_use_ema else None

    last_epoch, best_map, best_epoch, resumed = load_resume(args, model, optimizer, scaler, ema)
    start_epoch = last_epoch + 1
    print(f"Resumed from epoch {last_epoch}" if resumed else "Starting clean experiment")

    prepare_csv(args.metrics_csv, [
        "epoch", "train_loss", "box_loss", "cls_loss", "avg_pos", "quick_test_mAP", "full_test_mAP",
        "best_full_test_mAP", "learning_rate", "epoch_time_sec", "quick_pred_boxes", "quick_true_boxes",
        "full_pred_boxes", "full_true_boxes", "skipped_batches",
    ])
    prepare_csv(args.test_metrics_csv, [
        "epoch", "eval_type", "conf_threshold", "test_mAP", "test_pred_boxes", "test_true_boxes",
        "map_iou_threshold", "nms_iou_threshold", "max_detections_per_image", "eval_branch",
    ])

    final = {}
    session_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        stats = train_one_epoch(args, train_loader, model, optimizer, loss_fn, scaler, ema, epoch, device)
        quick_mAP = None
        full_mAP = None
        quick_pred = quick_true = full_pred = full_true = None

        if epoch >= args.quick_eval_start and args.quick_eval_every > 0 and epoch % args.quick_eval_every == 0:
            quick_mAP, quick_pred, quick_true, _ = evaluate_map(
                args, test_loader, model, ema, device,
                desc=f"Quick mAP epoch {epoch}",
                max_eval_batches=args.quick_eval_batches,
                max_detections=args.quick_max_detections,
                return_per_class=False,
            )
            append_csv(args.test_metrics_csv, [
                epoch, "quick", args.map_conf, f"{quick_mAP:.6f}", quick_pred, quick_true,
                args.map_iou, args.nms_iou, args.quick_max_detections, args.eval_branch,
            ])

        if epoch >= args.full_eval_start and args.full_eval_every > 0 and epoch % args.full_eval_every == 0:
            full_mAP, full_pred, full_true, ap_by_class = evaluate_map(
                args, test_loader, model, ema, device,
                desc=f"FULL mAP epoch {epoch}",
                max_eval_batches=None,
                max_detections=args.full_max_detections,
                return_per_class=True,
            )
            append_per_class_ap(args, epoch, "test_full", ap_by_class)
            append_csv(args.test_metrics_csv, [
                epoch, "full", args.map_conf, f"{full_mAP:.6f}", full_pred, full_true,
                args.map_iou, args.nms_iou, args.full_max_detections, args.eval_branch,
            ])
            if full_mAP > best_map:
                best_map = full_mAP
                best_epoch = epoch
                save_training_checkpoint(args.best_model_file, model, optimizer, scaler, ema, epoch, best_map, best_epoch, args)
                print(f"New best full mAP@{args.map_iou}: {best_map:.4f} at epoch {epoch}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_training_checkpoint(args.last_model_file, model, optimizer, scaler, ema, epoch, best_map, best_epoch, args)

        elapsed = time.time() - epoch_start
        append_csv(args.metrics_csv, [
            epoch,
            f"{stats['loss']:.6f}",
            f"{stats['box']:.6f}",
            f"{stats['cls']:.6f}",
            f"{stats['pos']:.3f}",
            "" if quick_mAP is None else f"{quick_mAP:.6f}",
            "" if full_mAP is None else f"{full_mAP:.6f}",
            f"{best_map:.6f}",
            f"{stats['lr']:.8f}",
            f"{elapsed:.2f}",
            "" if quick_pred is None else quick_pred,
            "" if quick_true is None else quick_true,
            "" if full_pred is None else full_pred,
            "" if full_true is None else full_true,
            stats["skipped"],
        ])

        print(
            f"Epoch {epoch:03d} done: loss={stats['loss']:.4f}, "
            f"quick={quick_mAP}, full={full_mAP}, best={best_map:.4f}@{best_epoch}, time={elapsed:.1f}s"
        )

        final = {
            "last_epoch": epoch,
            "best_full_mAP": f"{best_map:.6f}",
            "best_epoch": best_epoch,
            "last_train_loss": f"{stats['loss']:.6f}",
            "elapsed_hours": f"{(time.time() - session_start) / 3600.0:.3f}",
        }

    if best_epoch == 0:
        save_training_checkpoint(args.best_model_file, model, optimizer, scaler, ema, args.epochs, best_map, best_epoch, args)
    write_report(args, train_stats, test_stats, final)
    print("Training complete.")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
