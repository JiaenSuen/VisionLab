import argparse
import json
import os
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import (
    CaptionEvalTransform,
    CaptionTrainTransform,
    FlickrCaptionDataset,
    FlickrEvalImageDataset,
    Vocabulary,
    eval_image_collate,
    group_references,
    load_json,
    make_caption_collate_fn,
    parse_caption_file,
    records_for_split,
    save_json,
    split_by_image,
)
from loss import LabelSmoothingCrossEntropy, perplexity_from_loss, token_accuracy
from model import build_caption_model
from utils import (
    AverageMeter,
    append_csv,
    count_trainable_parameters,
    ensure_dir,
    evaluate_caption_metrics,
    format_metrics,
    load_checkpoint,
    save_checkpoint,
    save_config,
    seconds_to_hms,
    set_seed,
    write_csv,
    write_final_report,
)


# Default training configuration. CLI arguments can override common fields.
CONFIG: Dict = {
    "data_dir": "data",
    "image_dir": "data/Images",
    "captions_file": "data/captions.txt",
    "record_dir": "Record",
    "seed": 42,
    "train_ratio": 0.80,
    "valid_ratio": 0.10,
    "test_ratio": 0.10,
    "vocab_min_freq": 2,
    "vocab_max_size": 12000,
    "image_size": 224,
    "max_len": 36,
    "encoder_variant": "mini",  # mini / tiny / small
    "d_model": 256,
    "visual_pool_size": 7,
    "visual_queries": 32,
    "resampler_layers": 2,
    "decoder_layers": 3,
    "nhead": 4,
    "ffn_dim": 768,
    "dropout": 0.15,
    "batch_size": 48,
    "eval_batch_size": 64,
    "num_workers": 4,
    "epochs": 60,
    "warmup_epochs": 3,
    "grad_accum_steps": 1,
    "base_lr": 3.0e-4,
    "encoder_lr": 1.0e-4,
    "min_lr_ratio": 0.05,
    "weight_decay": 0.05,
    "label_smoothing": 0.10,
    "grad_clip_norm": 1.0,
    "amp": True,
    "compile_model": False,
    "resume_if_exists": True,
    "save_every_epoch": True,
    "beam_size_epoch": 1,
    "beam_size_final": 3,
    # Use a subset for epoch-wise train generation to keep evaluation inexpensive; final evaluation uses all splits.
    "trend_train_eval_limit": 800,
    "progress_bar": True,
}



def parse_args():
    parser = argparse.ArgumentParser(description="Train E-ConvNeXt + Transformer image captioning on Flickr8k/Flickr30k.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing Flickr images.")
    parser.add_argument("--captions-file", type=str, default=None, help="Caption annotation file.")
    parser.add_argument("--record-dir", type=str, default=None, help="Output directory for logs and checkpoints.")
    parser.add_argument("--encoder-variant", choices=["mini", "tiny", "small"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from record_dir/checkpoints/last.pt.")
    return parser.parse_args()


def build_runtime_config(args) -> Dict:
    config = dict(CONFIG)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config.update(json.load(f))

    overrides = {
        "image_dir": args.image_dir,
        "captions_file": args.captions_file,
        "record_dir": args.record_dir,
        "encoder_variant": args.encoder_variant,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "max_len": args.max_len,
        "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if args.no_amp:
        config["amp"] = False
    if args.no_resume:
        config["resume_if_exists"] = False

    # Keep data_dir coherent for reports when custom paths are supplied.
    if args.image_dir or args.captions_file:
        config["data_dir"] = os.path.commonpath([os.path.abspath(config["image_dir"]), os.path.abspath(config["captions_file"])])
    return config

def make_loaders(config: Dict, vocab: Vocabulary, splits: Dict, all_records):
    image_dir = config["image_dir"]
    train_records = records_for_split(all_records, splits["train"])
    valid_records = records_for_split(all_records, splits["valid"])
    test_records = records_for_split(all_records, splits["test"])

    train_tf = CaptionTrainTransform(config["image_size"])
    eval_tf = CaptionEvalTransform(config["image_size"], resize_short_side=256)
    collate = make_caption_collate_fn(vocab.pad_idx)

    train_ds = FlickrCaptionDataset(train_records, image_dir, vocab, config["max_len"], transform=train_tf)
    train_loss_ds = FlickrCaptionDataset(train_records, image_dir, vocab, config["max_len"], transform=eval_tf)
    valid_loss_ds = FlickrCaptionDataset(valid_records, image_dir, vocab, config["max_len"], transform=eval_tf)
    test_loss_ds = FlickrCaptionDataset(test_records, image_dir, vocab, config["max_len"], transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
        persistent_workers=config["num_workers"] > 0,
    )
    train_loss_loader = DataLoader(
        train_loss_ds,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=config["num_workers"] > 0,
    )
    valid_loss_loader = DataLoader(
        valid_loss_ds,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=config["num_workers"] > 0,
    )
    test_loss_loader = DataLoader(
        test_loss_ds,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=config["num_workers"] > 0,
    )

    train_eval_images = splits["train"]
    if config.get("trend_train_eval_limit", 0) and len(train_eval_images) > config["trend_train_eval_limit"]:
        train_eval_images = train_eval_images[: config["trend_train_eval_limit"]]

    eval_loaders = {
        "train": DataLoader(
            FlickrEvalImageDataset(train_eval_images, image_dir, transform=eval_tf),
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=eval_image_collate,
            persistent_workers=config["num_workers"] > 0,
        ),
        "train_full": DataLoader(
            FlickrEvalImageDataset(splits["train"], image_dir, transform=eval_tf),
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=eval_image_collate,
            persistent_workers=config["num_workers"] > 0,
        ),
        "valid": DataLoader(
            FlickrEvalImageDataset(splits["valid"], image_dir, transform=eval_tf),
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=eval_image_collate,
            persistent_workers=config["num_workers"] > 0,
        ),
        "test": DataLoader(
            FlickrEvalImageDataset(splits["test"], image_dir, transform=eval_tf),
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=eval_image_collate,
            persistent_workers=config["num_workers"] > 0,
        ),
    }
    loss_loaders = {"train": train_loss_loader, "valid": valid_loss_loader, "test": test_loss_loader}
    return train_loader, loss_loaders, eval_loaders


def build_optimizer_and_scheduler(model, config: Dict, steps_per_epoch: int):
    # AdamW best practice: do not decay bias, BatchNorm/LayerNorm, or scalar parameters.
    groups = {
        "encoder_decay": {"params": [], "lr": config["encoder_lr"], "weight_decay": config["weight_decay"]},
        "encoder_no_decay": {"params": [], "lr": config["encoder_lr"], "weight_decay": 0.0},
        "other_decay": {"params": [], "lr": config["base_lr"], "weight_decay": config["weight_decay"]},
        "other_no_decay": {"params": [], "lr": config["base_lr"], "weight_decay": 0.0},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = name.startswith("encoder.")
        no_decay = param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or ".bn" in name.lower()
        key = ("encoder" if is_encoder else "other") + ("_no_decay" if no_decay else "_decay")
        groups[key]["params"].append(param)

    optimizer = torch.optim.AdamW(
        [g for g in groups.values() if g["params"]],
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    total_steps = max(1, steps_per_epoch * config["epochs"])
    warmup_steps = max(1, steps_per_epoch * config["warmup_epochs"])
    min_ratio = float(config["min_lr_ratio"])

    def lr_lambda(step: int):
        if step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def autocast_context(enabled: bool):
    return torch.amp.autocast("cuda", enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, config: Dict, epoch: int):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    grad_accum = max(1, int(config["grad_accum_steps"]))
    amp_enabled = bool(config["amp"] and device.type == "cuda")
    use_progress = bool(config.get("progress_bar", True))

    iterator = tqdm(
        loader,
        total=len(loader),
        desc=f"Train E{epoch:03d}",
        dynamic_ncols=True,
        leave=True,
        disable=not use_progress,
    )

    for step, batch in enumerate(iterator, start=1):
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target_ids = batch["target_ids"].to(device, non_blocking=True)

        with autocast_context(amp_enabled):
            logits = model(images, input_ids)
            loss = criterion(logits, target_ids) / grad_accum

        scaler.scale(loss).backward()

        if step % grad_accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        batch_loss = float(loss.detach().item()) * grad_accum
        with torch.no_grad():
            acc = token_accuracy(logits.detach(), target_ids, ignore_index=criterion.ignore_index)
        loss_meter.update(batch_loss, images.size(0))
        acc_meter.update(acc, images.size(0))

        lr = optimizer.param_groups[-1]["lr"]
        if use_progress:
            iterator.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                ppl=f"{perplexity_from_loss(loss_meter.avg):.2f}",
                tok_acc=f"{acc_meter.avg:.4f}",
                lr=f"{lr:.2e}",
            )

    return {"loss": loss_meter.avg, "ppl": perplexity_from_loss(loss_meter.avg), "token_acc": acc_meter.avg}


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, amp_enabled: bool, desc: str = "Eval Loss", progress_bar: bool = True):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    iterator = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        dynamic_ncols=True,
        leave=False,
        disable=not progress_bar,
    )
    for batch in iterator:
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target_ids = batch["target_ids"].to(device, non_blocking=True)
        with autocast_context(amp_enabled):
            logits = model(images, input_ids)
            loss = criterion(logits, target_ids)
        acc = token_accuracy(logits, target_ids, ignore_index=criterion.ignore_index)
        loss_meter.update(float(loss.item()), images.size(0))
        acc_meter.update(acc, images.size(0))
        if progress_bar:
            iterator.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                ppl=f"{perplexity_from_loss(loss_meter.avg):.2f}",
                tok_acc=f"{acc_meter.avg:.4f}",
            )
    return {"loss": loss_meter.avg, "ppl": perplexity_from_loss(loss_meter.avg), "token_acc": acc_meter.avg}


@torch.no_grad()
def generate_predictions(
    model,
    loader,
    vocab: Vocabulary,
    device,
    max_len: int,
    beam_size: int = 1,
    desc: str = "Generate",
    progress_bar: bool = True,
):
    model.eval()
    predictions = {}
    rows = []
    iterator = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        dynamic_ncols=True,
        leave=False,
        disable=not progress_bar,
    )
    for batch in iterator:
        images = batch["images"].to(device, non_blocking=True)
        if beam_size > 1:
            seqs = model.generate_beam(images, max_len=max_len, beam_size=beam_size)
        else:
            seqs = model.generate_greedy(images, max_len=max_len)
        seqs = seqs.detach().cpu().tolist()
        for image_name, ids in zip(batch["image_names"], seqs):
            caption = vocab.decode(ids)
            predictions[image_name] = caption
            rows.append({"image": image_name, "prediction": caption})
        if progress_bar:
            iterator.set_postfix(images=len(rows), beam=beam_size)
    return predictions, rows


def prepare_data(config: Dict):
    record_dir = config["record_dir"]
    ensure_dir(record_dir)
    all_records = parse_caption_file(config["captions_file"])
    split_path = os.path.join(record_dir, "split.json")
    if os.path.isfile(split_path):
        splits = load_json(split_path)
    else:
        splits = split_by_image(
            all_records,
            train_ratio=config["train_ratio"],
            valid_ratio=config["valid_ratio"],
            test_ratio=config["test_ratio"],
            seed=config["seed"],
        )
        save_json(splits, split_path)

    vocab_path = os.path.join(record_dir, "vocab.json")
    if os.path.isfile(vocab_path):
        vocab = Vocabulary.load(vocab_path)
    else:
        train_records = records_for_split(all_records, splits["train"])
        vocab = Vocabulary(config["vocab_min_freq"], config["vocab_max_size"]).build(r["caption"] for r in train_records)
        vocab.save(vocab_path)

    references_all = group_references(all_records)
    references_by_phase = {
        "train": {k: references_all[k] for k in splits["train"] if k in references_all},
        "valid": {k: references_all[k] for k in splits["valid"] if k in references_all},
        "test": {k: references_all[k] for k in splits["test"] if k in references_all},
    }
    return all_records, splits, vocab, references_by_phase


def main():
    args = parse_args()
    config = build_runtime_config(args)
    set_seed(config["seed"])
    record_dir = config["record_dir"]
    ckpt_dir = os.path.join(record_dir, "checkpoints")
    ensure_dir(record_dir)
    ensure_dir(ckpt_dir)
    save_config(config, os.path.join(record_dir, "config_used.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    all_records, splits, vocab, references_by_phase = prepare_data(config)
    print(f"Images split: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")
    print(f"Caption records: {len(all_records)} | Vocab: {len(vocab)}")

    train_loader, loss_loaders, eval_loaders = make_loaders(config, vocab, splits, all_records)
    model = build_caption_model(config, len(vocab), vocab.pad_idx, vocab.bos_idx, vocab.eos_idx).to(device)
    if config.get("compile_model", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    model_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {model_params:.3f} M")

    criterion = LabelSmoothingCrossEntropy(config["label_smoothing"], ignore_index=vocab.pad_idx)
    import math
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, config["grad_accum_steps"])))
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"] and device.type == "cuda"))

    start_epoch = 1
    best_score = -1.0
    best_epoch = 0
    last_ckpt = os.path.join(ckpt_dir, "last.pt")
    if config["resume_if_exists"] and os.path.isfile(last_ckpt):
        ckpt = load_checkpoint(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt.get("best_score", best_score))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        print(f"Resumed from epoch {start_epoch - 1}, best CIDEr={best_score:.4f}")

    history_path = os.path.join(record_dir, "history.csv")
    start_time = time.time()
    amp_enabled = bool(config["amp"] and device.type == "cuda")

    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, config, epoch)
        valid_loss = evaluate_loss(
            model,
            loss_loaders["valid"],
            criterion,
            device,
            amp_enabled,
            desc=f"Valid Loss E{epoch:03d}",
            progress_bar=bool(config.get("progress_bar", True)),
        )

        train_preds, _ = generate_predictions(
            model,
            eval_loaders["train"],
            vocab,
            device,
            config["max_len"],
            beam_size=config["beam_size_epoch"],
            desc=f"Train Caption E{epoch:03d}",
            progress_bar=bool(config.get("progress_bar", True)),
        )
        valid_preds, _ = generate_predictions(
            model,
            eval_loaders["valid"],
            vocab,
            device,
            config["max_len"],
            beam_size=config["beam_size_epoch"],
            desc=f"Valid Caption E{epoch:03d}",
            progress_bar=bool(config.get("progress_bar", True)),
        )
        train_gen = evaluate_caption_metrics(train_preds, references_by_phase["train"])
        valid_gen = evaluate_caption_metrics(valid_preds, references_by_phase["valid"])

        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "phase": "train",
            **{k: round(v, 6) for k, v in train_metrics.items()},
            **{k: round(v, 6) for k, v in train_gen.items()},
            "lr": optimizer.param_groups[-1]["lr"],
            "epoch_time_sec": round(elapsed, 2),
        }
        append_csv(history_path, row)
        row = {
            "epoch": epoch,
            "phase": "valid",
            **{k: round(v, 6) for k, v in valid_loss.items()},
            **{k: round(v, 6) for k, v in valid_gen.items()},
            "lr": optimizer.param_groups[-1]["lr"],
            "epoch_time_sec": round(elapsed, 2),
        }
        append_csv(history_path, row)

        print(f"Epoch {epoch:03d} done in {seconds_to_hms(elapsed)}")
        print("Train:", format_metrics({**train_metrics, **train_gen}))
        print("Valid:", format_metrics({**valid_loss, **valid_gen}))

        score = valid_gen.get("CIDEr", 0.0)
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "config": config,
                    "vocab": vocab.to_dict(),
                },
                os.path.join(ckpt_dir, "best.pt"),
            )
            print(f"Saved new best checkpoint: CIDEr={best_score:.4f}")

        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "config": config,
                "vocab": vocab.to_dict(),
            },
            last_ckpt,
        )
        if config["save_every_epoch"]:
            save_checkpoint({"epoch": epoch, "model": model.state_dict(), "config": config, "vocab": vocab.to_dict()}, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.isfile(best_path):
        ckpt = load_checkpoint(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        best_score = float(ckpt.get("best_score", best_score))
        print(f"Loaded best checkpoint for final evaluation: epoch={best_epoch}, CIDEr={best_score:.4f}")

    final_metrics_by_phase = {}
    summary_rows = []
    for phase, loader_name in [("train", "train_full"), ("valid", "valid"), ("test", "test")]:
        loss_metrics = evaluate_loss(
            model,
            loss_loaders[phase],
            criterion,
            device,
            amp_enabled,
            desc=f"Final {phase} Loss",
            progress_bar=bool(config.get("progress_bar", True)),
        )
        preds, pred_rows = generate_predictions(
            model,
            eval_loaders[loader_name],
            vocab,
            device,
            config["max_len"],
            beam_size=config["beam_size_final"],
            desc=f"Final {phase} Caption",
            progress_bar=bool(config.get("progress_bar", True)),
        )
        gen_metrics = evaluate_caption_metrics(preds, references_by_phase[phase])
        metrics = {**loss_metrics, **gen_metrics}
        final_metrics_by_phase[phase] = metrics
        for row in pred_rows:
            row["references"] = " ||| ".join(references_by_phase[phase].get(row["image"], []))
        write_csv(os.path.join(record_dir, f"predictions_{phase}.csv"), pred_rows)
        summary_rows.append({"phase": phase, **{k: round(v, 6) for k, v in metrics.items()}})
        print(f"Final {phase}:", format_metrics(metrics))

    write_csv(os.path.join(record_dir, "metrics_summary.csv"), summary_rows)
    total_elapsed = time.time() - start_time
    write_final_report(
        os.path.join(record_dir, "final_report.txt"),
        config,
        final_metrics_by_phase,
        model_params,
        best_epoch,
        best_score,
        total_elapsed,
    )
    print(f"All outputs saved to: {record_dir}")


if __name__ == "__main__":
    main()
