import csv
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import YOLOv5Nano
from dataset import VOCDataset
from utils import (
    VOC_CLASSES,
    mean_average_precision,
    get_bboxes,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss


# --------------------
# Reproducibility
# --------------------
SEED = 123
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------
# Configuration
# --------------------
MODEL_NAME = "YOLOv5nano"

AUTO_RESUME = True
SESSION_EPOCHS = 100

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHT_DECAY = 0
NUM_WORKERS = 1
PIN_MEMORY = True
NUM_CLASSES = 20
IMAGE_SIZE = 320
USE_AMP = True

IMG_DIR = "data/images/"
LABEL_DIR = "data/labels/"
TRAIN_CSV = "data/mini_train.csv"
TEST_CSV = "data/mini_test.csv"

METRICS_CSV = f"record/{MODEL_NAME}_training_metrics.csv"
TEST_METRICS_CSV = f"record/{MODEL_NAME}_test_metrics.csv"
DATASET_RECORD_CSV = f"record/{MODEL_NAME}_dataset_record.csv"
REPORT_TXT = f"record/{MODEL_NAME}_experiment_report.txt"
BEST_MODEL_FILE = f"record/{MODEL_NAME}_best.pth.tar"
LAST_MODEL_FILE = f"record/{MODEL_NAME}_last.pth.tar"    

EVAL_EVERY = 10
EVAL_START_EPOCH = 10
MAX_EVAL_BATCHES = None

MAP_CONF_THRESHOLD = 0.05
NMS_IOU_THRESHOLD = 0.45
MAP_IOU_THRESHOLD = 0.5
MAX_DETECTIONS_PER_IMAGE = 300

# Anchors normalized by image size.
# The order must match model output order:
#   output[0] -> smallest grid, largest objects
#   output[1] -> medium grid
#   output[2] -> largest grid, smallest objects
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]


class Compose:
    """Apply image transforms while keeping bounding boxes unchanged."""

    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, img, bboxes):
        for transform in self.transforms:
            img = transform(img)
        return img, bboxes


transform = Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)


def prepare_csv(csv_path, header):
    """Create a CSV file with a header if it does not exist."""

    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_csv(csv_path, row):
    """Append one row to a CSV file."""

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def read_annotation_csv(csv_path):
    """Read an img,label CSV file."""

    rows = []
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [name.strip() for name in reader.fieldnames]

        if "img" not in fieldnames or "label" not in fieldnames:
            raise ValueError(f"CSV must contain 'img' and 'label' columns: {csv_path}")

        for row in reader:
            image_name = str(row["img"]).strip()
            label_name = str(row["label"]).strip()
            if image_name and label_name:
                rows.append((image_name, label_name))

    return rows


def scan_dataset(csv_path, label_dir, split_name):
    """Collect dataset statistics for report and dataset-record CSV."""

    rows = read_annotation_csv(csv_path)
    per_class_counts = {class_id: 0 for class_id in range(NUM_CLASSES)}
    total_boxes = 0
    label_files_found = 0
    empty_label_files = 0
    missing_label_files = 0

    for _, label_name in rows:
        label_path = os.path.join(label_dir, label_name)
        if not os.path.exists(label_path):
            missing_label_files += 1
            continue

        label_files_found += 1
        file_box_count = 0

        with open(label_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(float(parts[0]))
                if 0 <= class_id < NUM_CLASSES:
                    per_class_counts[class_id] += 1
                    total_boxes += 1
                    file_box_count += 1

        if file_box_count == 0:
            empty_label_files += 1

    stats = {
        "split": split_name,
        "csv_path": csv_path,
        "image_count": len(rows),
        "label_files_found": label_files_found,
        "missing_label_files": missing_label_files,
        "empty_label_files": empty_label_files,
        "total_boxes": total_boxes,
        "per_class_counts": per_class_counts,
    }

    return stats


def write_dataset_record(train_stats, test_stats):
    """Write dataset statistics to CSV."""

    with open(DATASET_RECORD_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "csv_path",
                "image_count",
                "label_files_found",
                "missing_label_files",
                "empty_label_files",
                "total_boxes",
                "class_id",
                "class_name",
                "class_box_count",
            ]
        )

        for stats in [train_stats, test_stats]:
            for class_id in range(NUM_CLASSES):
                writer.writerow(
                    [
                        stats["split"],
                        stats["csv_path"],
                        stats["image_count"],
                        stats["label_files_found"],
                        stats["missing_label_files"],
                        stats["empty_label_files"],
                        stats["total_boxes"],
                        class_id,
                        VOC_CLASSES.get(class_id, f"class_{class_id}"),
                        stats["per_class_counts"].get(class_id, 0),
                    ]
                )


def count_targets(y):
    """Count positive object targets in a YOLO target tuple."""

    return int(sum((target[..., 0] == 1).sum().item() for target in y))


def train_one_epoch(train_loader, model, optimizer, loss_fn, scaled_anchors, scaler, epoch):
    """Train the model for one epoch."""

    model.train()

    loop = tqdm(
        train_loader,
        leave=True,
        desc=f"Training epoch {epoch}",
    )

    mean_loss = []
    positive_targets = 0
    use_amp_now = USE_AMP and DEVICE.startswith("cuda")

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE, non_blocking=True)

        if isinstance(y, (list, tuple)):
            y = [target.to(DEVICE, non_blocking=True) for target in y]
        else:
            raise TypeError("YOLO targets must be a tuple/list of 3 tensors.")

        positive_targets += count_targets(y)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp_now):
            outputs = model(x)

            if not isinstance(outputs, (list, tuple)):
                raise TypeError("Model output must be a tuple/list of 3 tensors.")

            loss = sum(
                loss_fn(outputs[i], y[i], scaled_anchors[i])
                for i in range(len(outputs))
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss.append(loss.item())

        loop.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{sum(mean_loss) / len(mean_loss):.4f}",
            pos_targets=positive_targets,
        )

    avg_loss = sum(mean_loss) / max(len(mean_loss), 1)
    return avg_loss, positive_targets


def evaluate_map(
    loader,
    model,
    anchors,
    iou_threshold,
    nms_iou_threshold,
    threshold,
    num_classes,
    desc,
    max_eval_batches=None,
    max_detections=300,
):
    """Evaluate mAP for a DataLoader."""

    pred_boxes, target_boxes = get_bboxes(
        loader=loader,
        model=model,
        iou_threshold=nms_iou_threshold,
        threshold=threshold,
        anchors=anchors,
        box_format="midpoint",
        device=DEVICE,
        desc=desc,
        max_eval_batches=max_eval_batches,
        max_detections=max_detections,
    )

    mAP = mean_average_precision(
        pred_boxes,
        target_boxes,
        iou_threshold=iou_threshold,
        box_format="midpoint",
        num_classes=num_classes,
    )

    return mAP, len(pred_boxes), len(target_boxes)


def load_resume_checkpoint(model, optimizer, scaler):
    """Load the latest checkpoint if AUTO_RESUME is enabled."""

    if not AUTO_RESUME:
        return 0, 0.0, False

    if not os.path.exists(LAST_MODEL_FILE):
        return 0, 0.0, False

    checkpoint = torch.load(LAST_MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    last_epoch = int(checkpoint.get("epoch", 0))
    best_mAP = float(checkpoint.get("best_mAP", 0.0))

    return last_epoch, best_mAP, True


def save_training_checkpoint(model, optimizer, scaler, epoch, best_mAP, filename):
    """Save a checkpoint that can be resumed later."""

    save_checkpoint(
        {
            "model_name": MODEL_NAME,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_mAP": best_mAP,
            "image_size": IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
        },
        filename=filename,
    )


def write_report(
    report_path,
    train_stats,
    test_stats,
    start_epoch,
    end_epoch,
    final_epoch,
    best_epoch,
    best_mAP,
    final_train_loss,
    final_train_mAP,
    final_test_mAP,
    resumed,
    elapsed_sec,
):
    """Write an English experiment report."""

    lines = []
    lines.append("YOLOv5nano Experiment Report")
    lines.append("=" * 32)
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("1. Experiment Configuration")
    lines.append("---------------------------")
    lines.append(f"Model name: {MODEL_NAME}")
    lines.append("Implementation: Custom PyTorch YOLOv5-style nano detector; Ultralytics is not used.")
    lines.append(f"Input image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    lines.append(f"Number of classes: {NUM_CLASSES}")
    lines.append(f"Batch size: {BATCH_SIZE}")
    lines.append(f"Learning rate: {LEARNING_RATE}")
    lines.append(f"Weight decay: {WEIGHT_DECAY}")
    lines.append(f"Optimizer: Adam")
    lines.append(f"Mixed precision training: {USE_AMP and DEVICE.startswith('cuda')}")
    lines.append(f"Device: {DEVICE}")
    lines.append(f"Session start epoch: {start_epoch}")
    lines.append(f"Session end epoch: {end_epoch}")
    lines.append(f"Training was resumed from checkpoint: {resumed}")
    lines.append("")

    lines.append("2. Dataset Summary")
    lines.append("------------------")
    lines.append(f"Training CSV: {TRAIN_CSV}")
    lines.append(f"Testing CSV: {TEST_CSV}")
    lines.append(f"Training images: {train_stats['image_count']}")
    lines.append(f"Training boxes: {train_stats['total_boxes']}")
    lines.append(f"Testing images: {test_stats['image_count']}")
    lines.append(f"Testing boxes: {test_stats['total_boxes']}")
    lines.append(f"Dataset record CSV: {DATASET_RECORD_CSV}")
    lines.append("")

    lines.append("3. Evaluation Protocol")
    lines.append("----------------------")
    lines.append(f"Evaluation frequency: every {EVAL_EVERY} epochs starting at epoch {EVAL_START_EPOCH}")
    lines.append(f"mAP IoU threshold: {MAP_IOU_THRESHOLD}")
    lines.append(f"NMS IoU threshold: {NMS_IOU_THRESHOLD}")
    lines.append(f"Detection confidence threshold for mAP collection: {MAP_CONF_THRESHOLD}")
    lines.append(f"Maximum detections per image: {MAX_DETECTIONS_PER_IMAGE}")
    lines.append("")

    lines.append("4. Final Results")
    lines.append("----------------")
    lines.append(f"Final epoch completed: {final_epoch}")
    lines.append(f"Final training loss: {final_train_loss if final_train_loss is not None else 'N/A'}")
    lines.append(f"Final training mAP@0.5: {final_train_mAP if final_train_mAP is not None else 'N/A'}")
    lines.append(f"Final testing mAP@0.5: {final_test_mAP if final_test_mAP is not None else 'N/A'}")
    lines.append(f"Best testing mAP@0.5: {best_mAP:.6f}")
    lines.append(f"Best epoch: {best_epoch if best_epoch is not None else 'N/A'}")
    lines.append(f"Elapsed time for this session: {elapsed_sec:.2f} seconds")
    lines.append("")

    lines.append("5. Output Files")
    lines.append("---------------")
    lines.append(f"Last checkpoint: {LAST_MODEL_FILE}")
    lines.append(f"Best checkpoint: {BEST_MODEL_FILE}")
    lines.append(f"Training metrics CSV: {METRICS_CSV}")
    lines.append(f"Test metrics CSV: {TEST_METRICS_CSV}")
    lines.append(f"Experiment report: {REPORT_TXT}")
    lines.append("")

    lines.append("6. Notes")
    lines.append("--------")
    lines.append("The model is a YOLOv5-Nano-style architecture implemented directly in PyTorch.")
    lines.append("The checkpoint stores the model weights, optimizer state, AMP scaler state, epoch number, and best mAP.")
    lines.append("To continue training, keep AUTO_RESUME=True and run this script again. If the last checkpoint was saved at epoch 100, the next session starts at epoch 101.")

    with open(report_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("Model:", MODEL_NAME)
    print("Device:", DEVICE)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            f"CSV files not found. Expected:\n  {TRAIN_CSV}\n  {TEST_CSV}"
        )

    train_stats = scan_dataset(TRAIN_CSV, LABEL_DIR, "train")
    test_stats = scan_dataset(TEST_CSV, LABEL_DIR, "test")
    write_dataset_record(train_stats, test_stats)

    print("Train images:", train_stats["image_count"], "Train boxes:", train_stats["total_boxes"])
    print("Test images:", test_stats["image_count"], "Test boxes:", test_stats["total_boxes"])
    print(f"Dataset record saved to {DATASET_RECORD_CSV}")

    model = YOLOv5Nano(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = YoloLoss(num_classes=NUM_CLASSES)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE.startswith("cuda"))

    last_epoch, best_mAP, resumed = load_resume_checkpoint(model, optimizer, scaler)
    start_epoch = last_epoch + 1
    end_epoch = last_epoch + SESSION_EPOCHS

    if resumed:
        print(f"Resumed from {LAST_MODEL_FILE}. Starting at epoch {start_epoch}.")
    else:
        print("No resume checkpoint found. Starting from epoch 1.")

    model.eval()
    with torch.inference_mode():
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        dummy_outputs = model(dummy_input)
        grid_sizes = tuple(output.shape[2] for output in dummy_outputs)
    model.train()

    print("Image size:", IMAGE_SIZE)
    print("Grid sizes:", grid_sizes)

    train_dataset = VOCDataset(
        csv_file=TRAIN_CSV,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        anchors=ANCHORS,
        image_size=IMAGE_SIZE,
        S=grid_sizes,
        C=NUM_CLASSES,
    )

    test_dataset = VOCDataset(
        csv_file=TEST_CSV,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        anchors=ANCHORS,
        image_size=IMAGE_SIZE,
        S=grid_sizes,
        C=NUM_CLASSES,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    train_eval_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    anchors = torch.tensor(ANCHORS, device=DEVICE)
    scaled_anchors = anchors * torch.tensor(grid_sizes, device=DEVICE).reshape(3, 1, 1)

    prepare_csv(
        METRICS_CSV,
        [
            "epoch",
            "train_loss",
            "positive_targets",
            "train_mAP",
            "test_mAP",
            "best_mAP",
            "learning_rate",
            "epoch_time_sec",
        ],
    )

    prepare_csv(
        TEST_METRICS_CSV,
        [
            "epoch",
            "test_mAP",
            "test_pred_boxes",
            "test_true_boxes",
            "map_iou_threshold",
            "nms_iou_threshold",
            "confidence_threshold",
            "max_detections_per_image",
        ],
    )

    session_start_time = time.time()
    final_epoch = last_epoch
    best_epoch = None
    final_train_loss = None
    final_train_mAP = None
    final_test_mAP = None

    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\nEpoch [{epoch}/{end_epoch}]")
        epoch_start_time = time.time()

        train_loss, positive_targets = train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaled_anchors=scaled_anchors,
            scaler=scaler,
            epoch=epoch,
        )

        final_epoch = epoch
        final_train_loss = train_loss
        train_mAP = None
        test_mAP = None

        if epoch >= EVAL_START_EPOCH and epoch % EVAL_EVERY == 0:
            train_mAP, train_pred_count, train_true_count = evaluate_map(
                loader=train_eval_loader,
                model=model,
                anchors=ANCHORS,
                iou_threshold=MAP_IOU_THRESHOLD,
                nms_iou_threshold=NMS_IOU_THRESHOLD,
                threshold=MAP_CONF_THRESHOLD,
                num_classes=NUM_CLASSES,
                desc=f"Evaluating train mAP epoch {epoch}",
                max_eval_batches=MAX_EVAL_BATCHES,
                max_detections=MAX_DETECTIONS_PER_IMAGE,
            )

            test_mAP, test_pred_count, test_true_count = evaluate_map(
                loader=test_loader,
                model=model,
                anchors=ANCHORS,
                iou_threshold=MAP_IOU_THRESHOLD,
                nms_iou_threshold=NMS_IOU_THRESHOLD,
                threshold=MAP_CONF_THRESHOLD,
                num_classes=NUM_CLASSES,
                desc=f"Evaluating test mAP epoch {epoch}",
                max_eval_batches=MAX_EVAL_BATCHES,
                max_detections=MAX_DETECTIONS_PER_IMAGE,
            )

            final_train_mAP = train_mAP
            final_test_mAP = test_mAP

            print(f"Train loss: {train_loss:.6f}")
            print(f"Positive targets this epoch: {positive_targets}")
            print(f"Train mAP:  {train_mAP:.6f} | pred boxes: {train_pred_count} | true boxes: {train_true_count}")
            print(f"Test mAP:   {test_mAP:.6f} | pred boxes: {test_pred_count} | true boxes: {test_true_count}")

            append_csv(
                TEST_METRICS_CSV,
                [
                    epoch,
                    f"{test_mAP:.6f}",
                    test_pred_count,
                    test_true_count,
                    MAP_IOU_THRESHOLD,
                    NMS_IOU_THRESHOLD,
                    MAP_CONF_THRESHOLD,
                    MAX_DETECTIONS_PER_IMAGE,
                ],
            )

            if test_mAP > best_mAP:
                best_mAP = test_mAP
                best_epoch = epoch
                save_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_mAP=best_mAP,
                    filename=BEST_MODEL_FILE,
                )
                print(f"New best test mAP: {best_mAP:.6f}. Model saved as {BEST_MODEL_FILE}")
        else:
            print(f"Train loss: {train_loss:.6f}")
            print(f"Positive targets this epoch: {positive_targets}")

        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_mAP=best_mAP,
            filename=LAST_MODEL_FILE,
        )

        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        append_csv(
            METRICS_CSV,
            [
                epoch,
                f"{train_loss:.6f}",
                positive_targets,
                "" if train_mAP is None else f"{train_mAP:.6f}",
                "" if test_mAP is None else f"{test_mAP:.6f}",
                f"{best_mAP:.6f}",
                f"{current_lr:.10f}",
                f"{epoch_time:.2f}",
            ],
        )

        print(f"Metrics appended to {METRICS_CSV}")
        if epoch >= EVAL_START_EPOCH and epoch % EVAL_EVERY == 0:
            print(f"Test metrics appended to {TEST_METRICS_CSV}")

    elapsed_sec = time.time() - session_start_time

    write_report(
        report_path=REPORT_TXT,
        train_stats=train_stats,
        test_stats=test_stats,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        final_epoch=final_epoch,
        best_epoch=best_epoch,
        best_mAP=best_mAP,
        final_train_loss=None if final_train_loss is None else f"{final_train_loss:.6f}",
        final_train_mAP=None if final_train_mAP is None else f"{final_train_mAP:.6f}",
        final_test_mAP=None if final_test_mAP is None else f"{final_test_mAP:.6f}",
        resumed=resumed,
        elapsed_sec=elapsed_sec,
    )

    print(f"Experiment report saved to {REPORT_TXT}")


if __name__ == "__main__":
    main()
