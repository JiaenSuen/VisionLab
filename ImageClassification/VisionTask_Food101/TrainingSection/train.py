import os
import json
import torch 
import torch.nn as nn

from tqdm import tqdm

from models.utils import check_accuracy, save_model, load_model
from models._modelRouter import modelRouter, Existing_model_names
from TrainingSection.dataset import Food101_224Dataset, NUM_CLASSES_OF_FOOD101, DATASET_NAME
from TrainingSection.training_recipe import (
    get_training_config,
    build_criterion,
    build_optimizer,
    build_scheduler,
)


def Train(modelName="", device="cuda", epochs=100):
    torch.cuda.empty_cache()

    modelName = modelName.lower()

    if modelName not in Existing_model_names:
        print(f"Model {modelName} not recognized. Available models are: {Existing_model_names}")
        return

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    config = get_training_config(modelName, epochs)

    print("-------------------------------------")
    print(f"Training model: {modelName}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Model family: {config['family']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"LR: {config['lr']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Label smoothing: {config['label_smoothing']}")
    print(f"AMP: {config['use_amp']}")
    print("-------------------------------------")

    # Model
    model = modelRouter[modelName](num_classes=NUM_CLASSES_OF_FOOD101)
    model.to(device)

    # Data Loader for training, with augmentation
    train_loader = Food101_224Dataset.GetTrainLoader(
        batch_size=config["batch_size"],
        augment=True,
    )

    # Data Loader for final train accuracy, without augmentation
    train_eval_loader = Food101_224Dataset.GetTrainLoader(
        batch_size=config["test_batch_size"],
        augment=False,
    )

    # Test / validation loader
    test_loader = Food101_224Dataset.GetTestLoader(
        batch_size=config["test_batch_size"],
    )

    # Loss / Optimizer / Scheduler
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(config["use_amp"] and device.type == "cuda")
    )

    best_acc = 0.0
    best_epoch = 0
    history = []

    best_model_path = f"trainedRelease/{DATASET_NAME}_{modelName}_best.pth"
    last_model_path = f"trainedRelease/{DATASET_NAME}_{modelName}_last.pth"
    log_path = f"trainedRelease/{DATASET_NAME}_{modelName}_history.json"

    for epoch in range(config["epochs"]):
        model.train()
        losses = []

        current_lr = optimizer.param_groups[0]["lr"]

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config['epochs']} | lr={current_lr:.6f}",
            unit="batch"
        )

        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=(config["use_amp"] and device.type == "cuda")
            ):
                scores = model(data)
                loss = criterion(scores, targets)

            scaler.scale(loss).backward()

            if config["grad_clip"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config["grad_clip"]
                )

            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)

            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        scheduler.step()

        train_loss = sum(losses) / len(losses)

        print(f"\nEpoch {epoch + 1}/{config['epochs']} completed")
        print(f"Average Loss: {train_loss:.4f}")

        # During training, only use test accuracy to select the best checkpoint.
        test_acc = check_accuracy(
            test_loader,
            model,
            calculate_profile=False
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "lr": current_lr,
        })

        # Save best model by test accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1

            save_model(model, best_model_path)

            print(
                f"New best model saved | "
                f"Epoch: {best_epoch} | "
                f"Best Test Acc: {best_acc:.2f}%"
            )

        print("-------------------------------------")

    # Save last model
    save_model(model, last_model_path)

    # ============================================================
    # Final evaluation using the best checkpoint
    # ============================================================

    print("\n=====================================")
    print("Final evaluation using best checkpoint")
    print("=====================================")

    best_model = modelRouter[modelName](num_classes=NUM_CLASSES_OF_FOOD101)
    best_model = load_model(
        best_model,
        best_model_path,
        device=device
    )

    print("\n[Best Checkpoint] Train Accuracy without augmentation")
    best_train_acc = check_accuracy(
        train_eval_loader,
        best_model,
        calculate_profile=False
    )

    print("\n[Best Checkpoint] Test Accuracy with profiling")
    best_test_acc = check_accuracy(
        test_loader,
        best_model,
        calculate_profile=True
    )

    final_report = {
        "best_epoch": best_epoch,
        "best_test_acc_during_training": best_acc,
        "best_checkpoint_train_acc_no_aug": best_train_acc,
        "best_checkpoint_test_acc": best_test_acc,
        "best_model_path": best_model_path,
        "last_model_path": last_model_path,
    }

    # Save training history
    os.makedirs("trainedRelease", exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": modelName,
                "dataset": DATASET_NAME,
                "config": config,
                "best_epoch": best_epoch,
                "best_acc": best_acc,
                "final_report": final_report,
                "history": history,
            },
            f,
            indent=4
        )

    print("\nTraining finished")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Test Accuracy During Training: {best_acc:.2f}%")
    print(f"Best Checkpoint Train Accuracy : {best_train_acc:.2f}%")
    print(f"Best Checkpoint Test  Accuracy : {best_test_acc:.2f}%")
    print(f"Best model path: {best_model_path}")
    print(f"Last model path: {last_model_path}")
    print(f"History path: {log_path}")