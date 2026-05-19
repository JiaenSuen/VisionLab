import torch
import torch.nn as nn
import torch.optim as optim


def get_model_family(model_name: str):
    name = model_name.lower()

    if "e-convnext" in name or "convnext" in name:
        return "convnext"

    if "rep_vit" in name or "vit" in name:
        return "vit"

    if "mamba" in name:
        return "mamba"

    if "resnet" in name or "resnext" in name or "wide-resnet" in name:
        return "resnet"

    if "inception" in name or "googlenet" in name or "xception" in name:
        return "inception"

    return "default"


def get_training_config(model_name: str, epochs: int):
    family = get_model_family(model_name)

    config = {
        "family": family,
        "epochs": epochs,
        "batch_size": 32,
        "test_batch_size": 64,
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.05,
        "label_smoothing": 0.0,
        "warmup_epochs": 5,
        "use_amp": True,
        "grad_clip": None,
    }

    if family == "resnet":
        config.update({
            "batch_size": 64,
            "test_batch_size": 128,
            "optimizer": "sgd",
            "lr": 0.05,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "warmup_epochs": 5,
            "grad_clip": None,
        })

    elif family == "convnext":
        config.update({
            "batch_size": 32,
            "test_batch_size": 64,
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.05,
            "label_smoothing": 0.1,
            "warmup_epochs": 5,
            "grad_clip": 1.0,
        })

    elif family == "vit":
        config.update({
            "batch_size": 32,
            "test_batch_size": 64,
            "optimizer": "adamw",
            "lr": 5e-4,
            "weight_decay": 0.05,
            "label_smoothing": 0.1,
            "warmup_epochs": 10,
            "grad_clip": 1.0,
        })

    elif family == "mamba":
        config.update({
            "batch_size": 8,
            "test_batch_size": 16,
            "optimizer": "adamw",
            "lr": 5e-4,
            "weight_decay": 0.05,
            "label_smoothing": 0.1,
            "warmup_epochs": 10,
            "grad_clip": 1.0,
        })

    elif family == "inception":
        config.update({
            "batch_size": 32,
            "test_batch_size": 64,
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "warmup_epochs": 5,
            "grad_clip": None,
        })

    return config


def build_criterion(config):
    return nn.CrossEntropyLoss(
        label_smoothing=config["label_smoothing"]
    )


def build_optimizer(model, config):
    if config["optimizer"] == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=0.9,
            weight_decay=config["weight_decay"],
            nesterov=True,
        )

    if config["optimizer"] == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    raise ValueError(f"Unsupported optimizer: {config['optimizer']}")


def build_scheduler(optimizer, config):
    epochs = config["epochs"]
    warmup_epochs = config["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )