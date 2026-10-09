import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(config):
    device = config.training.device.lower()

    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    return torch.device(device)


def get_loss(config):
    loss_name = config.training.loss.lower()

    if loss_name == "mse":
        return nn.MSELoss()

    if loss_name == "mae":
        return nn.L1Loss()

    raise ValueError(f"Unknown loss: {loss_name}")


def get_optimizer(model, config):
    optimizer_name = config.training.optimizer.lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, config):
    scheduler_name = config.training.scheduler.lower()

    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
        )

    if scheduler_name == "none":
        return None

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_metric(config):
    metric_name = config.training.metric.lower()

    if metric_name == "rmse":
        def rmse(predictions, targets):
            return torch.sqrt(torch.mean((predictions - targets) ** 2))
        return rmse

    if metric_name == "mse":
        def mse(predictions, targets):
            return torch.mean((predictions - targets) ** 2)
        return mse

    if metric_name == "mae":
        def mae(predictions, targets):
            return torch.mean(torch.abs(predictions - targets))
        return mae

    raise ValueError(f"Unknown metric: {metric_name}")


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    torch.save(checkpoint, path)


def log_experiment(config, device, input_size, loss_func, optimizer, scheduler):
    print(f"Experiment: {config.general.experiment_name}")
    print(f"Device: {device}")
    print(f"Input size: {input_size}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Loss: {loss_func.__class__.__name__}")
    print(f"Metric: {config.training.metric.upper()}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Scheduler: {scheduler.__class__.__name__}")
    print()