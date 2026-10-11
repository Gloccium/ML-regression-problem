import numpy as np
import torch

from sklearn.model_selection import KFold

from deep_learning_pipeline.config import config
from deep_learning_pipeline.data import load_train_data, create_fold_dataloaders
from deep_learning_pipeline.model import MLPRegressor
from deep_learning_pipeline.train import train, validate
from deep_learning_pipeline.predict import predict
from deep_learning_pipeline.utils import (
    set_seed,
    get_device,
    get_loss,
    get_metric,
    get_optimizer,
    get_scheduler,
    log_experiment,
    save_fold_artifacts,
)


def fit(config):
    device = get_device(config)

    X, y = load_train_data(config)

    loss_func = get_loss(config)
    metric_func = get_metric(config)

    kfold = KFold(
        n_splits=config.data.n_splits,
        shuffle=True,
        random_state=config.general.seed,
    )

    fold_histories = []
    fold_metrics = []

    oof_predictions = np.zeros(len(X), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        print(f"\nFold {fold}/{config.data.n_splits}")

        set_seed(config.general.seed + fold - 1)

        (
            train_loader,
            val_loader,
            preprocessor,
            input_size,
            target_mean,
            target_std,
        ) = create_fold_dataloaders(
            X=X,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            config=config,
        )

        model = MLPRegressor(
            input_size=input_size,
            hidden_dims=config.model.hidden_dims,
            dropout=config.model.dropout,
        ).to(device)

        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        log_experiment(
            config=config,
            device=device,
            input_size=input_size,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_func=loss_func,
            metric_func=metric_func,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            target_mean=target_mean,
            target_std=target_std,
        )

        save_fold_artifacts(
            model=model,
            preprocessor=preprocessor,
            target_mean=target_mean,
            target_std=target_std,
            input_size=input_size,
            config=config,
            fold=fold,
        )

        val_predictions = predict(
            model=model,
            data_loader=val_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )

        oof_predictions[val_idx] = val_predictions

        val_loss, val_metric = validate(
            model=model,
            val_loader=val_loader,
            loss_func=loss_func,
            metric_func=metric_func,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )

        print(
            f"Fold {fold} result | "
            f"Val loss: {val_loss:.4f} | "
            f"Val {config.training.metric.upper()}: {val_metric:.4f}"
        )

        fold_histories.append(history)
        fold_metrics.append(val_metric)

        model.to("cpu")

    cv_mean = np.mean(fold_metrics)
    cv_std = np.std(fold_metrics)

    y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32)
    oof_tensor = torch.tensor(oof_predictions, dtype=torch.float32)

    oof_metric = metric_func(oof_tensor, y_tensor).item()

    print(
        f"\n{config.data.n_splits}-Fold "
        f"{config.training.metric.upper()}: "
        f"{cv_mean:.4f} ± {cv_std:.4f}"
    )

    print(
        f"OOF {config.training.metric.upper()}: "
        f"{oof_metric:.4f}"
    )

    return {
        "histories": fold_histories,
        "fold_metrics": fold_metrics,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "oof_metric": oof_metric,
        "oof_predictions": oof_predictions,
    }


if __name__ == "__main__":
    results = fit(config)