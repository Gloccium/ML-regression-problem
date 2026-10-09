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
    create_submission,
)


def fit(config):
    device = get_device(config)

    X, y, X_test, test_ids = load_train_data(config)

    loss_func = get_loss(config)
    metric_func = get_metric(config)

    kfold = KFold(
        n_splits=config.data.n_splits,
        shuffle=True,
        random_state=config.general.seed
    )

    fold_models = []
    fold_histories = []
    fold_preprocessors = []
    fold_metrics = []

    oof_predictions = np.zeros(len(X), dtype=np.float32)

    test_predictions = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        print(f"Fold {fold}/{config.data.n_splits}")

        set_seed(config.general.seed + fold - 1)

        (
            train_loader,
            val_loader,
            test_loader,
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

        model = MLPRegressor(input_size=input_size,
                             hidden_dims=config.model.hidden_dims,
                             dropout=config.model.dropout,).to(device)
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        log_experiment(config=config,
                       device=device,
                       input_size=input_size,
                       loss_func=loss_func,
                       optimizer=optimizer,
                       scheduler=scheduler)

        history = train(model=model,
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

        print(f"Fold {fold} result | "
              f"Val loss: {val_loss:.4f} | "
              f"Val {config.training.metric.upper()}: {val_metric:.4f}")

        fold_metrics.append(val_metric)
        fold_histories.append(history)
        fold_preprocessors.append(preprocessor)

        fold_models.append(model.to("cpu"))

    mean_metric = np.mean(fold_metrics)
    std_metric = np.std(fold_metrics)

    y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32)
    oof_tensor = torch.tensor(oof_predictions, dtype=torch.float32)

    oof_metric = metric_func(oof_tensor, y_tensor).item()

    test_predictions = np.stack(test_predictions)
    ensemble_test_predictions = np.mean(test_predictions, axis=0)


    print(f"{config.data.n_splits}-Fold "
          f"{config.training.metric.upper()}: {mean_metric:.4f} ± {std_metric:.4f}")

    print(f"OOF {config.training.metric.upper()}: {oof_metric:.4f}")

    return {
        "models": fold_models,
        "histories": fold_histories,
        "preprocessors": fold_preprocessors,
        "fold_metrics": fold_metrics,

        "cv_mean": mean_metric,
        "cv_std": std_metric,
        "oof_metric": oof_metric,
        "oof_predictions": oof_predictions,
    }


if __name__ == "__main__":
    results = fit(config)

    submission_path = create_submission(
        test_ids=results["test_ids"],
        test_predictions=results["test_predictions"],
        config=config,
    )