import argparse
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from classic_pipeline.config import config
from classic_pipeline.data import load_train_data
from classic_pipeline.train import train_fold
from classic_pipeline.utils import save_experiment_config, save_fold_pipeline


def fit(config, model_name):
    X, y = load_train_data(config)

    kfold = KFold(
        n_splits=config.data.n_splits,
        shuffle=True,
        random_state=config.general.seed,
    )

    save_experiment_config(config)

    fold_metrics = []
    oof_predictions = np.zeros(len(X), dtype=np.float64)

    print(f"{model_name.upper()}")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        pipeline, val_predictions, val_rmse = train_fold(
            X=X,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            model_name=model_name,
            config=config,
        )

        oof_predictions[val_idx] = val_predictions
        fold_metrics.append(val_rmse)

        save_fold_pipeline(
            pipeline=pipeline,
            model_name=model_name,
            fold=fold,
            config=config,
        )

        print(f"Fold {fold}/{config.data.n_splits} | RMSE: {val_rmse:.4f}")

    cv_mean = np.mean(fold_metrics)
    cv_std = np.std(fold_metrics)
    oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))

    print(f"{config.data.n_splits}-Fold RMSE: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"OOF RMSE: {oof_rmse:.4f}")

    return {
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "oof_rmse": oof_rmse,
        "fold_metrics": fold_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["ridge", "random_forest", "xgboost"],
    )

    args = parser.parse_args()

    results = fit(config, args.model)