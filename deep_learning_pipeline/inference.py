import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf

from deep_learning_pipeline.data import (
    load_test_data,
    create_inference_loader,
)
from deep_learning_pipeline.model import MLPRegressor
from deep_learning_pipeline.predict import predict
from deep_learning_pipeline.utils import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_inference(experiment_name):
    experiment_dir = (
        PROJECT_ROOT
        / "checkpoints"
        / experiment_name
    )

    config_path = experiment_dir / "config.yaml"
    config = OmegaConf.load(config_path)
    config.paths.test_data = str(PROJECT_ROOT / "data" / "test.csv")

    device = get_device(config)

    X_test, test_ids = load_test_data(config)

    fold_predictions = []

    for fold in range(1, config.data.n_splits + 1):
        print(f"Inference fold {fold}/{config.data.n_splits}")

        fold_dir = experiment_dir / f"fold_{fold}"

        checkpoint = torch.load(
            fold_dir / "model.pt",
            map_location="cpu",
        )

        preprocessor = joblib.load(
            fold_dir / "preprocessor.joblib"
        )

        model = MLPRegressor(
            input_size=checkpoint["input_size"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(device)

        test_loader = create_inference_loader(
            X=X_test,
            preprocessor=preprocessor,
            batch_size=config.training.batch_size,
        )

        predictions = predict(
            model=model,
            data_loader=test_loader,
            device=device,
            target_mean=checkpoint["target_mean"],
            target_std=checkpoint["target_std"],
        )

        fold_predictions.append(predictions)

    fold_predictions = np.stack(fold_predictions)

    ensemble_predictions = np.mean(
        fold_predictions,
        axis=0,
    )

    # log1p(SalePrice) → SalePrice
    sale_prices = np.expm1(
        ensemble_predictions
    )

    submission = pd.DataFrame({
        config.data.id_column: test_ids.to_numpy(),
        config.data.target: sale_prices,
    })

    submission_dir = PROJECT_ROOT / "submissions"
    submission_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission_path = (
            submission_dir
            / f"{experiment_name}_inference.csv"
    )

    submission.to_csv(
        submission_path,
        index=False,
    )

    print()
    print(f"Submission saved to: {submission_path}")

    return submission_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        required=True,
        help="Saved experiment name"
    )

    args = parser.parse_args()

    run_inference(args.experiment)


