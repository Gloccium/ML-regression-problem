import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from classic_pipeline.data import load_test_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_inference(experiment_name, model_name):
    experiment_dir = PROJECT_ROOT / "checkpoints" / "classic" / experiment_name
    config = OmegaConf.load(experiment_dir / "config.yaml")

    config.paths.test_data = str(PROJECT_ROOT / "data" / "test.csv")

    X_test, test_ids = load_test_data(config)

    fold_predictions = []

    print(f"{model_name.upper()}")

    for fold in range(1, config.data.n_splits + 1):
        pipeline_path = experiment_dir / model_name / f"fold_{fold}.joblib"
        pipeline = joblib.load(pipeline_path)

        predictions = pipeline.predict(X_test)
        fold_predictions.append(predictions)

        print(f"Fold {fold}/{config.data.n_splits} completed")

    fold_predictions = np.stack(fold_predictions)
    ensemble_predictions = np.mean(fold_predictions, axis=0)

    sale_prices = np.expm1(ensemble_predictions)

    submission = pd.DataFrame({
        config.data.id_column: test_ids.to_numpy(),
        config.data.target: sale_prices,
    })

    submission_dir = PROJECT_ROOT / "submissions" / "classic"
    submission_dir.mkdir(parents=True, exist_ok=True)

    submission_path = submission_dir / f"{experiment_name}_{model_name}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")

    return submission_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=["ridge", "random_forest", "xgboost"],
    )

    args = parser.parse_args()

    run_inference(args.experiment, args.model)