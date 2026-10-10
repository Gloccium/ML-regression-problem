from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from classic_pipeline.config import config
from classic_pipeline.data import load_train_data, load_test_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def blend_models(config, experiment_name="classic_baselines", ridge_weight=0.5):
    X, y = load_train_data(config)
    X_test, test_ids = load_test_data(config)

    experiment_dir = PROJECT_ROOT / "checkpoints" / "classic" / experiment_name

    kfold = KFold(
        n_splits=config.data.n_splits,
        shuffle=True,
        random_state=config.general.seed,
    )

    ridge_oof = np.zeros(len(X))
    xgb_oof = np.zeros(len(X))

    ridge_test_predictions = []
    xgb_test_predictions = []

    for fold, (_, val_idx) in enumerate(kfold.split(X), start=1):
        ridge_pipeline = joblib.load(experiment_dir / "ridge" / f"fold_{fold}.joblib")
        xgb_pipeline = joblib.load(experiment_dir / "xgboost" / f"fold_{fold}.joblib")

        X_val = X.iloc[val_idx]

        ridge_oof[val_idx] = ridge_pipeline.predict(X_val)
        xgb_oof[val_idx] = xgb_pipeline.predict(X_val)

        ridge_test_predictions.append(ridge_pipeline.predict(X_test))
        xgb_test_predictions.append(xgb_pipeline.predict(X_test))

    xgb_weight = 1 - ridge_weight

    blend_oof = ridge_weight * ridge_oof + xgb_weight * xgb_oof
    blend_rmse = np.sqrt(mean_squared_error(y, blend_oof))

    ridge_test = np.mean(ridge_test_predictions, axis=0)
    xgb_test = np.mean(xgb_test_predictions, axis=0)

    blend_test = ridge_weight * ridge_test + xgb_weight * xgb_test
    sale_prices = np.expm1(blend_test)

    submission = pd.DataFrame({
        config.data.id_column: test_ids.to_numpy(),
        config.data.target: sale_prices,
    })

    submission_dir = PROJECT_ROOT / "submissions" / "classic"
    submission_dir.mkdir(parents=True, exist_ok=True)

    submission_path = submission_dir / f"blend_ridge_xgboost_{ridge_weight:.2f}.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Ridge weight: {ridge_weight:.2f}")
    print(f"XGBoost weight: {xgb_weight:.2f}")
    print(f"Blend OOF RMSE: {blend_rmse:.4f}")
    print(f"Submission saved to: {submission_path}")

    return blend_rmse


if __name__ == "__main__":
    blend_models(config)