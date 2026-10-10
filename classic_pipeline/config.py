from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


config = OmegaConf.create({
    "general": {
        "experiment_name": "xgboost_tuned_2",
        "seed": 42
    },

    "paths": {
        "train_data": str(PROJECT_ROOT / "data" / "train.csv"),
        "test_data": str(PROJECT_ROOT / "data" / "test.csv"),
        "checkpoints": str(PROJECT_ROOT / "checkpoints" / "classic"),
    },

    "data": {
        "target": "SalePrice",
        "id_column": "Id",
        "target_transform": "log1p",
        "drop_outliers": True,
        "n_splits": 5,
    },

    "models": {
        "ridge": {
            "alpha": 10.0,
        },

        "random_forest": {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
        },

        "xgboost": {
            "n_estimators": 1200,
            "learning_rate": 0.025,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_lambda": 2.0,
        },
    },
})