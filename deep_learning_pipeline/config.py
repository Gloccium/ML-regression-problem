from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

config = OmegaConf.create({
    "general": {
            "experiment_name": "mlp_baseline",
            "seed": 42,
    },

    "paths": {
        "train_data": str(PROJECT_ROOT / "data" / "train.csv"),
        "test_data": str(PROJECT_ROOT / "data" / "test.csv"),
        "checkpoints": str(PROJECT_ROOT / "checkpoints"),
        "submissions": str(PROJECT_ROOT / "submissions"),
    },

    "data": {
        "target": "SalePrice",
        "id_column": "Id",
        "target_transform": "log1p",
        "standardize_target": True,
        "drop_outliers": True,
        "n_splits": 5,
    },

    "model": {
        "hidden_dims": [256, 128, 64],
        "dropout": 0.15
    },

    "training": {
        "epochs": 50,
        "batch_size": 64,

        "loss": "mse",
        "metric": "rmse",

        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 1e-4,

        "scheduler": "reduce_on_plateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,

        "early_stopping_patience": 15,
        "device": "auto"
    },
})