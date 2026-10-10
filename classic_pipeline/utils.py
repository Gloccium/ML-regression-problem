from pathlib import Path

import joblib
from omegaconf import OmegaConf


def save_fold_pipeline(pipeline, model_name, fold, config):
    experiment_dir = Path(config.paths.checkpoints) / config.general.experiment_name
    model_dir = experiment_dir / model_name

    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = model_dir / f"fold_{fold}.joblib"
    joblib.dump(pipeline, pipeline_path)


def save_experiment_config(config):
    experiment_dir = Path(config.paths.checkpoints) / config.general.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config=config, f=experiment_dir / "config.yaml")