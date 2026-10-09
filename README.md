# House Prices Regression

Reproducible machine learning pipeline for the Kaggle **House Prices - Advanced Regression Techniques** competition.

The project contains two separate approaches:

- `deep_learning_pipeline/` — PyTorch MLP pipeline
- `classic_pipeline/` — classic ML pipeline

## Deep Learning Pipeline

The current deep learning solution uses a fully connected neural network with 5-fold cross-validation.

### Pipeline

- EDA and outlier analysis
- semantic missing-value handling
- median imputation for numerical features
- most-frequent imputation for categorical features
- numerical feature standardization
- one-hot encoding
- `log1p(SalePrice)` target transformation
- fold-specific target standardization
- MLP: `301 → 256 → 128 → 64 → 1`
- BatchNorm + ReLU + Dropout
- AdamW optimizer
- ReduceLROnPlateau scheduler
- early stopping
- 5-fold cross-validation
- OOF evaluation
- 5-model ensemble for Kaggle test predictions
- saved preprocessing and model artifacts for reproducible inference

### Results

```text
5-Fold CV RMSE: 0.1189 ± 0.0080
OOF RMSE:       0.1192
Kaggle Public:  0.13091
```

## Project Structure

```text
ML-regression-problem/
├── classic_pipeline/
├── deep_learning_pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── main.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
├── notebooks/
│   └── EDA.ipynb
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── checkpoints/
├── submissions/
├── environment.yml
├── train.sh
└── README.md
```

`data/`, `checkpoints/` and `submissions/` are generated/local directories and are not stored in Git.

---

## Reproduce Training

### Requirements

You only need:

- Git
- Miniconda or Anaconda
- House Prices dataset

Clone the repository:

```bash
git clone https://github.com/Gloccium/ML-regression-problem.git
cd ML-regression-problem
```

Download the Kaggle **House Prices - Advanced Regression Techniques** dataset and place the files in:

```text
data/
├── train.csv
└── test.csv
```

Then run:

```bash
chmod +x train.sh
./train.sh
```

The script automatically:

1. checks that Conda is installed;
2. creates the `house-prices-ml` environment if it does not exist;
3. installs all required Python dependencies;
4. checks that the dataset is available;
5. runs the complete 5-fold cross-validation training pipeline;
6. saves the best model and preprocessing artifacts for every fold.

No manually created Python environment is required.

### Training Output

The trained artifacts are saved to:

```text
checkpoints/<experiment_name>/
```

For the current baseline:

```text
checkpoints/mlp_baseline/
├── config.yaml
├── fold_1/
│   ├── model.pt
│   └── preprocessor.joblib
├── fold_2/
│   ├── model.pt
│   └── preprocessor.joblib
├── fold_3/
│   ├── model.pt
│   └── preprocessor.joblib
├── fold_4/
│   ├── model.pt
│   └── preprocessor.joblib
└── fold_5/
    ├── model.pt
    └── preprocessor.joblib
```

Each fold contains:

- trained PyTorch model weights;
- fitted preprocessing pipeline;
- target normalization statistics;
- model architecture parameters.

The experiment config is also saved alongside the folds.

---

## Reproduce Inference

Inference can be run later **without retraining the models**.

First create the environment if it does not already exist:

```bash
conda env create -f environment.yml
```

Then run:

```bash
conda run -n house-prices-ml \
  python -m deep_learning_pipeline.inference \
  --experiment mlp_baseline
```

The inference pipeline:

1. loads `test.csv`;
2. loads all five fitted preprocessors;
3. loads all five trained PyTorch models;
4. restores fold-specific target normalization;
5. generates predictions from every fold;
6. averages the five predictions in log-space;
7. converts them back to `SalePrice`;
8. creates a Kaggle-compatible CSV file.

The generated file is saved to:

```text
submissions/mlp_baseline_inference.csv
```

The inference generated exclusively from saved artifacts reproduces the original Kaggle score:

```text
Kaggle Public Score: 0.13091
```

---

## Run Training Manually

Instead of using `train.sh`, the environment can also be created manually:

```bash
conda env create -f environment.yml
```

Then run training with:

```bash
conda run -n house-prices-ml \
  python -m deep_learning_pipeline.main
```

---

## Model

The current baseline architecture is:

```text
Input
  ↓
Linear(301 → 256)
BatchNorm
ReLU
Dropout
  ↓
Linear(256 → 128)
BatchNorm
ReLU
Dropout
  ↓
Linear(128 → 64)
BatchNorm
ReLU
Dropout
  ↓
Linear(64 → 1)
```

The actual input dimension is determined automatically by the fitted preprocessing pipeline.

Current configuration:

```text
Hidden layers: [256, 128, 64]
Dropout:       0.15
Optimizer:     AdamW
Scheduler:     ReduceLROnPlateau
Loss:          MSE
Metric:        RMSE
CV:            5 folds
```

## Notes

The neural network is intentionally kept relatively simple.

The goal of the project is not only to obtain a good Kaggle score, but to build a clean and reproducible end-to-end machine learning pipeline with:

- leakage-safe preprocessing;
- cross-validation;
- OOF evaluation;
- model ensembling;
- saved model artifacts;
- independent inference from saved weights.

The `classic_pipeline/` implementation will be used to compare the neural network against classical tabular machine learning approaches.