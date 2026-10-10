# House Prices Regression

Reproducible machine learning pipeline for the Kaggle **House Prices - Advanced Regression Techniques** competition.

The project compares two approaches:

- `deep_learning_pipeline/` — PyTorch MLP
- `classic_pipeline/` — Ridge, Random Forest and XGBoost

Both pipelines use the same core data preparation decisions, 5-fold cross-validation and OOF evaluation. Saved artifacts can be loaded later for inference without retraining.

## Results

| Model | 5-Fold CV RMSE | OOF RMSE | Kaggle Public RMSE |
|---|---:|---:|---:|
| Ridge | 0.1150 ± 0.0082 | 0.1153 | 0.13367 |
| Random Forest | 0.1390 ± 0.0092 | 0.1393 | — |
| XGBoost | 0.1150 ± 0.0073 | 0.1152 | 0.12992 |
| MLP | 0.1189 ± 0.0080 | 0.1192 | 0.13091 |
| **Ridge + XGBoost 50/50 blend** | — | **0.1097** | **0.12741** |

The best final solution is the **50/50 Ridge + XGBoost ensemble**.

---

## Quick Start — Reproduce Best Model

The best solution in this project is a **50/50 ensemble of Ridge and XGBoost**.

### Requirements

- Git
- Miniconda or Anaconda
- Kaggle House Prices dataset

Clone the repository:

```bash
git clone https://github.com/Gloccium/ML-regression-problem.git
cd ML-regression-problem
```

Place the dataset files in:

```text
data/
├── train.csv
└── test.csv
```

Then run:

```bash
chmod +x run_best.sh
./run_best.sh
```

The script automatically:

1. checks that Conda is installed;
2. creates the `house-prices-ml` environment if necessary;
3. trains Ridge using 5-fold cross-validation;
4. trains XGBoost using the same 5 folds;
5. saves all fitted fold pipelines;
6. creates a 50/50 Ridge + XGBoost ensemble;
7. evaluates the ensemble using OOF predictions;
8. generates the final test predictions.

Expected result:

```text
Ridge OOF RMSE:          ~0.1153
XGBoost OOF RMSE:        ~0.1152
Ridge + XGBoost OOF:     ~0.1097
Kaggle Public RMSE:       0.12741
```

The final prediction file is saved to:

```text
submissions/classic/blend_ridge_xgboost_0.50.csv
```

The script does not submit anything to Kaggle.

## Project Structure

```text
ML-regression-problem/
├── classic_pipeline/
│   ├── __init__.py
│   ├── blend.py
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── main.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
│
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
├── run_best.sh
├── train.sh
└── README.md
```

`data/`, `checkpoints/` and `submissions/` are local/generated directories and are not stored in Git.

---

## Data Preparation

The main preprocessing decisions are shared between the deep learning and classical pipelines:

- EDA and outlier analysis
- removal of two anomalous observations with very large `GrLivArea` and unusually low `SalePrice`
- semantic missing values such as missing garage/basement/pool features filled with `"None"`
- median imputation for numerical features
- most-frequent imputation for remaining categorical features
- one-hot encoding for categorical features
- numerical feature standardization
- `log1p(SalePrice)` target transformation
- leakage-safe preprocessing fitted separately inside every CV fold

The MLP additionally uses fold-specific target standardization.

---

# Setup

## Requirements

You need:

- Git
- Miniconda or Anaconda
- the Kaggle House Prices dataset

Clone the repository:

```bash
git clone https://github.com/Gloccium/ML-regression-problem.git
cd ML-regression-problem
```

Place the dataset files in:

```text
data/
├── train.csv
└── test.csv
```

Create the environment:

```bash
conda env create -f environment.yml
```

The environment is named:

```text
house-prices-ml
```

Commands below use `conda run`, so manually activating the environment is not required.

---

# Deep Learning Pipeline

## Model

The PyTorch baseline is a fully connected neural network:

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

Main configuration:

```text
Hidden layers: [256, 128, 64]
Dropout:       0.15
Optimizer:     AdamW
Scheduler:     ReduceLROnPlateau
Loss:          MSE
Metric:        RMSE
CV:            5 folds
```

Training includes:

- 5-fold cross-validation
- fold-specific preprocessing
- fold-specific target standardization
- AdamW optimization
- ReduceLROnPlateau scheduling
- early stopping
- OOF evaluation
- best-model restoration
- saved preprocessing and model artifacts for every fold

## Train the MLP

The simplest option is:

```bash
chmod +x train.sh
./train.sh
```

`train.sh`:

1. checks that Conda is installed;
2. creates the `house-prices-ml` environment if it does not exist;
3. checks that the dataset is available;
4. runs the complete MLP training pipeline;
5. saves the trained fold artifacts.

The same training can be launched manually with:

```bash
conda run -n house-prices-ml \
  python -m deep_learning_pipeline.main
```

Artifacts are saved to:

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

Each fold stores the trained PyTorch weights together with the preprocessing and target-normalization information required for reproducible inference.

## MLP Inference

Inference can be reproduced later without retraining:

```bash
conda run -n house-prices-ml \
  python -m deep_learning_pipeline.inference \
  --experiment mlp_baseline
```

The inference pipeline:

1. loads `test.csv`;
2. loads all five saved preprocessors;
3. loads all five trained MLPs;
4. restores fold-specific target normalization;
5. generates predictions from all folds;
6. averages predictions in `log1p(SalePrice)` space;
7. applies `expm1`;
8. creates a Kaggle-compatible CSV.

Output:

```text
submissions/mlp_baseline_inference.csv
```

Reproducing inference exclusively from saved artifacts gives the same Kaggle Public score as the original training run:

```text
0.13091
```

---

# Classic ML Pipeline

The classical pipeline uses the same 5 folds and comparable preprocessing to evaluate three different model families:

- **Ridge** — regularized linear regression
- **Random Forest** — bagging ensemble of decision trees
- **XGBoost** — gradient boosting on decision trees

For every fold, preprocessing and the fitted estimator are stored together as a single sklearn `Pipeline`.

Example:

```text
checkpoints/classic/classic_baselines/
├── config.yaml
├── ridge/
│   ├── fold_1.joblib
│   ├── fold_2.joblib
│   ├── fold_3.joblib
│   ├── fold_4.joblib
│   └── fold_5.joblib
├── random_forest/
│   └── ...
└── xgboost/
    └── ...
```

Each `.joblib` file contains both:

```text
fitted preprocessing
+
trained model
```

## Train Ridge

```bash
conda run -n house-prices-ml \
  python -m classic_pipeline.main \
  --model ridge
```

Result:

```text
5-Fold RMSE: 0.1150 ± 0.0082
OOF RMSE:    0.1153
```

## Train Random Forest

```bash
conda run -n house-prices-ml \
  python -m classic_pipeline.main \
  --model random_forest
```

Result:

```text
5-Fold RMSE: 0.1390 ± 0.0092
OOF RMSE:    0.1393
```

## Train XGBoost

```bash
conda run -n house-prices-ml \
  python -m classic_pipeline.main \
  --model xgboost
```

Result:

```text
5-Fold RMSE: 0.1150 ± 0.0073
OOF RMSE:    0.1152
```

---

## Classic Model Inference

Inference loads the already fitted fold pipelines and produces an ensemble prediction over the Kaggle test set.

### Ridge

```bash
conda run -n house-prices-ml \
  python -m classic_pipeline.inference \
  --experiment classic_baselines \
  --model ridge
```

Output:

```text
submissions/classic/classic_baselines_ridge.csv
```

### Random Forest

```bash
conda run -n house-prices-ml \
  python -m classic_pipeline.inference \
  --experiment classic_baselines \
  --model random_forest
```

Output:

```text
submissions/classic/classic_baselines_random_forest.csv
```

### XGBoost

```bash
conda run -n house-prices-ml \
  python -m classic_pipeline.inference \
  --experiment classic_baselines \
  --model xgboost
```

Output:

```text
submissions/classic/classic_baselines_xgboost.csv
```

Inference does **not** retrain the models and does **not** upload anything to Kaggle. It only loads saved artifacts, calculates predictions and writes a local CSV file.

---

# Ridge + XGBoost Blend

Ridge and XGBoost achieved almost identical OOF scores while representing very different model families.

Their prediction errors are therefore partially complementary.

The final solution averages their predictions in `log1p(SalePrice)` space:

```text
50% Ridge
+
50% XGBoost
=
final prediction
```

Run the blend with:

```bash
conda run -n house-prices-ml \
  python -m classic_pipeline.blend
```

The script:

1. loads the saved Ridge fold pipelines;
2. loads the saved XGBoost fold pipelines;
3. reconstructs OOF predictions;
4. evaluates the 50/50 OOF blend;
5. performs inference on `test.csv`;
6. averages Ridge and XGBoost predictions in log-space;
7. applies `expm1`;
8. creates the final submission CSV.

Result:

```text
Blend OOF RMSE: 0.1097
```

Output:

```text
submissions/classic/blend_ridge_xgboost_0.50.csv
```

Kaggle Public RMSE:

```text
0.12741
```

This was the best result among all tested approaches.

---

# Kaggle Results

Final leaderboard comparison:

| Model | Kaggle Public RMSE |
|---|---:|
| Ridge | 0.13367 |
| MLP | 0.13091 |
| XGBoost | 0.12992 |
| **Ridge + XGBoost 50/50** | **0.12741** |

The Random Forest model was kept as a local baseline and was not submitted because its OOF score was substantially worse.

---