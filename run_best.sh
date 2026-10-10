#!/bin/bash

set -euo pipefail

ENV_NAME="house-prices-ml"

echo "House Prices: Best Model Pipeline"
echo "Ridge + XGBoost 50/50 ensemble"
echo

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: Conda is not installed."
    echo "Install Miniconda or Anaconda and run the script again."
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating Conda environment: $ENV_NAME"
    conda env create -f environment.yml
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

if [[ ! -f "data/train.csv" || ! -f "data/test.csv" ]]; then
    echo "Error: dataset not found."
    echo
    echo "Expected files:"
    echo "  data/train.csv"
    echo "  data/test.csv"
    exit 1
fi

echo
echo "Training Ridge"
conda run -n "$ENV_NAME" \
    python -m classic_pipeline.main --model ridge

echo
echo "Training XGBoost"
conda run -n "$ENV_NAME" \
    python -m classic_pipeline.main --model xgboost

echo
echo "Building Ridge + XGBoost Blend"
conda run -n "$ENV_NAME" \
    python -m classic_pipeline.blend

echo
echo "Done"
echo
echo "Best model: 50/50 Ridge + XGBoost ensemble"
echo "Expected OOF RMSE: ~0.1097"
echo
echo "Final prediction file:"
echo "submissions/classic/blend_ridge_xgboost_0.50.csv"