#!/bin/bash

set -euo pipefail

ENV_NAME="house-prices-ml"

echo "=== House Prices MLP Training ==="
echo

# Check Conda
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: Conda is not installed."
    echo "Install Miniconda or Anaconda and run this script again."
    exit 1
fi

# Create environment if necessary
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating Conda environment: $ENV_NAME"
    conda env create -f environment.yml
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

# Check dataset
if [ ! -f "data/train.csv" ]; then
    echo "Error: data/train.csv was not found."
    echo "Download the House Prices dataset and place train.csv in data/"
    exit 1
fi

if [ ! -f "data/test.csv" ]; then
    echo "Error: data/test.csv was not found."
    echo "Download the House Prices dataset and place test.csv in data/"
    exit 1
fi

echo
echo "Starting training..."
echo

conda run -n "$ENV_NAME" \
    python -m deep_learning_pipeline.main

echo
echo "Training completed successfully."
echo "Model artifacts are available in checkpoints/"