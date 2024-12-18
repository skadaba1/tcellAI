#!/bin/bash

# Install required packages
echo "Installing required packages..."
pip install pandas tables anndata scanpy pyensembl biopython torch-geometric wandb gseapy networkx matplotlib

# Set up PyEnsembl
echo "Setting up PyEnsembl..."
pyensembl install --release 77 --species homo_sapiens

# Check if running in Google Colab
if python -c "import sys; print('google.colab' in sys.modules)" | grep -q "True"; then
    echo "Running on Google Colab"
    echo "Installing additional dependencies for Colab..."
    pip install -U scgpt
    pip install louvain

    echo "Downloading data and model checkpoints..."

    # Install gdown for downloading files from Google Drive
    pip install -q -U gdown

    # Define paths
    DATA_DIR="/content/data"
    MODEL_DIR_HUMAN="/content/save/scGPT_human"
    MODEL_DIR_CP="/content/save/scGPT_CP"

    # Create directories if they do not exist
    mkdir -p $DATA_DIR
    mkdir -p $MODEL_DIR_HUMAN
    mkdir -p $MODEL_DIR_CP

    # Download dataset
    if [ ! -f "$DATA_DIR/human_pancreas_norm_complexBatch.h5ad" ]; then
        echo "Downloading human pancreas dataset..."
        wget --content-disposition https://figshare.com/ndownloader/files/24539828 -O $DATA_DIR/human_pancreas_norm_complexBatch.h5ad
    fi

    # Download model checkpoints
    echo "Downloading model checkpoint for scGPT_CP..."
    gdown --folder "https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing" --output $MODEL_DIR_CP

    echo "Downloading model checkpoint for scGPT_human..."
    gdown --folder "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y" --output $MODEL_DIR_HUMAN
fi

echo "Environment setup complete."
