#!/usr/bin/env bash

DATASET_DIR="./datasets/Sub-URMP"
WORKSPACE="./workspaces/pann_transfer"

CUDA_VISIBLE_DEVICES=1

python3 workspaces/pann_transfer/utils/plot_statistics.py plot --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --select=transfer_cnn14