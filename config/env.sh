#!/usr/bin/env bash

config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ISLES22_PROJECT="${ISLES22_PROJECT:-$(dirname "$config_dir")}"
export ISLES22_DATA="${ISLES22_DATA:-$HOME/datasets/ISLES-2022/extracted/ISLES-2022}"
export nnUNet_raw="$ISLES22_PROJECT/nnunet_data/raw"
export nnUNet_preprocessed="$ISLES22_PROJECT/nnunet_data/preprocessed"
export nnUNet_results="$ISLES22_PROJECT/nnunet_data/results"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"
export PYTHONNOUSERSITE=1
