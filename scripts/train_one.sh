#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 DATASET_ID [FOLD]" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname "$script_dir")"
source "$project_dir/config/env.sh"
export CUDA_VISIBLE_DEVICES=0
# GTX 1070 (compute capability 6.1) cannot use nnU-Net's Triton-backed
# torch.compile path. Keep this disabled on both machines for a fair run.
export nnUNet_compile=false

dataset_id="$1"
trainer="nnUNetTrainer_250epochs"
configuration="3d_fullres_common"
fold="${2:-0}"
if [[ ! "$fold" =~ ^[0-4]$ ]]; then
    echo "FOLD must be an integer from 0 to 4" >&2
    exit 2
fi

train_bin="${NNUNET_TRAIN_BIN:-$(command -v nnUNetv2_train || true)}"
if [ -z "$train_bin" ]; then
    echo "nnUNetv2_train was not found. Activate the nnU-Net environment or set NNUNET_TRAIN_BIN." >&2
    exit 1
fi

exec "$train_bin" \
    "$dataset_id" "$configuration" "$fold" -tr "$trainer" --npz
