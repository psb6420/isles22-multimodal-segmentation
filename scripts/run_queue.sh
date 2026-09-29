#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 DATASET_ID [DATASET_ID ...]" >&2
    echo "Example: $0 501 504" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname "$script_dir")"
mkdir -p "$project_dir/logs"

for dataset_id in "$@"; do
    log="$project_dir/logs/train_${dataset_id}.log"
    echo "[$(date -Iseconds)] Starting Dataset${dataset_id}; log: $log"
    bash "$script_dir/train_one.sh" "$dataset_id" 2>&1 | tee "$log"
    echo "[$(date -Iseconds)] Finished Dataset${dataset_id}"
done
