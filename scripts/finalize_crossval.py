#!/usr/bin/env python3
"""Verify the 20 completed fold outputs and evaluate all five folds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = (
    "Dataset501_ISLES22_DWI",
    "Dataset502_ISLES22_DWI_ADC",
    "Dataset503_ISLES22_DWI_FLAIR",
    "Dataset504_ISLES22_DWI_ADC_FLAIR",
)
TRAINER = "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres_common"


def main() -> None:
    for dataset in DATASETS:
        for fold in range(5):
            relative = Path("nnunet_data/results") / dataset / TRAINER / f"fold_{fold}"
            destination = PROJECT / relative
            if not (destination / "checkpoint_final.pth").is_file() or not (destination / "validation/summary.json").is_file():
                raise RuntimeError(f"Incomplete fold result: {destination}")
            print(f"Verified {dataset}, fold {fold}", flush=True)
    subprocess.run([sys.executable, str(PROJECT / "scripts/evaluate_crossval.py")], cwd=PROJECT, check=True)
    print("Five-fold evaluation complete", flush=True)


if __name__ == "__main__":
    main()
