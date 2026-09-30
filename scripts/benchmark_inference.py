#!/usr/bin/env python3
"""Measure one-run end-to-end nnU-Net inference time and PyTorch GPU peak.

This deliberately benchmarks the same ten fold-0 patients for every model,
using fold-0 final weights and nnU-Net's default test-time mirroring. Loading,
preprocessing, prediction and export are included in wall time. GPU memory is
the predictor process's peak PyTorch CUDA reserved memory, not whole-device use.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = {
    "501": ("DWI", "Dataset501_ISLES22_DWI", 1),
    "502": ("DWI+ADC", "Dataset502_ISLES22_DWI_ADC", 2),
    "503": ("DWI+FLAIR", "Dataset503_ISLES22_DWI_FLAIR", 2),
    "504": ("DWI+ADC+FLAIR", "Dataset504_ISLES22_DWI_ADC_FLAIR", 3),
}
PRIVATE = PROJECT / "reports/private_inference_benchmark"
PUBLIC = PROJECT / "reports/inference_benchmark_summary.json"


def gpu_used_mib() -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    )
    values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise ValueError("Expected exactly one GPU")
    return values[0]


def main() -> None:
    splits = json.loads((PROJECT / "splits/splits_master.json").read_text(encoding="utf-8"))
    subjects = sorted(splits[0]["val"])[:10]
    if len(subjects) != 10:
        raise ValueError("Expected ten benchmark cases")
    initial_gpu_used = gpu_used_mib()
    if initial_gpu_used > 512:
        raise RuntimeError(f"GPU is not idle ({initial_gpu_used} MiB); stop other GPU services first")
    predictor = PROJECT / "scripts/predict_with_memory.py"
    if not predictor.is_file():
        raise FileNotFoundError(predictor)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = PRIVATE / run_id
    run_dir.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "nnUNet_raw": str(PROJECT / "nnunet_data/raw"),
            "nnUNet_preprocessed": str(PROJECT / "nnunet_data/preprocessed"),
            "nnUNet_results": str(PROJECT / "nnunet_data/results"),
            "CUDA_VISIBLE_DEVICES": "0",
            "nnUNet_compile": "false",
            "PYTHONNOUSERSITE": "1",
        }
    )
    result: dict[str, object] = {
        "definition": "One run, ten common fold-0 cases per model; wall time includes model load, preprocessing, TTA prediction, and export",
        "torch_memory_definition": "Peak PyTorch CUDA reserved memory in predictor process; excludes other processes",
        "n_cases_per_model": len(subjects),
        "checkpoint": "checkpoint_final.pth",
        "fold": 0,
        "tta": "nnU-Net default mirroring enabled",
        "gpu_idle_baseline_mib": initial_gpu_used,
        "models": {},
    }
    for dataset_id, (name, dataset_dir, channels) in DATASETS.items():
        input_dir = run_dir / f"input_{dataset_id}"
        output_dir = run_dir / f"predictions_{dataset_id}"
        input_dir.mkdir()
        for subject in subjects:
            for channel in range(channels):
                filename = f"{subject}_{channel:04d}.nii.gz"
                source = PROJECT / "nnunet_data/raw" / dataset_dir / "imagesTr" / filename
                if not source.is_file():
                    raise FileNotFoundError(source)
                (input_dir / filename).symlink_to(source)
        command = [
            sys.executable, str(predictor), "-i", str(input_dir), "-o", str(output_dir),
            "-d", dataset_id, "-c", "3d_fullres_common", "-f", "0",
            "-tr", "nnUNetTrainer_250epochs", "-p", "nnUNetPlans",
            "-chk", "checkpoint_final.pth", "-npp", "1", "-nps", "1",
            "--disable_progress_bar",
        ]
        baseline = gpu_used_mib()
        if baseline > 512:
            raise RuntimeError(f"GPU became busy before {name}: {baseline} MiB")
        start = time.perf_counter()
        with (run_dir / f"predict_{dataset_id}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        if completed.returncode:
            raise RuntimeError(f"Inference failed for {name}; see {run_dir / f'predict_{dataset_id}.log'}")
        predicted = list(output_dir.glob("*.nii.gz"))
        if len(predicted) != len(subjects):
            raise RuntimeError(f"Expected {len(subjects)} predictions for {name}; got {len(predicted)}")
        marker_lines = [
            line for line in (run_dir / f"predict_{dataset_id}.log").read_text(encoding="utf-8").splitlines()
            if line.startswith("CUDA_PEAK_RESERVED_MIB=")
        ]
        if len(marker_lines) != 1:
            raise RuntimeError(f"Missing PyTorch memory peak for {name}")
        torch_peak_mib = float(marker_lines[0].split("=", 1)[1])
        result["models"][dataset_id] = {
            "input": name,
            "wall_seconds_total": round(elapsed, 3),
            "wall_seconds_per_case": round(elapsed / len(subjects), 3),
            "torch_peak_reserved_mib": torch_peak_mib,
        }
        PUBLIC.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{name}: {elapsed / len(subjects):.1f} s/case, {torch_peak_mib:.0f} MiB PyTorch GPU peak", flush=True)
    print(f"Saved aggregate benchmark: {PUBLIC}")


if __name__ == "__main__":
    main()
