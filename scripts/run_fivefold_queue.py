#!/usr/bin/env python3
"""Run assigned nnU-Net folds sequentially with a conservative thermal pause.

Run one queue per GPU host. Jobs are resumable: completed folds are skipped,
and incomplete folds with a latest checkpoint use nnU-Net's --c option.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = {
    "501": "Dataset501_ISLES22_DWI",
    "502": "Dataset502_ISLES22_DWI_ADC",
    "503": "Dataset503_ISLES22_DWI_FLAIR",
    "504": "Dataset504_ISLES22_DWI_ADC_FLAIR",
}
TRAINER_DIR = "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres_common"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gpu_status() -> tuple[int, int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    lines = completed.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("Expected one GPU")
    temperature, memory = [int(value.strip()) for value in lines[0].split(",")]
    return temperature, memory


def write_status(path: Path, **fields: object) -> None:
    payload = {"updated_utc": now(), **fields}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--max-temperature", type=int, default=78)
    parser.add_argument("--resume-temperature", type=int, default=68)
    parser.add_argument("--max-pause-minutes", type=int, default=30)
    args = parser.parse_args()
    if not args.folds or len(set(args.folds)) != len(args.folds) or any(fold not in range(1, 5) for fold in args.folds):
        parser.error("Folds must be unique values from 1 to 4")
    if not 0 < args.resume_temperature < args.max_temperature < 90:
        parser.error("Expected 0 < resume temperature < maximum temperature < 90 C")

    preprocessed = PROJECT / "nnunet_data/preprocessed"
    expected_split = json.loads((preprocessed / DATASETS["501"] / "splits_final.json").read_text())
    if len(expected_split) != 5:
        raise ValueError("The shared five-fold split must be installed first")
    for dataset in DATASETS.values():
        split = json.loads((preprocessed / dataset / "splits_final.json").read_text())
        if split != expected_split:
            raise ValueError(f"Different folds in {dataset}")

    results = PROJECT / "nnunet_data/results"
    logs = PROJECT / "logs"
    logs.mkdir(exist_ok=True)
    status_file = logs / "cv_queue_status.json"
    train_bin = Path(sys.executable).with_name("nnUNetv2_train")
    if not train_bin.is_file():
        raise FileNotFoundError(train_bin)
    environment = os.environ.copy()
    environment.update(
        {
            "nnUNet_raw": str(PROJECT / "nnunet_data/raw"),
            "nnUNet_preprocessed": str(preprocessed),
            "nnUNet_results": str(results),
            "nnUNet_n_proc_DA": "4",
            "nnUNet_compile": "false",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONNOUSERSITE": "1",
        }
    )
    jobs = [(fold, dataset_id) for fold in args.folds for dataset_id in args.datasets]
    for job_index, (fold, dataset_id) in enumerate(jobs, 1):
        dataset = DATASETS[dataset_id]
        output = results / dataset / TRAINER_DIR / f"fold_{fold}"
        final = output / "checkpoint_final.pth"
        summary = output / "validation/summary.json"
        if final.is_file() and summary.is_file():
            print(f"Skipping completed fold {fold}, dataset {dataset_id}", flush=True)
            continue
        temperature, memory = gpu_status()
        if memory > 1024:
            raise RuntimeError(f"GPU busy before fold {fold}, dataset {dataset_id}: {memory} MiB")
        while temperature >= args.max_temperature:
            print(f"GPU at {temperature} C before launch; waiting to cool", flush=True)
            time.sleep(30)
            temperature, memory = gpu_status()

        command = [
            str(train_bin), dataset_id, "3d_fullres_common", str(fold),
            "-tr", "nnUNetTrainer_250epochs", "--npz",
        ]
        if (output / "checkpoint_latest.pth").is_file():
            command.append("--c")
        elif final.is_file() and not summary.is_file():
            command.extend(["--val", "--npz"])
        log_path = logs / f"cv_fold{fold}_dataset{dataset_id}.log"
        write_status(
            status_file, state="running", job=job_index, total_jobs=len(jobs),
            fold=fold, dataset=dataset_id, temperature_c=temperature, log=str(log_path),
        )
        print(f"Starting fold {fold}, dataset {dataset_id} at {now()}", flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== Queue launch {now()} ===\n")
            handle.flush()
            process = subprocess.Popen(
                command, cwd=PROJECT, env=environment, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            paused_at: float | None = None
            while process.poll() is None:
                time.sleep(10)
                temperature, memory = gpu_status()
                if paused_at is None and temperature >= args.max_temperature:
                    os.killpg(process.pid, signal.SIGSTOP)
                    paused_at = time.monotonic()
                    print(f"Paused hot GPU at {temperature} C: fold {fold}, dataset {dataset_id}", flush=True)
                elif paused_at is not None and temperature <= args.resume_temperature:
                    os.killpg(process.pid, signal.SIGCONT)
                    paused_at = None
                    print(f"Resumed after cooling to {temperature} C", flush=True)
                elif paused_at is not None and time.monotonic() - paused_at > args.max_pause_minutes * 60:
                    os.killpg(process.pid, signal.SIGCONT)
                    os.killpg(process.pid, signal.SIGTERM)
                    write_status(status_file, state="failed_overheat", fold=fold, dataset=dataset_id, temperature_c=temperature)
                    raise RuntimeError("GPU did not cool sufficiently during thermal pause")
                write_status(
                    status_file, state="paused_hot" if paused_at is not None else "running",
                    job=job_index, total_jobs=len(jobs), fold=fold, dataset=dataset_id,
                    temperature_c=temperature, gpu_memory_mib=memory, log=str(log_path),
                )
            return_code = process.returncode
        if return_code != 0 or not final.is_file() or not summary.is_file():
            write_status(status_file, state="failed", fold=fold, dataset=dataset_id, return_code=return_code, log=str(log_path))
            raise RuntimeError(f"Training or validation failed: fold {fold}, dataset {dataset_id}; see {log_path}")
        print(f"Completed fold {fold}, dataset {dataset_id} at {now()}", flush=True)
        write_status(status_file, state="between_jobs", completed_job=job_index, total_jobs=len(jobs))
    write_status(status_file, state="complete", completed_jobs=len(jobs), total_jobs=len(jobs))
    print("All assigned folds complete", flush=True)


if __name__ == "__main__":
    main()
