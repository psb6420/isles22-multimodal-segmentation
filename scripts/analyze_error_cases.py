#!/usr/bin/env python3
"""Summarize paired errors and draw local-only MRI prediction montages.

The figures contain patient MR images and IDs. They are deliberately written
only to the git-ignored private_error_analysis directory.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "nnunet_data/results"
RAW = PROJECT / "nnunet_data/raw"
TRAINER = "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres_common"
DATASETS = {
    "501": "Dataset501_ISLES22_DWI",
    "502": "Dataset502_ISLES22_DWI_ADC",
    "503": "Dataset503_ISLES22_DWI_FLAIR",
    "504": "Dataset504_ISLES22_DWI_ADC_FLAIR",
}
PAIRS = {"DWI_plus_FLAIR": ("503", "501"), "DWI_ADC_plus_FLAIR": ("504", "502")}
PRIVATE = PROJECT / "reports/private_error_analysis"


def mask(dataset_id: str, subject: str) -> np.ndarray:
    location = RESULTS / DATASETS[dataset_id] / TRAINER / "fold_0/validation" / f"{subject}.nii.gz"
    return np.asarray(nib.load(location).dataobj, dtype=bool)


def overlay(ax: plt.Axes, base: np.ndarray, reference: np.ndarray, prediction: np.ndarray | None, z: int) -> None:
    display = np.rot90(base[:, :, z])
    ax.imshow(display, cmap="gray", vmin=np.percentile(display, 1), vmax=np.percentile(display, 99))
    rgba = np.zeros((*display.shape, 4), dtype=float)
    truth = np.rot90(reference[:, :, z])
    if prediction is None:
        rgba[truth] = (0.1, 0.9, 0.2, 0.65)
    else:
        predicted = np.rot90(prediction[:, :, z])
        rgba[truth & predicted] = (0.1, 0.9, 0.2, 0.6)  # Correct lesion voxels.
        rgba[truth & ~predicted] = (1.0, 0.1, 0.1, 0.7)  # Missed voxels.
        rgba[~truth & predicted] = (1.0, 0.7, 0.0, 0.7)  # Extra voxels.
    ax.imshow(rgba)
    ax.axis("off")


def draw_montage(subject: str, target_id: str, baseline_id: str, title: str, output: Path) -> None:
    source = nib.load(RAW / DATASETS["501"] / "imagesTr" / f"{subject}_0000.nii.gz")
    base = np.asarray(source.dataobj, dtype=np.float32)
    reference = np.asarray(
        nib.load(RAW / DATASETS["501"] / "labelsTr" / f"{subject}.nii.gz").dataobj,
        dtype=bool,
    )
    baseline = mask(baseline_id, subject)
    target = mask(target_id, subject)
    if not (base.shape == reference.shape == baseline.shape == target.shape):
        raise ValueError(f"Image/mask shape mismatch: {subject}")
    disagreement = np.logical_xor(baseline, reference) | np.logical_xor(target, reference)
    slice_scores = disagreement.sum(axis=(0, 1))
    if not slice_scores.any():
        slice_scores = reference.sum(axis=(0, 1))
    z = int(np.argmax(slice_scores))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), layout="constrained")
    for ax, pred, label in zip(
        axes,
        (None, None, baseline, target),
        ("DWI", "Reference", f"Baseline {baseline_id}", f"+FLAIR {target_id}"),
    ):
        overlay(ax, base, reference if label != "DWI" else np.zeros_like(reference), pred, z)
        ax.set_title(label)
    fig.suptitle(f"{title} — {subject} — slice {z} | TP green, FN red, FP yellow", fontsize=10)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    input_path = PROJECT / "reports/detailed_evaluation/per_case_private.csv"
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_dataset = {
        dataset_id: {row["subject"]: row for row in rows if row["dataset_id"] == dataset_id}
        for dataset_id in DATASETS
    }
    subjects = set(by_dataset["501"])
    if len(subjects) != 50 or any(set(cases) != subjects for cases in by_dataset.values()):
        raise ValueError("Expected four matching 50-case validation result sets")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    public: dict[str, object] = {"n_patients": len(subjects), "comparisons": {}}
    private_lines = ["# 환자별 오류 사례 (로컬 전용)", "", "그림: 녹색=맞춘 병변, 빨강=놓친 병변, 노랑=과잉 예측.", ""]

    for name, (target_id, baseline_id) in PAIRS.items():
        changes = []
        categories: Counter[str] = Counter()
        for subject in sorted(subjects):
            target, baseline = by_dataset[target_id][subject], by_dataset[baseline_id][subject]
            f1_delta = float(target["lesion_f1"]) - float(baseline["lesion_f1"])
            dice_delta = float(target["dice"]) - float(baseline["dice"])
            small_delta = int(target["detected_small_lesions"]) - int(baseline["detected_small_lesions"])
            fp_delta = int(target["false_positive_lesions"]) - int(baseline["false_positive_lesions"])
            changes.append((f1_delta, subject, dice_delta, small_delta, fp_delta))
            categories["multiple_reference_lesions" if int(target["reference_lesions"]) > 1 else "zero_or_one_reference_lesion"] += 1
            categories["small_lesions_found_more" if small_delta > 0 else "small_lesions_found_fewer" if small_delta < 0 else "small_lesions_found_same"] += 1
            categories["false_positives_more" if fp_delta > 0 else "false_positives_fewer" if fp_delta < 0 else "false_positives_same"] += 1
            categories["f1_higher" if f1_delta > 1e-12 else "f1_lower" if f1_delta < -1e-12 else "f1_same"] += 1
            categories["dice_higher" if dice_delta > 1e-12 else "dice_lower" if dice_delta < -1e-12 else "dice_same"] += 1
        public["comparisons"][name] = dict(categories)

        losers = sorted((case for case in changes if case[0] < -1e-12), key=lambda case: case[0])[:3]
        winners = sorted((case for case in changes if case[0] > 1e-12), key=lambda case: -case[0])[:3]
        private_lines.extend([f"## {name}", ""])
        for direction, selected in (("lower_F1", losers), ("higher_F1", winners)):
            for rank, (f1_delta, subject, dice_delta, small_delta, fp_delta) in enumerate(selected, 1):
                filename = f"{name}_{direction}_{rank}.png"
                draw_montage(subject, target_id, baseline_id, name, PRIVATE / filename)
                private_lines.append(
                    f"- {subject}: F1 {f1_delta:+.3f}, Dice {dice_delta:+.3f}, "
                    f"small detections {small_delta:+d}, FP lesions {fp_delta:+d} — ![]({filename})"
                )
        private_lines.append("")

    (PROJECT / "reports/error_analysis_summary.json").write_text(
        json.dumps(public, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (PRIVATE / "case_report.md").write_text("\n".join(private_lines) + "\n", encoding="utf-8")
    print("Saved aggregate error categories: reports/error_analysis_summary.json")
    print("Saved local-only case report and montages: reports/private_error_analysis/")


if __name__ == "__main__":
    main()
