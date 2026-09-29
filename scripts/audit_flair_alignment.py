#!/usr/bin/env python3
"""Measure coarse FLAIR-to-DWI geometric agreement using nonzero brain masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def world_centroid(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return np.full(3, np.nan)
    voxel_center = indices.mean(axis=0)
    return nib.affines.apply_affine(affine, voxel_center)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    subjects = sorted(path.name for path in args.data_root.glob("sub-strokecase*") if path.is_dir())

    for index, subject in enumerate(subjects, start=1):
        prefix = f"{subject}_ses-0001"
        dwi_path = args.data_root / subject / "ses-0001" / "dwi" / f"{prefix}_dwi.nii.gz"
        flair_path = args.data_root / subject / "ses-0001" / "anat" / f"{prefix}_FLAIR.nii.gz"
        dwi = nib.load(dwi_path)
        flair = nib.load(flair_path)

        dwi_data = np.asanyarray(dwi.dataobj)
        flair_data = np.asanyarray(flair.dataobj)
        dwi_mask = np.isfinite(dwi_data) & (np.abs(dwi_data) > 1e-6)
        flair_mask_image = nib.Nifti1Image(
            (np.isfinite(flair_data) & (np.abs(flair_data) > 1e-6)).astype(np.uint8),
            flair.affine,
        )
        flair_on_dwi = np.asanyarray(resample_from_to(flair_mask_image, dwi, order=0).dataobj) > 0

        intersection = int(np.logical_and(dwi_mask, flair_on_dwi).sum())
        denominator = int(dwi_mask.sum() + flair_on_dwi.sum())
        dice = (2.0 * intersection / denominator) if denominator else 1.0
        dwi_count = int(dwi_mask.sum())
        flair_count = int(flair_on_dwi.sum())
        centroid_distance = float(
            np.linalg.norm(world_centroid(dwi_mask, dwi.affine) - world_centroid(flair_on_dwi, dwi.affine))
        )

        rows.append(
            {
                "subject": subject,
                "brainmask_dice": dice,
                "centroid_distance_mm": centroid_distance,
                "dwi_foreground_voxels": dwi_count,
                "flair_on_dwi_foreground_voxels": flair_count,
                "dwi_covered_by_flair": intersection / dwi_count if dwi_count else 0.0,
                "flair_covered_by_dwi": intersection / flair_count if flair_count else 0.0,
            }
        )
        if index % 25 == 0 or index == len(subjects):
            print(f"Checked {index}/{len(subjects)} subjects", flush=True)

    csv_path = args.output_dir / "flair_alignment_proxy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dice_values = np.asarray([row["brainmask_dice"] for row in rows])
    distance_values = np.asarray([row["centroid_distance_mm"] for row in rows])
    worst_dice = sorted(rows, key=lambda row: row["brainmask_dice"])[:10]
    worst_distance = sorted(rows, key=lambda row: row["centroid_distance_mm"], reverse=True)[:10]
    summary = {
        "note": "Nonzero-mask overlap is a coarse geometric proxy, not proof of anatomical registration.",
        "brainmask_dice": {
            "min": float(dice_values.min()),
            "median": float(np.median(dice_values)),
            "max": float(dice_values.max()),
        },
        "centroid_distance_mm": {
            "min": float(distance_values.min()),
            "median": float(np.median(distance_values)),
            "max": float(distance_values.max()),
        },
        "worst_dice_cases": worst_dice,
        "largest_centroid_distance_cases": worst_distance,
    }
    json_path = args.output_dir / "flair_alignment_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
