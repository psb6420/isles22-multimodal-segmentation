#!/usr/bin/env python3
"""Create a visual QA montage for native and resampled ISLES'22 modalities."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registered-flair-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("subjects", nargs="+")
    return parser.parse_args()


def paths(root: Path, subject: str) -> tuple[Path, Path, Path]:
    prefix = f"{subject}_ses-0001"
    session = root / subject / "ses-0001"
    return (
        session / "dwi" / f"{prefix}_dwi.nii.gz",
        session / "anat" / f"{prefix}_FLAIR.nii.gz",
        root / "derivatives" / subject / "ses-0001" / f"{prefix}_msk.nii.gz",
    )


def normalize_slice(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    foreground = finite[np.abs(finite) > 1e-6]
    values = foreground if foreground.size else finite
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - low) / (high - low), 0, 1)


def main() -> None:
    args = parse_args()
    figure, axes = plt.subplots(len(args.subjects), 4, figsize=(14, 3.3 * len(args.subjects)), squeeze=False)

    for row, subject in enumerate(args.subjects):
        dwi_path, flair_path, mask_path = paths(args.data_root, subject)
        dwi = nib.load(dwi_path)
        if args.registered_flair_root is not None:
            flair = nib.load(args.registered_flair_root / f"{subject}_flair_to_dwi.nii.gz")
        else:
            flair = nib.load(flair_path)
        mask = nib.load(mask_path)
        dwi_data = dwi.get_fdata(dtype=np.float32)
        flair_on_dwi = resample_from_to(flair, dwi, order=1).get_fdata(dtype=np.float32)
        mask_data = mask.get_fdata(dtype=np.float32) > 0.5

        brain_counts = np.sum(np.abs(dwi_data) > 1e-6, axis=(0, 1))
        nonempty = np.flatnonzero(brain_counts)
        slice_index = int(np.median(nonempty)) if nonempty.size else dwi_data.shape[2] // 2
        dwi_slice = normalize_slice(dwi_data[:, :, slice_index])
        flair_slice = normalize_slice(flair_on_dwi[:, :, slice_index])
        dwi_brain = np.abs(dwi_data[:, :, slice_index]) > 1e-6
        flair_brain = np.abs(flair_on_dwi[:, :, slice_index]) > 1e-6
        lesion_slice = mask_data[:, :, slice_index]

        panels = axes[row]
        panels[0].imshow(dwi_slice.T, cmap="gray", origin="lower")
        panels[0].set_title(f"{subject}\nDWI, z={slice_index}")
        panels[1].imshow(flair_slice.T, cmap="gray", origin="lower")
        panels[1].set_title("FLAIR → DWI grid")
        panels[2].imshow(dwi_slice.T, cmap="gray", origin="lower")
        panels[2].contour(dwi_brain.T, levels=[0.5], colors=["magenta"], linewidths=0.8)
        panels[2].contour(flair_brain.T, levels=[0.5], colors=["cyan"], linewidths=0.8)
        panels[2].set_title("Brain edge: DWI magenta / FLAIR cyan")
        panels[3].imshow(dwi_slice.T, cmap="gray", origin="lower")
        if lesion_slice.any():
            panels[3].contour(lesion_slice.T, levels=[0.5], colors=["lime"], linewidths=1.2)
        panels[3].set_title("Ground truth on DWI")
        for panel in panels:
            panel.axis("off")

    figure.suptitle("ISLES'22 FLAIR-to-DWI header-based resampling QA", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
