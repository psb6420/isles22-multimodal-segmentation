#!/usr/bin/env python3
"""Prepare nnU-Net v2 raw datasets from read-only ISLES'22 source files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


DATASETS = {
    "dwi": (501, "ISLES22_DWI", ("dwi",)),
    "dwi_adc": (502, "ISLES22_DWI_ADC", ("dwi", "adc")),
    "dwi_flair": (503, "ISLES22_DWI_FLAIR", ("dwi", "flair")),
    "dwi_adc_flair": (504, "ISLES22_DWI_ADC_FLAIR", ("dwi", "adc", "flair")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--configuration", choices=sorted(DATASETS), required=True)
    parser.add_argument("--registered-flair-root", type=Path)
    return parser.parse_args()


def source_paths(root: Path, subject: str) -> dict[str, Path]:
    prefix = f"{subject}_ses-0001"
    session = root / subject / "ses-0001"
    return {
        "dwi": session / "dwi" / f"{prefix}_dwi.nii.gz",
        "adc": session / "dwi" / f"{prefix}_adc.nii.gz",
        "mask": root / "derivatives" / subject / "ses-0001" / f"{prefix}_msk.nii.gz",
    }


def hardlink_once(source: Path, target: Path) -> None:
    if target.exists():
        if os.path.samefile(source, target):
            return
        raise FileExistsError(f"Target already exists but is not the expected hard link: {target}")
    os.link(source, target)


def normalized_label_once(source: Path, target: Path) -> None:
    if target.exists():
        existing = nib.load(target)
        labels = set(np.unique(np.asanyarray(existing.dataobj)).tolist())
        if labels.issubset({0, 1}):
            return
        raise ValueError(f"Existing label is not binary: {target}: {labels}")
    image = nib.load(source)
    data = (image.get_fdata(dtype=np.float32) > 0.5).astype(np.uint8)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1.0, 0.0)
    output = nib.Nifti1Image(data, image.affine, header=header)
    output.set_qform(image.get_qform(), int(image.header["qform_code"]))
    output.set_sform(image.get_sform(), int(image.header["sform_code"]))
    nib.save(output, target)


def main() -> None:
    args = parse_args()
    dataset_id, dataset_name, modalities = DATASETS[args.configuration]
    if "flair" in modalities and args.registered_flair_root is None:
        raise ValueError("FLAIR configurations require --registered-flair-root")

    dataset_dir = args.nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(path.name for path in args.source_root.glob("sub-strokecase*") if path.is_dir())
    for subject in subjects:
        paths = source_paths(args.source_root, subject)
        for channel, modality in enumerate(modalities):
            if modality == "flair":
                source = args.registered_flair_root / f"{subject}_flair_to_dwi.nii.gz"
            else:
                source = paths[modality]
            if not source.is_file():
                raise FileNotFoundError(source)
            hardlink_once(source, images_dir / f"{subject}_{channel:04d}.nii.gz")
        normalized_label_once(paths["mask"], labels_dir / f"{subject}.nii.gz")

    dataset_json = {
        "channel_names": {str(index): modality.upper() for index, modality in enumerate(modalities)},
        "labels": {"background": 0, "stroke": 1},
        "numTraining": len(subjects),
        "file_ending": ".nii.gz",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    print(dataset_dir)
    print(f"subjects={len(subjects)} modalities={modalities}")


if __name__ == "__main__":
    main()
