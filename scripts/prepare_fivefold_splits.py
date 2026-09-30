#!/usr/bin/env python3
"""Extend the fixed 200/50 split to shared, patient-disjoint five-fold CV.

Fold 0 is kept byte-for-byte in membership (not necessarily JSON formatting).
The original 200 training patients are deterministically shuffled into four
50-patient validation groups for folds 1-4. No images or labels are modified.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = (
    "Dataset501_ISLES22_DWI",
    "Dataset502_ISLES22_DWI_ADC",
    "Dataset503_ISLES22_DWI_FLAIR",
    "Dataset504_ISLES22_DWI_ADC_FLAIR",
)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def case_ids(raw_dataset: Path) -> set[str]:
    return {path.name.removesuffix(".nii.gz") for path in (raw_dataset / "labelsTr").glob("*.nii.gz")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    project = args.project.resolve()

    original = read_json(project / "splits/splits_master.json")
    if not isinstance(original, list) or len(original) != 1:
        raise ValueError("Expected the existing single fold-0 master split")
    fold0 = original[0]
    train0, val0 = set(fold0["train"]), set(fold0["val"])
    if len(train0) != 200 or len(val0) != 50 or train0 & val0:
        raise ValueError("Expected disjoint 200/50 original split")
    all_patients = train0 | val0

    for dataset in DATASETS:
        ids = case_ids(project / "nnunet_data/raw" / dataset)
        if ids != all_patients:
            raise ValueError(f"Raw label patient IDs differ from the master split: {dataset}")
        existing = read_json(project / "nnunet_data/preprocessed" / dataset / "splits_final.json")
        if not isinstance(existing, list) or not existing:
            raise ValueError(f"Missing existing fold-0 split: {dataset}")
        if set(existing[0]["train"]) != train0 or set(existing[0]["val"]) != val0:
            raise ValueError(f"Existing fold 0 differs; refusing to overwrite: {dataset}")

    shuffled = sorted(train0)
    random.Random(args.seed).shuffle(shuffled)
    folds = [{"train": sorted(train0), "val": sorted(val0)}]
    for index in range(4):
        validation = set(shuffled[index * 50 : (index + 1) * 50])
        folds.append({"train": sorted(all_patients - validation), "val": sorted(validation)})
    if {patient for fold in folds for patient in fold["val"]} != all_patients:
        raise AssertionError("Validation folds do not cover all patients")
    if any(len(fold["train"]) != 200 or len(fold["val"]) != 50 for fold in folds):
        raise AssertionError("An unexpected fold size was generated")
    if any(set(fold["train"]) & set(fold["val"]) for fold in folds):
        raise AssertionError("Training/validation overlap detected")
    if len({patient for fold in folds for patient in fold["val"]}) != sum(len(fold["val"]) for fold in folds):
        raise AssertionError("A patient occurs in more than one validation fold")

    payload = json.dumps(folds, indent=2, ensure_ascii=False) + "\n"
    destination = project / "splits/splits_5fold.json"
    destination.write_text(payload, encoding="utf-8")
    for dataset in DATASETS:
        target = project / "nnunet_data/preprocessed" / dataset / "splits_final.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
    print(f"Prepared 5 shared folds (200 training / 50 validation per fold; seed {args.seed}).")
    print("Fold 0 unchanged; each of 250 patients validates exactly once; four datasets share the same split.")


if __name__ == "__main__":
    main()
