#!/usr/bin/env python3
"""Create one deterministic center- and lesion-volume-stratified split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.audit_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    positive_volumes = np.asarray(
        [float(row["mask_lesion_volume_ml"]) for row in rows if float(row["mask_lesion_volume_ml"]) > 0]
    )
    quartiles = np.quantile(positive_volumes, [0.25, 0.5, 0.75])
    for row in rows:
        volume = float(row["mask_lesion_volume_ml"])
        volume_bin = 0 if volume == 0 else int(np.digitize(volume, quartiles, right=True)) + 1
        row["volume_bin"] = volume_bin
        row["stratum"] = f"center{row['center']}_volume{volume_bin}"

    subjects = [row["subject"] for row in rows]
    strata = [row["stratum"] for row in rows]
    train, val = train_test_split(
        subjects,
        test_size=0.2,
        random_state=args.seed,
        shuffle=True,
        stratify=strata,
    )
    train_set = set(train)
    val_set = set(val)
    split = {"train": sorted(train_set), "val": sorted(val_set)}

    split_path = args.output_dir / "splits_master.json"
    split_path.write_text(json.dumps([split], indent=2), encoding="utf-8")

    manifest_path = args.output_dir / "split_manifest.csv"
    fieldnames = ["subject", "split", "center", "mask_lesion_volume_ml", "volume_bin", "stratum"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["subject"]):
            writer.writerow(
                {
                    "subject": row["subject"],
                    "split": "train" if row["subject"] in train_set else "val",
                    "center": row["center"],
                    "mask_lesion_volume_ml": row["mask_lesion_volume_ml"],
                    "volume_bin": row["volume_bin"],
                    "stratum": row["stratum"],
                }
            )

    print(f"train={len(train_set)} val={len(val_set)} seed={args.seed}")
    for subset_name, subset in (("train", train_set), ("val", val_set)):
        selected = [row for row in rows if row["subject"] in subset]
        centers = {center: sum(row["center"] == center for row in selected) for center in ("1", "2")}
        empty = sum(float(row["mask_lesion_volume_ml"]) == 0 for row in selected)
        print(f"{subset_name}: centers={centers}, empty_masks={empty}")
    print(split_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
