#!/usr/bin/env python3
"""Rigidly register skull-stripped FLAIR images to the native DWI grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_paths(root: Path, subject: str) -> tuple[Path, Path]:
    prefix = f"{subject}_ses-0001"
    session = root / subject / "ses-0001"
    return (
        session / "dwi" / f"{prefix}_dwi.nii.gz",
        session / "anat" / f"{prefix}_FLAIR.nii.gz",
    )


def foreground_mask(image: sitk.Image) -> sitk.Image:
    return sitk.Cast(sitk.Abs(image) > 1e-6, sitk.sitkUInt8)


def dice(first: sitk.Image, second: sitk.Image) -> float:
    a = sitk.GetArrayViewFromImage(first) > 0
    b = sitk.GetArrayViewFromImage(second) > 0
    denominator = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / denominator) if denominator else 1.0


def register_case(dwi_path: Path, flair_path: Path) -> tuple[sitk.Image, sitk.Transform, dict[str, object]]:
    fixed = sitk.Cast(sitk.ReadImage(str(dwi_path)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(str(flair_path)), sitk.sitkFloat32)
    fixed_mask = foreground_mask(fixed)
    moving_mask = foreground_mask(moving)

    identity_transform = sitk.Transform(3, sitk.sitkIdentity)
    identity_resampled = sitk.Resample(
        moving,
        fixed,
        identity_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    identity_resampled_mask = sitk.Resample(
        moving_mask,
        fixed,
        identity_transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    dice_before = dice(fixed_mask, identity_resampled_mask)

    # Most cases already have sufficiently consistent physical geometry. Avoid
    # optimizing them because multimodal mutual information can find a wrong
    # local optimum even when the header-based alignment is already good.
    if dice_before >= 0.95:
        return identity_resampled, identity_transform, {
            "method": "header_resample",
            "brainmask_dice_before": dice_before,
            "brainmask_dice_after": dice_before,
            "metric_value": "",
            "iterations": 0,
            "stop_condition": "Skipped rigid optimization: initial brain-mask Dice >= 0.95",
        }

    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2, seed=2026)
    registration.SetMetricFixedMask(fixed_mask)
    registration.SetMetricMovingMask(moving_mask)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-3,
        numberOfIterations=250,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-6,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)
    final_transform = registration.Execute(sitk.Normalize(fixed), sitk.Normalize(moving))

    registered = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    registered_mask = sitk.Resample(
        moving_mask,
        fixed,
        final_transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    dice_after = dice(fixed_mask, registered_mask)
    if dice_after + 0.005 < dice_before:
        return identity_resampled, identity_transform, {
            "method": "header_resample_fallback",
            "brainmask_dice_before": dice_before,
            "brainmask_dice_after": dice_before,
            "metric_value": float(registration.GetMetricValue()),
            "iterations": int(registration.GetOptimizerIteration()),
            "stop_condition": "Rigid result rejected because coarse overlap became worse",
        }

    stats = {
        "method": "rigid_mi",
        "brainmask_dice_before": dice_before,
        "brainmask_dice_after": dice_after,
        "metric_value": float(registration.GetMetricValue()),
        "iterations": int(registration.GetOptimizerIteration()),
        "stop_condition": registration.GetOptimizerStopConditionDescription(),
    }
    return registered, final_transform, stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transform_dir = args.output_dir / "transforms"
    transform_dir.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or sorted(
        path.name for path in args.data_root.glob("sub-strokecase*") if path.is_dir()
    )
    rows = []

    for index, subject in enumerate(subjects, start=1):
        output_path = args.output_dir / f"{subject}_flair_to_dwi.nii.gz"
        transform_path = transform_dir / f"{subject}_flair_to_dwi.tfm"
        if output_path.exists() and transform_path.exists() and not args.overwrite:
            print(f"[{index}/{len(subjects)}] skip {subject}", flush=True)
            continue
        dwi_path, flair_path = image_paths(args.data_root, subject)
        try:
            registered, transform, stats = register_case(dwi_path, flair_path)
            sitk.WriteImage(registered, str(output_path), True)
            sitk.WriteTransform(transform, str(transform_path))
            row = {"subject": subject, "status": "ok", **stats}
            print(
                f"[{index}/{len(subjects)}] {subject}: "
                f"Dice {stats['brainmask_dice_before']:.3f} -> {stats['brainmask_dice_after']:.3f}",
                flush=True,
            )
        except Exception as exc:
            row = {"subject": subject, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[{index}/{len(subjects)}] {subject}: ERROR {row['error']}", flush=True)
        rows.append(row)

    report_path = args.output_dir / "registration_report.csv"
    existing_rows = []
    if report_path.exists() and not args.overwrite:
        with report_path.open(newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
    merged = {row["subject"]: row for row in existing_rows}
    merged.update({row["subject"]: row for row in rows})
    fieldnames = sorted({key for row in merged.values() for key in row}, key=lambda key: key != "subject")
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged[subject] for subject in sorted(merged))
    print(report_path)


if __name__ == "__main__":
    main()
