#!/usr/bin/env python3
"""Evaluate the four nnU-Net validation masks on the same ISLES'22 patients.

The lesion F1 follows the ISLES'22 evaluation convention: 26-connected
reference components are detected by any voxel overlap, while prediction
components with no reference overlap are false positives. Small-lesion recall
and HD95 are additional, explicitly defined exploratory metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = {
    "501": ("DWI", "Dataset501_ISLES22_DWI"),
    "502": ("DWI+ADC", "Dataset502_ISLES22_DWI_ADC"),
    "503": ("DWI+FLAIR", "Dataset503_ISLES22_DWI_FLAIR"),
    "504": ("DWI+ADC+FLAIR", "Dataset504_ISLES22_DWI_ADC_FLAIR"),
}
PAIRS = [("503", "501"), ("504", "502"), ("504", "501"), ("502", "501")]
STRUCTURE_26 = ndimage.generate_binary_structure(3, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT / "nnunet_data/results")
    parser.add_argument("--raw-root", type=Path, default=PROJECT / "nnunet_data/raw")
    parser.add_argument("--split", type=Path, default=PROJECT / "splits/splits_master.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "reports/detailed_evaluation")
    parser.add_argument("--small-lesion-max-ml", type=float, default=1.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.small_lesion_max_ml <= 0 or args.bootstrap_iterations <= 0:
        parser.error("Threshold and bootstrap iterations must be positive")
    return args


def load_binary(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(path)
    array = np.asanyarray(image.dataobj)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite 3D mask: {path}")
    labels = np.unique(array)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Expected only labels 0 and 1 at {path}: {labels}")
    return image, array.astype(bool)


def check_grid(reference: nib.spatialimages.SpatialImage, prediction: nib.spatialimages.SpatialImage) -> None:
    if reference.shape != prediction.shape or not np.allclose(
        reference.affine, prediction.affine, rtol=0, atol=1e-4
    ):
        raise ValueError("Prediction and reference differ in shape or affine")
    directions = reference.affine[:3, :3]
    lengths = np.linalg.norm(directions, axis=0)
    normalized = directions / lengths
    if not np.allclose(normalized.T @ normalized, np.eye(3), atol=1e-3):
        raise ValueError("HD95 requires an orthogonal voxel grid")


def component_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    voxel_volume_ml: float,
    small_lesion_max_ml: float,
) -> dict[str, int | float]:
    reference_labels, n_reference = ndimage.label(reference, structure=STRUCTURE_26)
    prediction_labels, n_prediction = ndimage.label(prediction, structure=STRUCTURE_26)

    reference_sizes = np.bincount(reference_labels.ravel(), minlength=n_reference + 1)[1:]
    detected_reference = np.zeros(n_reference, dtype=bool)
    overlapping_reference = np.unique(reference_labels[prediction])
    detected_reference[overlapping_reference[overlapping_reference > 0] - 1] = True

    overlapping_prediction = np.unique(prediction_labels[reference])
    n_prediction_overlapping = np.count_nonzero(overlapping_prediction > 0)
    tp = int(detected_reference.sum())
    fn = int(n_reference - tp)
    fp = int(n_prediction - n_prediction_overlapping)
    small = reference_sizes * voxel_volume_ml < small_lesion_max_ml
    n_small = int(small.sum())
    detected_small = int(np.logical_and(small, detected_reference).sum())
    denominator = 2 * tp + fp + fn
    return {
        "reference_lesions": int(n_reference),
        "predicted_lesions": int(n_prediction),
        "detected_reference_lesions": tp,
        "missed_reference_lesions": fn,
        "false_positive_lesions": fp,
        "lesion_f1": float(2 * tp / denominator) if denominator else 1.0,
        "small_reference_lesions": n_small,
        "detected_small_lesions": detected_small,
        "small_lesion_recall": float(detected_small / n_small) if n_small else float("nan"),
    }


def hd95_mm(reference: np.ndarray, prediction: np.ndarray, spacing_mm: np.ndarray) -> float:
    """95th percentile of pooled bidirectional surface distances in millimeters."""
    if not reference.any() and not prediction.any():
        return 0.0
    if not reference.any() or not prediction.any():
        return float("nan")

    union = reference | prediction
    occupied = np.where(union)
    slices = tuple(
        slice(max(0, int(axis.min()) - 1), min(reference.shape[i], int(axis.max()) + 2))
        for i, axis in enumerate(occupied)
    )
    ref = reference[slices]
    pred = prediction[slices]
    ref_surface = ref & ~ndimage.binary_erosion(ref, structure=STRUCTURE_26)
    pred_surface = pred & ~ndimage.binary_erosion(pred, structure=STRUCTURE_26)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing_mm)
    distance_to_ref = ndimage.distance_transform_edt(~ref_surface, sampling=spacing_mm)
    distances = np.concatenate((distance_to_pred[ref_surface], distance_to_ref[pred_surface]))
    return float(np.percentile(distances, 95))


def evaluate_case(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing_mm: np.ndarray,
    small_lesion_max_ml: float,
) -> dict[str, int | float]:
    ref_count = int(reference.sum())
    pred_count = int(prediction.sum())
    overlap = int(np.logical_and(reference, prediction).sum())
    voxel_volume_ml = float(np.prod(spacing_mm) / 1000.0)
    dice = 2 * overlap / (ref_count + pred_count) if ref_count + pred_count else 1.0
    return {
        "dice": float(dice),
        "hd95_mm": hd95_mm(reference, prediction, spacing_mm),
        "reference_volume_ml": ref_count * voxel_volume_ml,
        "predicted_volume_ml": pred_count * voxel_volume_ml,
        "absolute_volume_difference_ml": abs(pred_count - ref_count) * voxel_volume_ml,
        "absolute_lesion_count_difference": 0,  # Set from component counts below.
        **component_metrics(reference, prediction, voxel_volume_ml, small_lesion_max_ml),
    }


def interval(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def macro_bootstrap(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    sampled = values[indices]
    with np.errstate(invalid="ignore"):
        return np.nanmean(sampled, axis=1)


def micro_recall_bootstrap(numerators: np.ndarray, denominators: np.ndarray, indices: np.ndarray) -> np.ndarray:
    numerator = numerators[indices].sum(axis=1)
    denominator = denominators[indices].sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(indices), np.nan), where=denominator > 0)


def summarise(rows: list[dict[str, object]], iterations: int, seed: int) -> dict[str, object]:
    subjects = sorted({str(row["subject"]) for row in rows})
    by_dataset = {
        dataset_id: {str(row["subject"]): row for row in rows if row["dataset_id"] == dataset_id}
        for dataset_id in DATASETS
    }
    if any(set(cases) != set(subjects) for cases in by_dataset.values()):
        raise ValueError("Datasets do not have the same validation cases")

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(subjects), size=(iterations, len(subjects)))
    arrays: dict[str, dict[str, np.ndarray]] = {}
    results: dict[str, object] = {"models": {}, "paired_comparisons": {}}
    numeric_fields = (
        "dice", "lesion_f1", "hd95_mm", "absolute_volume_difference_ml",
        "reference_volume_ml", "predicted_volume_ml",
        "reference_lesions", "predicted_lesions", "detected_reference_lesions",
        "missed_reference_lesions", "false_positive_lesions", "small_reference_lesions",
        "detected_small_lesions", "absolute_lesion_count_difference",
    )

    for dataset_id, (name, _) in DATASETS.items():
        cases = by_dataset[dataset_id]
        arrays[dataset_id] = {
            field: np.asarray([float(cases[subject][field]) for subject in subjects])
            for field in numeric_fields
        }
        data = arrays[dataset_id]
        small_total = int(data["small_reference_lesions"].sum())
        small_found = int(data["detected_small_lesions"].sum())
        both_empty = (data["reference_volume_ml"] == 0) & (data["predicted_volume_ml"] == 0)
        nnunet_style_mean_dice = float(data["dice"][~both_empty].mean())
        model_result: dict[str, object] = {
            "input": name,
            "n_cases": len(subjects),
            "mean_dice": float(data["dice"].mean()),
            "mean_dice_nnunet_empty_excluded": nnunet_style_mean_dice,
            "both_empty_cases": int(both_empty.sum()),
            "mean_dice_ci95": interval(macro_bootstrap(data["dice"], sampled)),
            "mean_lesion_f1": float(data["lesion_f1"].mean()),
            "mean_lesion_f1_ci95": interval(macro_bootstrap(data["lesion_f1"], sampled)),
            "small_lesion_recall_micro": float(small_found / small_total) if small_total else None,
            "small_lesion_recall_micro_ci95": interval(
                micro_recall_bootstrap(data["detected_small_lesions"], data["small_reference_lesions"], sampled)
            ) if small_total else None,
            "small_lesions_found": small_found,
            "small_lesions_total": small_total,
            "mean_hd95_mm": float(np.nanmean(data["hd95_mm"])),
            "median_hd95_mm": float(np.nanmedian(data["hd95_mm"])),
            "hd95_defined_cases": int(np.isfinite(data["hd95_mm"]).sum()),
            "mean_absolute_volume_difference_ml": float(data["absolute_volume_difference_ml"].mean()),
            "mean_absolute_lesion_count_difference": float(data["absolute_lesion_count_difference"].mean()),
            "reference_lesions_total": int(data["reference_lesions"].sum()),
            "predicted_lesions_total": int(data["predicted_lesions"].sum()),
            "detected_reference_lesions_total": int(data["detected_reference_lesions"].sum()),
            "false_positive_lesions_total": int(data["false_positive_lesions"].sum()),
        }
        results["models"][dataset_id] = model_result

    for target, baseline in PAIRS:
        target_data = arrays[target]
        baseline_data = arrays[baseline]
        pair_result: dict[str, object] = {
            "target": DATASETS[target][0],
            "baseline": DATASETS[baseline][0],
        }
        for field in ("dice", "lesion_f1"):
            differences = target_data[field] - baseline_data[field]
            bootstrapped = macro_bootstrap(differences, sampled)
            pair_result[f"mean_{field}_difference"] = float(differences.mean())
            pair_result[f"mean_{field}_difference_ci95"] = interval(bootstrapped)
            pair_result[f"{field}_better_cases"] = int((differences > 1e-12).sum())
            pair_result[f"{field}_worse_cases"] = int((differences < -1e-12).sum())
            pair_result[f"{field}_tied_cases"] = int((np.abs(differences) <= 1e-12).sum())

        hd95_common = np.isfinite(target_data["hd95_mm"]) & np.isfinite(baseline_data["hd95_mm"])
        if hd95_common.any():
            hd95_differences = target_data["hd95_mm"][hd95_common] - baseline_data["hd95_mm"][hd95_common]
            hd95_sampled = rng.integers(0, len(hd95_differences), size=(iterations, len(hd95_differences)))
            pair_result["mean_hd95_difference_mm"] = float(hd95_differences.mean())
            pair_result["mean_hd95_difference_mm_ci95"] = interval(hd95_differences[hd95_sampled].mean(axis=1))
            pair_result["hd95_paired_cases"] = int(hd95_common.sum())

        volume_error_differences = (
            target_data["absolute_volume_difference_ml"] - baseline_data["absolute_volume_difference_ml"]
        )
        pair_result["mean_absolute_volume_error_difference_ml"] = float(volume_error_differences.mean())
        pair_result["mean_absolute_volume_error_difference_ml_ci95"] = interval(
            volume_error_differences[sampled].mean(axis=1)
        )

        target_recall = micro_recall_bootstrap(
            target_data["detected_small_lesions"], target_data["small_reference_lesions"], sampled
        )
        baseline_recall = micro_recall_bootstrap(
            baseline_data["detected_small_lesions"], baseline_data["small_reference_lesions"], sampled
        )
        target_total = target_data["small_reference_lesions"].sum()
        baseline_total = baseline_data["small_reference_lesions"].sum()
        if target_total and baseline_total:
            pair_result["small_lesion_recall_difference"] = float(
                target_data["detected_small_lesions"].sum() / target_total
                - baseline_data["detected_small_lesions"].sum() / baseline_total
            )
            pair_result["small_lesion_recall_difference_ci95"] = interval(target_recall - baseline_recall)
        results["paired_comparisons"][f"{target}_minus_{baseline}"] = pair_result

    return results


def format_interval(values: list[float] | None) -> str:
    if values is None:
        return "N/A"
    return f"[{values[0]:+.4f}, {values[1]:+.4f}]"


def render_report(summary: dict[str, object]) -> str:
    config = summary["configuration"]
    models = summary["models"]
    comparisons = summary["paired_comparisons"]
    lines = [
        "# ISLES'22 검증 50건의 병변 단위 평가",
        "",
        "## 평가 설정",
        "",
        "- 동일한 환자 50명에 대한 네 모델의 fold 0 검증 예측을 비교했다.",
        "- 26-연결 병변이 예측과 한 voxel이라도 겹치면 발견으로 계산했다. 이 병변 F1 규칙은 ISLES'22 공식 평가 코드와 일치한다.",
        "- 본 평가의 Dice는 정답과 예측이 모두 비면 1로 계산한다(ISLES'22 공식 코드 기본값). 기존 nnU-Net 요약은 그 사례를 평균에서 제외한다.",
        f"- 작은 병변: 정답 병변 구성요소 부피 < {config['small_lesion_max_ml']:g} mL. 이 기준과 recall은 본 프로젝트의 추가 탐색 지표다.",
        "- HD95: 양방향 표면 거리(mm)를 합친 뒤 95번째 백분위수. 한쪽 마스크만 비어 있으면 정의되지 않아 평균에서 제외한다.",
        f"- 95% 구간은 환자 단위 paired bootstrap {config['bootstrap_iterations']:,}회, seed {config['seed']}으로 계산했다.",
        "- 값은 내부 검증 결과이며 ISLES'22 비공개 공식 테스트 점수가 아니다.",
        "",
        "## 네 모델의 결과",
        "",
        "| 입력 | Dice, 빈 마스크=1 ↑ | 기존 nnU-Net Dice ↑ | 병변 F1 ↑ | 작은 병변 recall ↑ | HD95 mm ↓ | 부피 오차 mL ↓ | 발견 병변 / 505 | 거짓 양성 병변 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_id in DATASETS:
        model = models[dataset_id]
        recall = model["small_lesion_recall_micro"]
        recall_text = f"{recall:.4f} ({model['small_lesions_found']}/{model['small_lesions_total']})" if recall is not None else "N/A"
        lines.append(
            f"| {model['input']} | {model['mean_dice']:.4f} "
            f"| {model['mean_dice_nnunet_empty_excluded']:.4f} | {model['mean_lesion_f1']:.4f} "
            f"| {recall_text} | {model['mean_hd95_mm']:.2f} | {model['mean_absolute_volume_difference_ml']:.2f} "
            f"| {model['detected_reference_lesions_total']} | {model['false_positive_lesions_total']} |"
        )
    lines.extend([
        "",
        "## FLAIR 추가의 같은 환자 비교",
        "",
        "| 비교 | 평균 Dice 차이 (95% 구간) | 평균 병변 F1 차이 (95% 구간) | 작은 병변 recall 차이 (95% 구간) |",
        "|---|---:|---:|---:|",
    ])
    for key in ("503_minus_501", "504_minus_502", "504_minus_501"):
        pair = comparisons[key]
        lines.append(
            f"| {pair['target']} − {pair['baseline']} "
            f"| {pair['mean_dice_difference']:+.4f} {format_interval(pair['mean_dice_difference_ci95'])} "
            f"| {pair['mean_lesion_f1_difference']:+.4f} {format_interval(pair['mean_lesion_f1_difference_ci95'])} "
            f"| {pair.get('small_lesion_recall_difference', float('nan')):+.4f} "
            f"{format_interval(pair.get('small_lesion_recall_difference_ci95'))} |"
        )
    lines.extend([
        "",
        "![FLAIR effect with paired bootstrap intervals](flair_effect_ci.png)",
        "",
        "| 비교 | HD95 평균 차이 mm (95% 구간) | 부피 오차 평균 차이 mL (95% 구간) |",
        "|---|---:|---:|",
    ])
    for key in ("503_minus_501", "504_minus_502"):
        pair = comparisons[key]
        lines.append(
            f"| {pair['target']} − {pair['baseline']} "
            f"| {pair['mean_hd95_difference_mm']:+.2f} "
            f"{format_interval(pair['mean_hd95_difference_mm_ci95'])} (공통 정의 {pair['hd95_paired_cases']}명) "
            f"| {pair['mean_absolute_volume_error_difference_ml']:+.2f} "
            f"{format_interval(pair['mean_absolute_volume_error_difference_ml_ci95'])} |"
        )
    lines.extend([
        "",
        "## 이번 자료가 말하는 것",
        "",
        "DWI에 FLAIR를 추가한 비교와 DWI+ADC에 FLAIR를 추가한 비교 모두에서 "
        "Dice·병변 F1·작은 병변 recall 차이의 환자 단위 bootstrap 95% 구간이 0을 포함한다. "
        "따라서 이번 한 번의 검증에서는 FLAIR가 일관되게 이롭거나 해롭다는 근거가 충분하지 않다.",
        "",
        "DWI+ADC+FLAIR와 DWI의 작은 병변 recall 차이는 양수였지만, 이 비교는 ADC와 FLAIR가 동시에 달라진다. "
        "그 차이를 FLAIR 단독 효과로 해석할 수 없다. 또한 한 voxel만 겹쳐도 병변 검출로 계산하는 "
        "공식 F1 규칙은 경계 품질까지 보장하지 않는다.",
        "",
        "DWI+ADC에 FLAIR를 추가한 모델은 정답 병변을 더 많이 발견했지만 거짓 양성 병변도 증가했다. "
        "따라서 발견률만 보고 모델을 선택하면 과잉 예측을 놓칠 수 있다.",
        "",
        "DWI+FLAIR는 DWI보다 평균 절대 병변 부피 오차가 1.56 mL 컸다. "
        "환자 단위 bootstrap 95% 구간은 0보다 컸지만, 여러 지표를 탐색한 한 번의 내부 검증 결과로 해석해야 한다.",
        "",
        "95% 구간이 0을 포함하면 이번 검증 자료만으로는 변화 방향이 안정적이라고 보기 어렵다. "
        "구간이 0을 벗어나더라도 한 번의 200/50 분할에서 얻은 탐색 결과이며, 외부 데이터 일반화의 증거는 아니다.",
        "",
        "## 재현",
        "",
        "```bash",
        "python scripts/evaluate_validation.py",
        "python scripts/plot_flair_ablation.py",
        "```",
        "",
        "환자별 결과는 공개 저장소에 올리지 않는 `reports/detailed_evaluation/per_case_private.csv`에 저장된다.",
        "",
        "공식 평가 코드: <https://github.com/ezequieldlrosa/isles22/blob/main/utils/eval_utils.py>",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if len(split) != 1:
        raise ValueError("Expected exactly one fixed train/validation split")
    train = set(split[0]["train"])
    validation = sorted(set(split[0]["val"]))
    if train & set(validation) or len(validation) != 50:
        raise ValueError("Unexpected or overlapping patient split")

    rows: list[dict[str, object]] = []
    reference_grid: dict[str, tuple[tuple[int, ...], np.ndarray, bytes]] = {}
    for dataset_id, (name, dataset_dir) in DATASETS.items():
        matches = list((args.results_root / dataset_dir).glob("*/fold_0/validation"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one validation folder for {dataset_dir}: {matches}")
        folder = matches[0]
        actual = {path.name[:-7] for path in folder.glob("*.nii.gz")}
        if actual != set(validation):
            raise ValueError(f"Validation subjects mismatch for {dataset_dir}")

        for index, subject in enumerate(validation, 1):
            reference_path = args.raw_root / dataset_dir / "labelsTr" / f"{subject}.nii.gz"
            prediction_path = folder / f"{subject}.nii.gz"
            reference_image, reference = load_binary(reference_path)
            prediction_image, prediction = load_binary(prediction_path)
            check_grid(reference_image, prediction_image)
            reference_digest = hashlib.blake2b(reference.tobytes(), digest_size=16).digest()
            if subject in reference_grid:
                shape, affine, digest = reference_grid[subject]
                if (
                    reference.shape != shape
                    or not np.allclose(reference_image.affine, affine, rtol=0, atol=1e-4)
                    or reference_digest != digest
                ):
                    raise ValueError(f"Reference differs across datasets for {subject}")
            else:
                reference_grid[subject] = (reference.shape, reference_image.affine, reference_digest)

            spacing = nib.affines.voxel_sizes(reference_image.affine)
            metrics = evaluate_case(reference, prediction, spacing, args.small_lesion_max_ml)
            metrics["absolute_lesion_count_difference"] = abs(
                metrics["predicted_lesions"] - metrics["reference_lesions"]
            )
            rows.append({"subject": subject, "dataset_id": dataset_id, "input": name, **metrics})
            if index % 10 == 0:
                print(f"{name}: {index}/{len(validation)} cases", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_path = args.output_dir / "per_case_private.csv"
    with private_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarise(rows, args.bootstrap_iterations, args.seed)
    for dataset_id, (_, dataset_dir) in DATASETS.items():
        native_summary = json.loads(
            next((args.results_root / dataset_dir).glob("*/fold_0/validation/summary.json")).read_text(
                encoding="utf-8"
            )
        )
        native_dice = float(native_summary["foreground_mean"]["Dice"])
        computed = summary["models"][dataset_id]["mean_dice_nnunet_empty_excluded"]
        if not np.isclose(native_dice, computed, rtol=0, atol=1e-10):
            raise ValueError(f"Dice mismatch for Dataset{dataset_id}: nnU-Net={native_dice}, computed={computed}")
    summary["configuration"] = {
        "small_lesion_max_ml": args.small_lesion_max_ml,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "lesion_connectivity": 26,
        "lesion_match": "any_voxel_overlap_as_in_official_isles22_eval",
        "hd95": "pooled_bidirectional_surface_distance_percentile_95_mm",
        "hd95_empty_policy": "both_empty_zero_one_empty_undefined",
    }
    summary_path = PROJECT / "reports/evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    report_path = PROJECT / "reports/flair_ablation_report.md"
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(f"Saved private patient-level results: {private_path}")
    print(f"Saved aggregate results: {summary_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
