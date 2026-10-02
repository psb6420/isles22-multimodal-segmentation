#!/usr/bin/env python3
"""Evaluate 250 out-of-fold predictions per model after all five folds finish."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluate_validation import DATASETS, PROJECT, check_grid, evaluate_case, load_binary, summarise


TRAINER = "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres_common"
PAIRS = (("503", "501"), ("504", "502"))


def render_report(summary: dict[str, object]) -> str:
    models = summary["models"]
    folds = summary["folds"]
    comparisons = summary["paired_comparisons"]
    lines = [
        "# ISLES'22 5-fold 교차검증 결과",
        "",
        "동일한 환자 250명을 5개 fold로 나눠 각 환자를 정확히 한 번 검증했다. 각 fold는 200명 학습, 50명 검증이며, 네 모델은 같은 분할을 사용했다.",
        "모든 완료된 fold는 RTX 2080에서 학습했다. GTX 1070은 냉각 문제로 첫 fold 학습 중 중단해 결과에 포함하지 않았다.",
        "기존 fold 0 결과를 본 뒤 실험 방향을 정했으므로, 이 교차검증도 완전히 독립적인 최종 테스트는 아니다.",
        "",
        "## 전체 환자의 out-of-fold 예측",
        "",
        "| 입력 | Dice (빈 마스크=1) ↑ | 병변 F1 ↑ | 작은 병변 recall (<1 mL) ↑ | HD95 mm ↓ | 거짓 양성 병변 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_id in DATASETS:
        model = models[dataset_id]
        lines.append(
            f"| {model['input']} | {model['mean_dice']:.4f} | {model['mean_lesion_f1']:.4f} "
            f"| {model['small_lesion_recall_micro']:.4f} "
            f"({model['small_lesions_found']}/{model['small_lesions_total']}) "
            f"| {model['mean_hd95_mm']:.2f} | {model['false_positive_lesions_total']} |"
        )
    lines.extend(["", "## Fold별 Dice", "", "| Fold | DWI | DWI+ADC | DWI+FLAIR | DWI+ADC+FLAIR |", "|---:|---:|---:|---:|---:|"])
    for fold in range(5):
        values = [f"{folds[str(fold)][dataset_id]['mean_dice']:.4f}" for dataset_id in DATASETS]
        lines.append(f"| {fold} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## FLAIR 추가 효과",
        "",
        "| 비교 | 전체 평균 Dice 차이 | Fold별 Dice 차이 평균 ± 표준편차 | 전체 병변 F1 차이 | 전체 작은 병변 recall 차이 |",
        "|---|---:|---:|---:|---:|",
    ])
    for target, baseline in PAIRS:
        pair = comparisons[f"{target}_minus_{baseline}"]
        per_fold = [
            folds[str(fold)][target]["mean_dice"] - folds[str(fold)][baseline]["mean_dice"]
            for fold in range(5)
        ]
        lines.append(
            f"| {pair['target']} − {pair['baseline']} "
            f"| {pair['mean_dice_difference']:+.4f} "
            f"| {statistics.mean(per_fold):+.4f} ± {statistics.stdev(per_fold):.4f} "
            f"| {pair['mean_lesion_f1_difference']:+.4f} "
            f"| {pair['small_lesion_recall_difference']:+.4f} |"
        )
    lines.extend([
        "",
        "## 평가 정의와 확인",
        "",
        "Dice와 병변 F1은 환자별 값의 평균이다. 양쪽 마스크가 모두 비면 1로 처리한다. nnU-Net은 양쪽이 빈 환자의 Dice를 평균에서 제외하므로 해당 규칙으로도 별도 계산했다.",
        "nnU-Net 규칙의 전체 Dice는 " + ", ".join(
            f"{models[key]['input']} {models[key]['mean_dice_nnunet_empty_excluded']:.4f}"
            for key in DATASETS
        ) + "이다. 20개 fold별 재계산 Dice는 원본 nnU-Net summary와 오차 1e-10 이내로 일치한다.",
        "병변은 26-연결 구성요소이며 한 voxel 이상 겹치는 정답 병변을 검출로 센다. 정답과 전혀 겹치지 않는 예측 구성요소는 거짓 양성이다. 일대일 매칭이나 최소 부피 제거는 적용하지 않았다.",
        "작은 병변 recall은 1 mL 미만 정답 구성요소 전체를 합산한 micro recall이며 본 프로젝트의 추가 지표다. HD95는 양방향 표면 거리의 합친 분포에서 95백분위(mm)다. 양쪽이 비면 0, 한쪽만 비면 미정의로 평균에서 제외한다.",
        "HD95가 정의된 환자 수는 " + ", ".join(
            f"{models[key]['input']} {models[key]['hd95_defined_cases']}/250"
            for key in DATASETS
        ) + "이다. 따라서 평균 HD95만으로 완전 미검출의 영향을 판단할 수 없다.",
        "환자 단위 train/validation 교집합 없음, 네 조건의 동일 분할, 환자별 검증 예측 한 번씩 총 250건을 확인했다. 환자별 자료와 영상·가중치는 로컬에만 보관한다.",
        "",
        "## 해석",
        "",
        "DWI에 ADC를 추가한 평균 Dice 변화는 +0.0066이다. DWI+ADC에 FLAIR를 더하면 평균 Dice와 병변 F1이 소폭 낮아지고 작은 병변 recall은 거의 같았다. 이번 조건에서는 FLAIR의 일관된 추가 이득을 확인하지 못했다. DWI+FLAIR의 평균 Dice가 가장 높지만 유의한 우월성이나 FLAIR의 일반적인 불필요성을 뜻하지 않는다.",
        "",
        "Fold가 5개뿐이므로 fold 간 표준편차는 불확실성의 대략적인 표시이며 유의성 검정이 아니다."
        " 환자 단위 bootstrap도 같은 학습 모델을 공유하는 환자들의 의존성을 반영하지 못할 수 있어 여기서는 유의성 주장에 사용하지 않았다.",
        "공식 비공개 테스트셋이나 외부 기관 데이터 평가가 아니다.",
        "",
        "재현: `python scripts/evaluate_crossval.py`",
        "",
        "공식 병변 평가 규칙: <https://github.com/ezequieldlrosa/isles22/blob/main/utils/eval_utils.py>",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    splits = json.loads((PROJECT / "splits/splits_5fold.json").read_text(encoding="utf-8"))
    if len(splits) != 5:
        raise ValueError("Expected five folds")
    fold_val = [set(fold["val"]) for fold in splits]
    all_patients = set.union(*fold_val)
    if len(all_patients) != 250 or sum(len(group) for group in fold_val) != 250:
        raise ValueError("Validation must contain 250 unique patients across the five folds")
    if any(len(fold["train"]) != 200 or len(fold["val"]) != 50 for fold in splits):
        raise ValueError("Unexpected fold size")

    rows: list[dict[str, object]] = []
    fold_metrics: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
    reference_hashes: dict[str, bytes] = {}
    for fold_index, subjects in enumerate(fold_val):
        for dataset_id, (input_name, dataset_dir) in DATASETS.items():
            folder = PROJECT / "nnunet_data/results" / dataset_dir / TRAINER / f"fold_{fold_index}/validation"
            actual = {path.name.removesuffix(".nii.gz") for path in folder.glob("*.nii.gz")}
            if actual != subjects:
                raise ValueError(f"Missing/out-of-fold predictions in {folder}: {len(actual)} instead of 50")
            subrows = []
            for subject in sorted(subjects):
                reference_image, reference = load_binary(
                    PROJECT / "nnunet_data/raw" / dataset_dir / "labelsTr" / f"{subject}.nii.gz"
                )
                digest = hashlib.blake2b(reference.tobytes(), digest_size=16).digest()
                if subject in reference_hashes and reference_hashes[subject] != digest:
                    raise ValueError(f"Different reference masks across datasets for {subject}")
                reference_hashes[subject] = digest
                prediction_image, prediction = load_binary(folder / f"{subject}.nii.gz")
                check_grid(reference_image, prediction_image)
                metrics = evaluate_case(
                    reference, prediction, nib.affines.voxel_sizes(reference_image.affine), 1.0
                )
                metrics["absolute_lesion_count_difference"] = abs(
                    metrics["predicted_lesions"] - metrics["reference_lesions"]
                )
                row = {"subject": subject, "dataset_id": dataset_id, "input": input_name, "fold": fold_index, **metrics}
                rows.append(row)
                subrows.append(row)
            nnunet_summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            native_dice = float(nnunet_summary["foreground_mean"]["Dice"])
            computed_native = np.mean([
                float(row["dice"]) for row in subrows
                if float(row["reference_volume_ml"]) + float(row["predicted_volume_ml"]) > 0
            ])
            if not np.isclose(native_dice, computed_native, atol=1e-10, rtol=0):
                raise ValueError(f"nnU-Net Dice mismatch for fold {fold_index}, dataset {dataset_id}")
            small_total = sum(int(row["small_reference_lesions"]) for row in subrows)
            fold_metrics[str(fold_index)][dataset_id] = {
                "mean_dice": float(np.mean([float(row["dice"]) for row in subrows])),
                "mean_lesion_f1": float(np.mean([float(row["lesion_f1"]) for row in subrows])),
                "small_lesion_recall_micro": sum(int(row["detected_small_lesions"]) for row in subrows) / small_total if small_total else float("nan"),
                "nnunet_empty_excluded_mean_dice": native_dice,
                "n_cases": len(subrows),
            }
            print(f"Evaluated fold {fold_index}, {input_name}: 50 cases", flush=True)

    private = PROJECT / "reports/detailed_evaluation"
    private.mkdir(exist_ok=True)
    with (private / "crossval_per_case_private.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows, iterations=10000, seed=2026)
    summary["folds"] = fold_metrics
    summary["configuration"] = {
        "n_folds": 5,
        "train_per_fold": 200,
        "validation_per_fold": 50,
        "total_out_of_fold_patients": 250,
        "fold_hardware": {str(fold): "RTX 2080" for fold in range(5)},
        "small_lesion_max_ml": 1.0,
        "lesion_connectivity": 26,
        "seed": 2026,
        "patient_bootstrap_note": "Exploratory only; fold-level training dependency is not fully represented",
    }
    (PROJECT / "reports/crossval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (PROJECT / "reports/crossval_report.md").write_text(render_report(summary), encoding="utf-8")
    print("Saved reports/crossval_summary.json and reports/crossval_report.md")


if __name__ == "__main__":
    main()
