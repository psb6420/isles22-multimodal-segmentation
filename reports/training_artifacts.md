# ISLES’22 nnU-Net 학습 산출물 정리

확인일: 2026-09-29 (UTC)

## 공통 실험 설정

- 모델: nnU-Net v2 `PlainConvUNet` 3D full-resolution
- trainer: `nnUNetTrainer_250epochs`
- configuration: `3d_fullres_common`
- fold: 0
- 환자 분할: 학습 200명, 검증 50명
- spacing: 2 × 2 × 2 mm
- patch: 80 × 96 × 80
- batch size: 2
- epochs: 250
- `torch.compile`: 비활성화
- FLAIR: DWI 공간으로 정합한 파생 영상을 사용

## 데이터셋과 입력 채널

| Dataset | 입력 채널 |
|---|---|
| Dataset501_ISLES22_DWI | 0: DWI |
| Dataset502_ISLES22_DWI_ADC | 0: DWI, 1: ADC |
| Dataset503_ISLES22_DWI_FLAIR | 0: DWI, 1: FLAIR |
| Dataset504_ISLES22_DWI_ADC_FLAIR | 0: DWI, 1: ADC, 2: FLAIR |

## 기본 검증 결과

| Dataset | mean voxel Dice | 검증 예측 수 |
|---|---:|---:|
| Dataset501_ISLES22_DWI | 0.7652709441 | 50 |
| Dataset502_ISLES22_DWI_ADC | 0.7570913697 | 50 |
| Dataset503_ISLES22_DWI_FLAIR | 0.7626252284 | 50 |
| Dataset504_ISLES22_DWI_ADC_FLAIR | 0.7665060449 | 50 |

이 수치는 nnU-Net의 `validation/summary.json`에 기록된 voxel-wise Dice이다. lesion-wise F1, small-lesion recall, HD95, 병변 부피 오차 및 통계 검정은 아직 포함하지 않는다.

## 주요 경로

- 원본 데이터: `$HOME/datasets/ISLES-2022/extracted/ISLES-2022`
- 정합된 FLAIR: `$HOME/datasets/ISLES-2022/derived/flair_registered_to_dwi`
- nnU-Net 전처리 데이터: `isles22_project/nnunet_data/preprocessed`
- 공식 학습 결과: `isles22_project/nnunet_data/results`
- 고정 분할: `isles22_project/splits/splits_master.json`
- 학습 실행 스크립트: `isles22_project/scripts/train_one.sh`

각 Dataset 결과 폴더의 `fold_0` 아래에는 다음 파일이 있다.

- `checkpoint_final.pth`: 250 epochs 종료 시점 가중치
- `checkpoint_best.pth`: 학습 중 best EMA pseudo Dice 기준 가중치
- `training_log_*.txt`: epoch별 loss, pseudo Dice, 학습 시간
- `validation/summary.json`: 기본 검증 지표
- `validation/*.nii.gz`: 50명 검증 예측 마스크
- `validation/*.npz`: softmax 확률 저장본

공식 결과의 기준 위치는 RTX 2080 데스크톱이다. GTX 1070 노트북은 환경 시험과 데이터 복사용으로 사용했으며 공식 최종 결과 위치로 사용하지 않는다.
