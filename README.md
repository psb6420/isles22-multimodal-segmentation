# ISLES'22 멀티모달 뇌경색 병변 분할

ISLES'22의 DWI, ADC, FLAIR MRI를 이용해 급성 허혈성 뇌졸중 병변을 3D로 분할하고, **입력 모달리티 조합의 효과를 동일한 nnU-Net 조건에서 비교**하는 학부 연구 프로젝트다.

> 이 저장소에는 원본 MRI, 정답 마스크, 환자별 식별자, 전처리 데이터, 예측 영상 및 모델 가중치가 포함되지 않는다. ISLES'22 데이터는 공식 이용 조건에 따라 별도로 받아야 한다.

이 저장소의 자체 작성 코드와 문서는 [Apache License 2.0](LICENSE)으로 배포한다. ISLES'22 데이터 및 외부 소프트웨어에는 이 라이선스가 적용되지 않는다. 데이터 재배포 제한과 외부 라이선스는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 참고한다.

## 연구 질문

동일한 3D 분할 모델에서 입력 채널만 바꾸었을 때 DWI 단독 대비 ADC와 FLAIR의 추가가 성능에 어떤 영향을 주는가?

비교 조건은 다음 네 가지 Early Fusion 설정이다.

1. DWI
2. DWI + ADC
3. DWI + FLAIR
4. DWI + ADC + FLAIR

핵심 가설은 FLAIR 추가가 전체 voxel Dice를 크게 높이지 않더라도 병변 경계 또는 작은 병변 검출에 도움을 줄 수 있다는 것이다. 반대 결과도 모달리티의 시간 의존성, 정합 오차, 입력 정규화, 작은 표본에서의 변동성 관점에서 분석한다.

## 현재까지 완료된 작업

- ISLES'22 공개 학습 데이터 250건의 DWI, ADC, FLAIR, 마스크 구조 확인
- DWI–ADC 및 DWI–마스크는 250건 모두 같은 voxel grid임을 확인
- 원본 FLAIR는 DWI와 같은 grid가 아니므로 DWI 공간으로 정합한 파생 영상 생성
- 중심(center)과 병변 부피를 고려한 고정 환자 단위 분할: 학습 200건 / 검증 50건
- nnU-Net v2 3D full-resolution 모델 네 조건을 각각 250 epoch 학습
- 각 조건의 50건 검증 예측 및 softmax 확률 저장
- 50건의 병변 단위 평가, 작은 병변 분석, 오류 사례 시각화와 추론 비용 측정
- 기존 fold 0을 유지한 5-fold 학습 20개 완료; 250명 전체의 out-of-fold 평가와 분할·Dice 검증 완료

## 사용한 모델과 공통 설정

- 모델: nnU-Net v2 2.8.1 `PlainConvUNet`, 3D full-resolution
- trainer: `nnUNetTrainer_250epochs`
- configuration: `3d_fullres_common`
- 기존 결과: fold 0. 5-fold 실험에서는 fold 0을 유지하고 1~4를 추가 학습 완료
- target spacing: 2 × 2 × 2 mm
- patch: 80 × 96 × 80
- batch size: 2
- epoch: 250
- `torch.compile`: 비활성화
- 데이터 분할, augmentation, loss, seed 및 학습 예산은 모달리티 조건 간 동일

이 구현은 SEALS 코드를 그대로 실행한 것이 아니라, **SEALS가 채택한 강력한 nnU-Net 계열 접근에서 연구 질문에 필요한 통제된 모달리티 ablation을 분리해 구현한 것**이다.

## 5-fold 교차검증 결과

250명 각각을 한 번씩 검증한 out-of-fold 결과다. 네 조건의 20개 학습·검증을 모두 RTX 2080에서 완료했다.

| 입력 | 평균 Dice | 병변 F1 | 작은 병변 recall (<1 mL) |
|---|---:|---:|---:|
| DWI | 0.7755 | 0.7357 | 0.5326 |
| DWI + ADC | 0.7822 | 0.7437 | 0.5375 |
| DWI + FLAIR | 0.7834 | 0.7512 | 0.5394 |
| DWI + ADC + FLAIR | 0.7800 | 0.7336 | 0.5370 |

표의 Dice는 양쪽 마스크가 빈 경우 1로 처리한다. nnU-Net의 빈 마스크 제외 규칙으로 재계산한 fold별 Dice는 원본 summary와 일치했다. 같은 fold에서 학습·검증 환자의 중복이 없고 네 조건이 동일한 분할을 사용함을 확인했다.

DWI에 ADC를 추가하면 Dice가 +0.0066, DWI+ADC에 FLAIR를 추가하면 -0.0022였다. 작은 차이이며 통계적 우월성을 주장하지 않는다. 이번 설정에서 FLAIR의 일관된 추가 이득은 확인되지 않았다. 기존 fold 0 결과를 본 뒤 연구 방향을 정했으므로 독립적인 최종 테스트로 해석하지 않는다. [전체 결과와 평가 정의](reports/crossval_report.md), [집계 JSON](reports/crossval_summary.json)을 참고한다.

## 초기 단일 fold 검증 결과

| 입력 | Dataset ID | mean voxel Dice |
|---|---:|---:|
| DWI | 501 | 0.7653 |
| DWI + ADC | 502 | 0.7571 |
| DWI + FLAIR | 503 | 0.7626 |
| DWI + ADC + FLAIR | 504 | **0.7665** |

초기 단일 분할의 최고값은 세 모달리티 모델이었지만 DWI 단독과의 차이는 약 `+0.0012`로 매우 작았다. DWI+ADC는 이 단일 분할에서 DWI 단독보다 낮았으나, 위의 5-fold 평균에서는 높았다. 이는 특정 분할만으로 결론을 내리기 어려움을 보여준다.

위 수치는 초기 fold 0의 nnU-Net `validation/summary.json` voxel-wise Dice다. 해당 50명의 병변 F1, 작은 병변 recall, HD95, 부피 오차와 환자 단위 paired bootstrap 분석은 [초기 상세 평가 보고서](reports/flair_ablation_report.md)에 정리했다. 해당 두 FLAIR 비교의 Dice·병변 F1·작은 병변 recall의 95% 구간은 모두 0을 포함했다. 외부 검증은 수행하지 않았다.

검증 50명의 [오류 사례 분석](reports/error_case_analysis.md)도 생성했다. DWI+ADC에 FLAIR를 추가할 때 병변 F1은 14명에서 상승, 14명에서 하락, 22명에서 같았고, 거짓 양성 병변은 19명에서 증가했다. 환자 MRI가 포함된 상세 그림은 로컬에만 보관한다.

RTX 2080에서 동일한 10명을 사용한 [추론 비용 측정](reports/inference_benchmark_summary.json)은 네 모델 모두 약 1.2초/명(모델 로딩·전처리·후처리 포함)이었다. PyTorch의 최대 예약 GPU 메모리는 약 1.0 GiB였다. 한 번의 소규모 측정이므로 정밀한 속도 우열 근거는 아니다.

## 검토한 다른 모델과 역할

| 후보 | 성격 | 프로젝트에서의 적합성 및 역할 |
|---|---|---|
| [SEALS](https://github.com/Tabrisrei/ISLES22_SEALS) | ISLES'22 상위권 nnU-Net 기반 방법, 공개 학습·추론 코드와 가중치 | 가장 직접적인 선행 기준. 공식 추론은 DWI+ADC를 사용한다. 이 프로젝트는 동일 backbone의 입력 ablation에 초점을 둔다. |
| [DeepISLES](https://github.com/ezequieldlrosa/DeepIsles) | SEALS, NVAUTO, Factorizer 계열을 결합한 상위 모델 앙상블 | 고정된 외부 benchmark로 적합하지만, 모델 여러 개의 효과가 섞여 모달리티 자체의 기여를 분리하기 어렵다. |
| [Factorizer](https://github.com/pashtari/factorizer-isles22) | Swin Factorizer와 Res-U-Net 기반 ISLES'22 방법 | Transformer 계열 비교 후보. 전체 앙상블보다 단일 모델 비교가 계산량과 해석 면에서 적합하다. |
| [MoME / MoME+](https://github.com/ZhangxinruBIT/MoME) | 여러 뇌 병변과 MRI 모달리티를 다루는 Mixture of Modality Experts 기반 3D foundation model | 뇌 병변·DWI와의 도메인 적합성이 좋다. 다만 ISLES'22가 사전학습 범위에 포함돼 평가 누수 가능성을 따져야 하고, 원 논문 설정은 계산량이 크다. 멀티모달 결합은 MoME+ 구현을 확인해야 한다. |
| [BrainSegFounder](https://github.com/lab-smile/BrainSegFounder) | UK Biobank 자기지도 사전학습 후 BraTS·ATLAS 등에 미세조정한 3D 뇌 MRI foundation model | 뇌 특화 사전학습의 장점은 있지만 T1/T2 계열 중심이라 DWI·ADC와 도메인 차이가 크다. 후순위 전이학습 후보이다. |
| [MedSAM3](https://github.com/Joey-S-Liu/MedSAM3) | 텍스트/프롬프트 기반 범용 의료 분할 후보, LoRA 가중치 공개 | 자동 3D DWI·ADC 분할과 프롬프트 생성 절차를 별도로 설계해야 한다. 2025년 arXiv 단계이므로 현재는 탐색 후보로만 둔다. 이름이 비슷한 다른 저장소와 구분해야 한다. |
| [정현수 교수님 제1저자 논문(Jeong et al., 2024)의 nnU-Net ensemble](https://pmc.ncbi.nlm.nih.gov/articles/PMC11522214/) | DWI·ADC joint training과 channel-wise training, 뇌종양 표현 전이를 결합 | 본 연구와 가장 직접적으로 연결되는 확장 방향이다. 단순 채널 결합 baseline을 확보한 뒤 joint/channel-wise 학습과 전이학습의 기여를 분리해 비교할 수 있다. |

## nnU-Net을 첫 모델로 선택한 이유

1. **연구 질문을 공정하게 검증할 수 있다.** 구조와 학습 조건을 고정하고 입력 채널만 바꾸므로 ADC와 FLAIR 추가 효과를 해석하기 쉽다.
2. **현재 하드웨어에서 실제 학습할 수 있다.** 8 GB급 RTX 2080에서 네 조건을 모두 완료했다. 대형 foundation model 전체 미세조정보다 실패 위험과 일정 부담이 작다.
3. **의료영상 분할의 강력한 기준선이다.** 자동 계획과 표준화된 전처리 덕분에 입문자도 재현 가능한 기준 성능을 만들기 좋다.
4. **앙상블의 혼입 변수를 피한다.** DeepISLES나 SEALS 전체 제출 모델부터 사용하면 backbone, 사전학습, 후처리, 앙상블 효과가 함께 바뀌어 모달리티 효과를 분리하기 어렵다.
5. **foundation model/LoRA 실험의 기준점이 된다.** 이후 MoME·BrainSegFounder·MedSAM 계열을 전체 미세조정 또는 LoRA로 비교할 때 성능뿐 아니라 학습 파라미터, 메모리, 시간 대비 이득을 판단할 기준이 생긴다.

## 전체 작업 흐름

```text
ISLES'22 원본 확인
  └─ DWI/ADC/FLAIR/마스크 형상·affine·spacing 감사
       └─ FLAIR → DWI 공간 정합 및 정합 품질 점검
            └─ 환자 단위 고정 분할(200/50)
                 └─ nnU-Net 형식으로 4개 입력 조합 생성
                      └─ 같은 설정으로 각각 학습·검증
                           └─ voxel/lesion/small-lesion/경계 지표 비교
                                └─ 실패 사례와 임상적 의미 분석
```

## 주요 파일

- `scripts/audit_dataset.py`: modality, 마스크, 형상, spacing, affine, 병변 통계 점검
- `scripts/audit_flair_alignment.py`: FLAIR–DWI의 거친 기하학적 정합 지표 계산
- `scripts/register_flair_to_dwi.py`: FLAIR를 DWI 공간으로 rigid registration
- `scripts/create_split.py`: center와 병변 부피를 고려한 환자 단위 분할 생성
- `scripts/prepare_nnunet_dataset.py`: 네 모달리티 조합을 nnU-Net v2 형식으로 구성
- `scripts/add_common_nnunet_config.py`: 모든 조건에 동일한 batch size 설정 추가
- `scripts/train_one.sh`: 한 조건 학습
- `scripts/run_queue.sh`: 여러 조건 순차 학습
- `reports/training_artifacts.md`: 로컬 학습 산출물 위치와 결과 요약
- `reports/model_comparison.csv`: 기본 voxel Dice 비교표
- `scripts/evaluate_validation.py`: 검증 50건의 병변·경계 지표와 paired bootstrap 계산
- `reports/flair_ablation_report.md`: FLAIR 추가 효과의 상세 평가 및 해석
- `reports/evaluation_summary.json`: 환자 식별자 없는 집계 결과
- `scripts/plot_flair_ablation.py`와 `reports/flair_effect_ci.png`: FLAIR 효과의 환자 단위 신뢰구간 그림
- `scripts/analyze_error_cases.py`와 `reports/error_analysis_summary.json`: 오류 사례의 로컬 시각화와 공개 가능한 집계
- `reports/error_case_analysis.md`: FLAIR 추가 전후의 오류 유형과 사례 그림의 해석 범위
- `scripts/benchmark_inference.py`와 `reports/inference_benchmark_summary.json`: 같은 환자 10명의 추론 시간·GPU 메모리 측정
- `scripts/prepare_fivefold_splits.py`, `scripts/run_fivefold_queue.py`, `scripts/evaluate_crossval.py`: 5-fold 준비·학습·평가
- `reports/crossval_protocol.md`: 분할과 평가 프로토콜
- `reports/crossval_report.md`, `reports/crossval_summary.json`: 완료된 5-fold 결과와 환자 식별자 없는 집계
- `docs/RESEARCH_DIRECTION.md`: 신규 팀원을 위한 연구 배경, 모델 선택과 향후 실험 방향

## 재현 환경

Python 가상환경을 만든 뒤 GPU와 호환되는 PyTorch를 먼저 설치하고 다음을 설치한다.

```bash
pip install nnunetv2==2.8.1 nibabel scipy scikit-learn openpyxl SimpleITK
```

데이터 경로는 환경변수로 지정한다.

```bash
export ISLES22_DATA=/path/to/ISLES-2022
source config/env.sh
```

예시는 DWI baseline 학습이다.

```bash
python scripts/prepare_nnunet_dataset.py \
  --source-root "$ISLES22_DATA" \
  --nnunet-raw "$nnUNet_raw" \
  --configuration dwi

# nnU-Net plan/preprocess와 공통 configuration 생성 후
bash scripts/train_one.sh 501
```

FLAIR 조건은 먼저 정합 영상을 생성하고 `--registered-flair-root`를 전달해야 한다. 원본은 수정하지 않고 파생 결과를 별도 디렉터리에 저장한다.

## 남은 평가 단계

- 동일 분할에서 사전학습 모델과 미세조정 방법 비교
- 가능하면 외부 데이터 검증과 반복 seed 실험

## 핵심 참고자료

- ISLES'22 공식 저장소: <https://github.com/ezequieldlrosa/isles22>
- nnU-Net: <https://github.com/MIC-DKFZ/nnUNet>
- SEALS: <https://github.com/Tabrisrei/ISLES22_SEALS>
- DeepISLES 논문: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12335569/>
- MoME (MICCAI 2024): <https://papers.miccai.org/miccai-2024/015-Paper0143.html>
- 정현수 교수님 제1저자 논문, Jeong et al. (Journal of Imaging Informatics in Medicine, 2024): <https://doi.org/10.1007/s10278-024-01099-6>

## 라이선스와 의료적 이용 제한

- 저장소 자체 코드·문서: Apache-2.0
- ISLES'22 데이터: 저장소에 포함하지 않으며 공식 Data Usage Policy와 CC BY 4.0을 별도로 준수
- 외부 코드·사전학습 가중치: 포함하지 않으며 각 원 저작자의 라이선스를 따름
- 본 코드는 연구용이며 의료기기 또는 임상 진단용으로 검증되지 않음
