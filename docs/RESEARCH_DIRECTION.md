# ISLES'22 멀티모달 뇌경색 분할 프로젝트 연구 방향

이 문서는 프로젝트를 처음 접하는 팀원이 현재 상황과 앞으로의 실험 방향을 빠르게 이해할 수 있도록 작성한 안내서다.

## 1. 한 문장 요약

ISLES'22의 DWI, ADC, FLAIR MRI를 이용해 급성 허혈성 뇌졸중 병변을 자동 분할하고, **모달리티 추가 효과와 뇌 MRI Foundation Model의 미세조정 방법을 분리해 검증**하는 프로젝트다.

## 2. 해결하려는 문제

뇌경색 병변은 크기가 작거나 여러 위치에 흩어져 있을 수 있어 단순한 voxel 정확도만으로는 임상적으로 유용한 분할인지 판단하기 어렵다.

이 프로젝트의 첫 번째 질문은 다음과 같다.

> 같은 3D 분할 모델에서 DWI만 사용할 때와 ADC·FLAIR를 추가할 때 성능이 어떻게 달라지는가?

두 번째 확장 질문은 다음과 같다.

> 뇌 MRI Foundation Model의 사전학습 지식이 DWI·ADC 뇌경색 분할에 도움이 되는가? 전체 미세조정 대신 LoRA를 사용해도 비슷한 성능을 얻을 수 있는가?

## 3. 데이터

주 데이터는 ISLES'22 공개 학습 데이터다.

- 환자 수: 250건
- 입력 영상: DWI, ADC, FLAIR
- 정답: 허혈성 뇌졸중 병변 마스크
- 형식: 3D NIfTI 의료영상
- 현재 분할: 학습 200건, 검증 50건
- 분할 단위: 환자 단위

슬라이스나 패치 단위로 무작위 분할하지 않는다. 같은 환자의 영상 또는 증강 결과가 학습과 검증에 동시에 들어가면 데이터 누수가 발생한다.

### 영상 정렬 상태

실제 데이터 감사 결과는 다음과 같다.

- DWI와 ADC: 250건 모두 같은 voxel grid
- DWI와 마스크: 250건 모두 같은 voxel grid
- 원본 FLAIR와 DWI: 같은 voxel grid가 아님

따라서 FLAIR가 포함된 실험은 원본을 직접 쌓지 않고, FLAIR를 DWI 공간으로 정합한 파생 영상을 사용했다. 원본 데이터는 수정하지 않는다.

## 4. 현재까지 완료된 실험

동일한 nnU-Net에서 입력 채널만 바꾸는 Early Fusion 실험을 완료했다.

| Dataset ID | 입력 | mean voxel Dice |
|---|---|---:|
| 501 | DWI | 0.7653 |
| 502 | DWI + ADC | 0.7571 |
| 503 | DWI + FLAIR | 0.7626 |
| 504 | DWI + ADC + FLAIR | **0.7665** |

공통 학습 조건은 다음과 같다.

- 모델: nnU-Net v2 `PlainConvUNet`, 3D full-resolution
- epoch: 250
- fold: 0
- spacing: 2 × 2 × 2 mm
- patch: 80 × 96 × 80
- batch size: 2
- GPU: RTX 2080
- 네 실험에서 분할, augmentation, loss와 학습 예산을 동일하게 유지

### 현재 결과의 올바른 해석

세 모달리티 모델의 Dice가 가장 높지만 DWI 단독과 차이는 약 `+0.0012`에 불과하다. DWI+ADC는 이번 분할에서 DWI 단독보다 낮았다.

아직 다음 결론을 내려서는 안 된다.

- ADC가 필요 없다는 결론
- 세 모달리티 모델이 확실히 더 좋다는 결론
- FLAIR가 작은 병변 검출에 도움이 된다는 결론

초기 결과는 한 번의 환자 분할에서 얻은 평균 voxel Dice였다. 이후 병변 단위 지표와 환자 단위 paired bootstrap 분석을 추가했으며, [상세 평가 보고서](../reports/flair_ablation_report.md)에 결과를 정리했다. FLAIR를 직접 추가한 두 비교에서 Dice·병변 F1·작은 병변 recall의 95% 구간은 모두 0을 포함했다. 반복 실험과 외부 검증이 없으므로 작은 차이를 일반화할 수는 없다.

## 5. nnU-Net을 먼저 사용한 이유

nnU-Net은 RTX 2080 때문에 어쩔 수 없이 선택한 모델이 아니다. 연구 질문에 가장 잘 맞기 때문에 기준 모델로 선택했다.

1. 모델 구조와 학습 조건을 고정하고 입력 채널만 바꿀 수 있다.
2. 모달리티 효과를 다른 구조·앙상블·사전학습 효과와 분리하기 쉽다.
3. 의료영상 분할에서 널리 사용되는 강력하고 재현 가능한 기준선이다.
4. Foundation Model과 LoRA가 실제로 이득인지 판단할 비교 기준을 제공한다.
5. 제한된 GPU에서도 전체 실험을 완료할 수 있다.

GPU가 더 좋아지더라도 nnU-Net 실험을 없애지 않는다. 더 큰 모델은 nnU-Net을 대체하는 것이 아니라 그 위에 추가하는 비교 대상이다.

## 6. 검토한 모델과 역할

### SEALS

- ISLES'22 상위권 nnU-Net 기반 방법
- DWI와 ADC를 사용하는 공개 추론 코드와 사전학습 가중치 제공
- 프로젝트와 직접 관련된 강력한 선행 기준
- 전체 제출 모델을 그대로 쓰기보다 nnU-Net 기반 modality ablation을 설계하는 참고 모델

### 정현수 교수님 제1저자 논문: Jeong et al. (2024)

- 논문명: *Robust Ensemble of Two Different Multimodal Approaches to Segment 3D Ischemic Stroke Segmentation Using Brain Tumor Representation Among Multiple Center Datasets*
- 제1저자: Hyunsu Jeong(정현수)
- 게재처: *Journal of Imaging Informatics in Medicine*, 37(5), 2375–2389, 2024
- DOI: <https://doi.org/10.1007/s10278-024-01099-6>
- 공개 원문: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11522214/>

이 문서에서 ‘정현수 교수님 논문’은 위 논문만을 의미한다. 교수님의 성함과 논문의 제1저자 영문명이 일치한다는 팀 제공 정보를 근거로 표기하며, 확인하지 않은 소속이나 직함 정보는 추가하지 않는다.

이 연구는 DWI와 ADC를 사용하는 두 가지 멀티모달 방법을 비교한다. joint training은 학습 단계에서 DWI와 ADC를 활용하지만 추론에는 DWI만 사용한다. channel-wise training은 DWI와 ADC를 입력 채널로 결합해 학습과 추론에 모두 사용한다. 또한 BraTS’21 뇌종양 표현을 이용한 전이학습과 두 접근의 앙상블을 다룬다.

현재 프로젝트는 이 논문의 전체 앙상블을 바로 재현하기 전에, 동일한 nnU-Net에서 DWI와 ADC의 단순 Early Fusion 효과를 먼저 측정한다. 이후 joint training, channel-wise training, 뇌종양 전이학습의 기여를 각각 분리해 비교하는 것이 직접적인 확장 방향이다.

### DeepISLES

- SEALS, NVAUTO, Factorizer 계열을 결합한 앙상블
- 최고 성능에 가까운 고정 benchmark로 적합
- 여러 모델과 후처리 효과가 섞이므로 모달리티 자체의 효과를 분석하는 주 모델로는 부적합

### Factorizer

- Swin Factorizer와 Res-U-Net 기반 ISLES'22 방법
- nnU-Net과 Transformer 계열 구조를 비교할 때 후보
- 전체 앙상블보다 단일 모델만 비교하는 것이 해석과 계산량 면에서 적절

### MoME / MoME+

- 여러 뇌 병변과 MRI modality를 다루는 3D 뇌 병변 Foundation Model
- DWI 및 뇌 병변과의 도메인 적합성이 높음
- 멀티모달 입력에는 MoME+ 구현을 우선 검토해야 함
- 공개 사전학습에 ISLES'22가 포함됐다면 현재 검증 환자를 이미 보았을 가능성이 있어 평가 누수에 주의해야 함

MoME를 사용할 때는 사전학습 데이터에 ISLES'22가 포함되었는지와 ISLES 제외 체크포인트를 구할 수 있는지를 먼저 확인한다.

### BrainSegFounder

- UK Biobank 등의 뇌 MRI를 활용한 3D 뇌 특화 사전학습 모델
- SwinUNETR 계열이므로 attention 계층에 LoRA를 적용하기 비교적 자연스러움
- ISLES'22 사전학습 중복 위험이 MoME보다 낮은 후보
- 주로 구조적 T1·T2 계열 MRI로 학습되어 DWI·ADC와 영상 특성이 다르다는 한계가 있음

이 모달리티 차이는 단점인 동시에 “구조 MRI 사전학습이 DWI·ADC로 전이되는가?”라는 연구 질문이 될 수 있다.

### MedSAM3 계열

- 프롬프트 또는 텍스트 기반 의료영상 분할 후보
- 자동 3D 분할을 위해 프롬프트 생성 방법을 추가로 설계해야 함
- 정답에서 생성한 box나 point를 쓰면 oracle 실험이 되므로 완전 자동 분할과 직접 비교하면 안 됨
- 현재 프로젝트의 첫 Foundation Model로는 우선순위가 낮음

## 7. GPU가 충분할 때의 추천 방향

가장 추천하는 방향은 다음과 같다.

> **nnU-Net 기준선은 유지하고, ISLES 데이터 중복이 없는 뇌 MRI Foundation Model에서 DWI와 DWI+ADC의 전체 미세조정 및 LoRA를 비교한다.**

### 우선순위 1: 공정성을 중시한 연구

BrainSegFounder 또는 ISLES가 제외된 사전학습 체크포인트를 사용한다.

- 장점: 평가 데이터 중복 위험이 낮음
- 핵심 질문: 뇌 MRI 사전학습이 DWI·ADC 병변 분할로 전이되는가?
- 추천 비교: Full fine-tuning vs LoRA

### 우선순위 2: 멀티모달 적합성을 중시한 연구

ISLES 제외 체크포인트를 확보할 수 있다면 MoME+를 사용한다.

- 장점: 뇌 병변과 다양한 modality에 특화
- 핵심 질문: modality expert 기반 사전학습이 DWI+ADC 융합에 도움이 되는가?
- 주의: ISLES가 포함된 체크포인트는 독립적인 성능 비교에 사용하지 않음

### 최고 성능만 필요한 경우

DeepISLES를 실행해 외부 benchmark로 사용한다. 다만 앙상블 결과이므로 본 프로젝트의 모달리티 또는 LoRA 효과에 대한 직접적인 근거로 사용하지 않는다.

## 8. 추천 Foundation Model 실험표

Foundation Model 단계에서는 범위를 DWI와 DWI+ADC로 줄인다. FLAIR까지 다시 모든 조합에 적용하면 실험 수가 과도하게 늘고 연구 질문이 흐려질 수 있다.

| 실험 | 모델 | 입력 | 학습 방식 | 목적 |
|---|---|---|---|---|
| B1 | nnU-Net | DWI | 전체 학습 | 지도학습 기준선 |
| B2 | nnU-Net | DWI+ADC | 전체 학습 | ADC 추가 효과 기준선 |
| F1 | Foundation Model | DWI | Full fine-tuning | 사전학습 효과 확인 |
| F2 | Foundation Model | DWI+ADC | Full fine-tuning | 사전학습 모델에서 ADC 효과 확인 |
| L1 | Foundation Model | DWI | LoRA | PEFT 효율 확인 |
| L2 | Foundation Model | DWI+ADC | LoRA | 멀티모달 LoRA 효과 확인 |

계산 자원이 충분하면 가장 유망한 LoRA 조건에서만 rank를 비교한다.

- `r=4`
- `r=8`
- `r=16`

모든 rank와 모든 적용 계층을 조합하는 대규모 탐색은 피한다.

## 9. 멀티모달 입력 방법

첫 구현은 Early Fusion으로 한다.

```text
DWI ──┐
      ├─ channel concatenation ─ encoder ─ decoder ─ lesion mask
ADC ──┘
```

입력 tensor 예시는 다음과 같다.

```text
[batch, 2, depth, height, width]
channel 0: DWI
channel 1: ADC
```

사전학습 모델의 원래 입력 채널 수와 다르면, 작은 `1×1×1` modality stem을 앞에 두어 입력을 모델의 기대 채널 수로 변환하는 방식을 검토한다. 입력 계층을 임의 초기화한 효과와 LoRA 효과가 혼동되지 않도록 초기화 방법과 학습 범위를 기록한다.

Early Fusion 결과를 확보한 뒤에만 다음 구조를 추가 후보로 고려한다.

- 모달리티별 입력 stem
- 모달리티별 encoder 일부 분리
- 중간 feature fusion

## 10. LoRA 적용 제안

Transformer 기반 모델이라면 다음 순서로 적용한다.

1. attention의 query, key, value projection
2. attention output projection
3. 필요할 때만 MLP 계층

처음에는 pretrained encoder의 기존 파라미터를 동결하고 다음 부분만 학습한다.

- LoRA 파라미터
- 새로 추가한 modality stem
- segmentation decoder 또는 output head

각 실험에서 반드시 기록할 항목은 다음과 같다.

- LoRA rank
- scaling 또는 alpha
- dropout
- 적용 계층 이름
- encoder·decoder 동결 범위
- 전체 파라미터와 학습 가능한 파라미터 수
- peak GPU memory
- 학습 시간과 추론 시간

## 11. 평가 지표

평균 Dice 하나만 보고 결론을 내리지 않는다.

### 분할 품질

- voxel-wise Dice
- HD95
- 병변 부피 오차

### 병변 검출

- lesion-wise precision, recall, F1
- small-lesion recall
- 환자당 false-positive lesion 수

병변 단위 평가에서는 연결성, 최소 병변 크기, 예측과 정답 병변의 매칭 규칙, 빈 마스크 처리 규칙을 문서화해야 한다.

### 효율성

- 학습 가능한 파라미터 수
- peak GPU memory
- epoch당 또는 전체 학습 시간
- 환자당 추론 시간

### 통계 분석

- 환자별 paired 비교
- bootstrap 95% 신뢰구간
- 가능하면 반복 seed 또는 5-fold 교차검증
- 작은 평균 차이는 신뢰구간과 환자별 분포를 함께 확인

## 12. 성공 판단 기준

Foundation Model 또는 LoRA가 성공적이라고 판단하려면 단순히 평균 Dice가 조금 높은 것만으로는 부족하다.

### Full fine-tuning의 성공 조건

- 동일한 환자 분할에서 nnU-Net보다 병변 단위 F1 또는 small-lesion recall이 개선됨
- 성능 차이가 일부 대형 병변에만 의존하지 않음
- 사전학습 데이터와 검증 데이터의 중복이 없음

### LoRA의 성공 조건

- Full fine-tuning과 비슷한 성능을 유지
- 학습 가능한 파라미터와 GPU 메모리가 크게 감소
- 반복 실험에서 결과가 안정적임

예를 들어 LoRA가 Full fine-tuning 대비 Dice 차이를 1~2 percentage point 이내로 유지하면서 학습 가능한 파라미터를 10% 이하로 줄인다면 실용적인 결과가 될 수 있다. 이 값은 사전에 정한 절대 규칙이 아니라 보고서에서 검토할 예시 기준이다.

## 13. 예상되는 결과와 해석

### ADC 추가가 성능을 높이는 경우

- ADC가 DWI의 T2 shine-through와 실제 확산 제한을 구분하는 데 도움을 주었을 가능성
- 작은 병변 또는 경계에서 효과가 집중되는지 확인

### ADC 추가가 성능을 낮추는 경우

- 두 modality의 intensity scale 차이
- ADC 노이즈 또는 scanner별 분포 차이
- 단순 Early Fusion의 한계
- 데이터 수 대비 입력 복잡도 증가

이 경우에도 “ADC가 쓸모없다”고 결론 내리지 않는다. 정규화와 융합 방식이 ADC 정보를 활용하지 못했을 가능성을 분석한다.

### FLAIR 효과가 작은 경우

- 발병 초기에는 FLAIR 병변이 뚜렷하지 않을 수 있음
- DWI와 FLAIR 정합 오차가 이득을 상쇄할 수 있음
- 시간 정보가 없는 상태에서 환자별 FLAIR 유용성이 다를 수 있음

### LoRA가 Full fine-tuning보다 좋은 경우

- 작은 데이터에서 과적합을 줄였을 가능성
- 사전학습 표현을 더 잘 보존했을 가능성

### LoRA가 낮은 경우

- DWI·ADC로의 modality shift가 커서 전체 파라미터 적응이 필요할 가능성
- LoRA를 적용한 계층 또는 rank가 적절하지 않았을 가능성

## 14. 권장 작업 순서

1. 완료: 네 nnU-Net 모델의 예측과 환자 단위 split 확인.
2. 완료: lesion-wise F1, 1 mL 미만 병변 recall, HD95, 병변 부피 오차 계산.
3. 완료: 같은 환자끼리 paired 비교하고 bootstrap 신뢰구간 계산.
4. 다음: 오류 사례를 병변 크기와 다발성 여부에 따라 시각화한다.
5. Foundation Model의 정확한 논문, 체크포인트, 입력 채널과 사전학습 데이터를 확인한다.
6. ISLES 중복이 없는 모델을 주 Foundation Model로 확정한다.
7. DWI와 DWI+ADC Full fine-tuning을 실행한다.
8. 같은 모델에 기본 LoRA를 적용한다.
9. 가장 유망한 조건만 rank 또는 적용 계층 ablation을 수행한다.
10. 성능과 연산 비용을 함께 비교해 최종 결론을 작성한다.

## 15. GPU에 따른 현실적인 범위

| GPU | 권장 범위 |
|---|---|
| 8 GB급 | nnU-Net 중심, 작은 patch와 batch, 단일 fold |
| 12~16 GB | nnU-Net 반복 실험, 단일 Transformer, 제한적인 LoRA |
| 24 GB급 | 3D Foundation Model Full fine-tuning과 LoRA 비교 |
| 40~80 GB급 | 더 큰 3D 모델, 5-fold, 다중 seed, MoME 계열 전체 미세조정 |

GPU가 커져도 학기 일정, 코드 재현성, 데이터 누수 검증과 평가 코드 작성 시간이 사라지는 것은 아니다. 실험 수를 무한히 늘리기보다 연구 질문에 직접 답하는 비교를 우선한다.

## 16. 최종 추천 연구 제목

현재 방향을 반영한 권장 제목은 다음과 같다.

> **DWI–ADC 멀티모달 급성 뇌경색 분할에서 3D nnU-Net과 뇌 MRI Foundation Model의 전체 미세조정 및 LoRA 비교**

영문 예시는 다음과 같다.

> **Comparing Full Fine-Tuning and LoRA of a Brain MRI Foundation Model for DWI–ADC Multimodal Acute Ischemic Stroke Lesion Segmentation**

## 17. 핵심 원칙

- nnU-Net과 Foundation Model을 같은 개념으로 취급하지 않는다.
- 입력 모달리티, 사전학습과 미세조정 방법의 효과를 분리한다.
- 테스트 또는 검증 데이터를 전처리 설정과 threshold 선택에 사용하지 않는다.
- 실행하지 않은 코드를 실행했다고 표현하지 않는다.
- 논문 수치와 로컬 재현 결과를 구분한다.
- 사전학습 데이터에 ISLES가 포함됐다면 독립 평가 결과로 주장하지 않는다.
- 성능뿐 아니라 메모리, 시간과 학습 가능 파라미터 수를 함께 보고한다.
- 환자 영상과 파생 의료영상은 GitHub에 올리지 않는다.

## 18. 관련 자료

- ISLES'22: <https://github.com/ezequieldlrosa/isles22>
- nnU-Net: <https://github.com/MIC-DKFZ/nnUNet>
- SEALS: <https://github.com/Tabrisrei/ISLES22_SEALS>
- DeepISLES: <https://github.com/ezequieldlrosa/DeepIsles>
- Factorizer: <https://github.com/pashtari/factorizer-isles22>
- MoME, MICCAI 2024: <https://papers.miccai.org/miccai-2024/015-Paper0143.html>
- BrainSegFounder: <https://github.com/lab-smile/BrainSegFounder>
- 정현수 교수님 제1저자 논문, Jeong et al. (2024): <https://doi.org/10.1007/s10278-024-01099-6>
