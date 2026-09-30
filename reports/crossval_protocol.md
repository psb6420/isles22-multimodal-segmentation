# ISLES'22 5-fold 교차검증 프로토콜

## 목적과 현재 상태

한 번의 200/50 분할에서 보인 작은 성능 차이가 특정 검증 환자 구성에 좌우되는지 확인한다. 2026-09-30 기준 기존 fold 0 외의 학습을 진행 중이며, **5-fold 결과는 아직 없다**.

## 환자 분할

- 250명 모두를 환자 단위로 5개 검증 fold에 한 번씩 배정한다. 매 fold는 200명 학습/50명 검증이다.
- 기존 fold 0의 200/50 구성은 그대로 유지한다. 나머지 200명을 고정 seed 2026으로 섞어 각각 50명인 fold 1~4의 검증 집합을 만든다.
- 네 모달리티 조건은 완전히 같은 fold를 사용한다. 정답 마스크와 영상의 복제본이 다른 fold로 갈라지지 않도록 환자 ID로 검사한다.
- 검증 fold의 센터 1/센터 2 환자 수는 순서대로 40/10, 40/10, 40/10, 37/13, 41/9이다. 중앙 병변 부피는 6.32, 6.95, 6.12, 8.16, 6.44 mL이다. 이는 분할 확인용 기술통계이며 성능 결과가 아니다.
- 실제 환자 ID가 담긴 분할 JSON은 로컬에 보관하고 공개 저장소에 올리지 않는다.

## 학습과 평가

- 동일한 nnU-Net v2 3D full-resolution 구조, spacing 2 mm, patch 80×96×80, batch 2, 250 epochs, 손실·증강·후처리를 고정한다.
- 입력만 DWI, DWI+ADC, DWI+FLAIR, DWI+ADC+FLAIR로 바꾼다. 각 fold 안에서 네 조건을 같은 GPU에서 비교한다.
- 모든 최종 학습 결과는 RTX 2080 데스크톱을 사용한다. GTX 1070 시험은 과열로 첫 epoch 도중 중단됐으며 평가에 포함하지 않는다.
- 각 fold의 검증 마스크는 해당 fold에서 학습하지 않은 50명에만 적용한다. 250명의 out-of-fold 예측으로 Dice, 병변 F1, 1 mL 미만 병변 recall, HD95, 절대 병변 부피 오차를 계산한다.
- 환자별 FLAIR 효과는 DWI+FLAIR−DWI 및 DWI+ADC+FLAIR−DWI+ADC로 계산한다. 다섯 fold의 차이를 모두 보여주고 평균·표준편차를 기술한다. 다섯 fold만으로 강한 통계적 유의성 주장을 하지 않는다.
- fold 0 결과를 보고 연구 방향을 정했으므로 이 교차검증도 완전히 독립적인 최종 테스트는 아니다. 공식 비공개 테스트셋과 외부 기관 일반화 성능으로 표현하지 않는다.

## 실행과 산출물

`scripts/prepare_fivefold_splits.py`로 동일 분할을 설치하고, `scripts/run_fivefold_queue.py`로 fold 1~4를 순차 학습한다. 완료 후 `scripts/finalize_crossval.py`가 20개 모델-fold 결과를 확인하고 `scripts/evaluate_crossval.py`를 실행한다. 환자별 표는 로컬 전용이며, 공개 보고서에는 집계만 포함한다.

nnU-Net 공식 5-fold 학습 안내: <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how-to/train-models.md>
