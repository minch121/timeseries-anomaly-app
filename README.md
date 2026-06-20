# 🔍 다변량 시계열 이상탐지 웹앱

임의의 다변량 시계열 CSV를 업로드하면 자동으로 분석하고,
**Scorer → Detector → Aggregator** 파이프라인으로 이상을 탐지하는 웹앱입니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 탭 구성

| 탭 | 기능 |
|---|---|
| ① 데이터 개요 | 시계열 시각화, 기초 통계, 상관관계, 결측 현황 |
| ② 전처리 & 분석 | 결측치/디노이징, ADF·Ljung-Box, ACF/PACF, STL 분해 |
| ③ 이상탐지 | **Scorer → Detector → Aggregator 파이프라인** |
| ④ 평가 대시보드 | AUC-ROC / PR / 혼동행렬 또는 임계값 민감도·Scorer 일치도 |
| ⑤ 결과 & 내보내기 | 이상치 상세표 + CSV 다운로드 |

## 이상탐지 파이프라인

- **Scorer** — 시계열에 이상 점수 부여
  - `NormScorer` : 예측 오차의 크기 (간단·해석 용이, 점/전역 이상)
  - `KMeansScorer` : 오차 윈도우 클러스터링 후 중심까지 거리 (복잡한 패턴)
  - `WassersteinScorer` : 윈도우 분포와 전체 분포의 거리 (분포·레짐 변화)
- **AnomalyModel(예측 기반)** — 예측모델(Ridge/RandomForest, 시·요일 covariate) 잔차에 Scorer 적용
- **Detector** — 점수를 분위수/σ 임계값으로 이진화
- **Aggregator** — 다변량 이진 결과를 OR / AND / count 로 통합

## 평가 (탐지가 적절한지 판단)

- **정답 라벨 컬럼(0/1)이 있으면**: AUC-ROC, AUC-PR, Precision/Recall/F1, ROC·PR 곡선, 혼동행렬
- **라벨이 없으면**: 이상 점수 분포, 임계값 민감도 곡선, Scorer 간 일치도(Jaccard)

> 샘플 데이터에는 `is_anomaly` 라벨이 포함되어 있어 정량 평가를 바로 확인할 수 있습니다.
> 직접 업로드한 CSV에 정답 컬럼이 있으면 사이드바에서 라벨 컬럼으로 지정하세요.

## CSV 형식

- 시간 컬럼 1개(datetime) + 수치형 변수 여러 개
- (선택) 0/1 정답 라벨 컬럼 → 사이드바에서 지정 시 정량 평가 활성화
- **파일을 바꾸면 자동으로 새로 분석**됩니다.

## Streamlit Community Cloud 배포

1. GitHub 저장소에 push
2. [share.streamlit.io](https://share.streamlit.io) → 저장소 연결 → `app.py` 선택 → Deploy
3. 생성된 공개 URL 제출

## 파일 구조

```
├── app.py                  # 메인 Streamlit 앱 (5탭)
├── requirements.txt        # 패키지 의존성 (버전 고정)
├── generate_sample_data.py # 샘플 데이터 생성 (라벨 포함)
├── sample_data/
│   ├── factory_sensor_data.csv
│   └── server_monitoring_data.csv
└── utils/
    ├── preprocessing.py    # 결측치/디노이징
    ├── analysis.py         # ADF·Ljung-Box·ACF/PACF·STL
    ├── scorers.py          # 이상탐지 엔진 (Scorer/Detector/Aggregator/평가)
    └── visualization.py    # Plotly 시각화
```
