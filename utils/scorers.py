"""
시계열 이상탐지 엔진

4모듈 파이프라인을 scikit-learn / numpy로 구현:
  1) Scorer      : 시계열 -> 이상 점수(anomaly score)
  2) Detector    : 이상 점수 -> 이진 라벨(0/1)
  3) Aggregator  : 다변량 이진 라벨 -> 단변량 통합 라벨
  4) AnomalyModel: 예측모델(forecasting) 잔차를 Scorer로 평가

Scorer 3종
  - NormScorer        : 예측 오차의 크기 |error|  (간단/해석 쉬움)
  - KMeansScorer      : 오차 윈도우를 클러스터링, 중심까지 거리 (복잡한 패턴)
  - WassersteinScorer : 윈도우 분포와 전체 분포의 거리 (분포 변화 감지)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from scipy.stats import wasserstein_distance


# ════════════════════════════════════════════════════════════
# AnomalyModel: 예측 기반 잔차 계산 (ForecastingAnomalyModel 개념)
# ════════════════════════════════════════════════════════════

def _build_lag_matrix(values: np.ndarray, lags: int,
                      covariates: Optional[np.ndarray] = None):
    """lag 특성 행렬 구성. covariates(시/요일 등)는 시점 t의 값을 그대로 사용."""
    n = len(values)
    X, y, idx = [], [], []
    for t in range(lags, n):
        row = list(values[t - lags:t])
        if covariates is not None:
            row.extend(covariates[t])
        X.append(row)
        y.append(values[t])
        idx.append(t)
    return np.array(X), np.array(y), np.array(idx)


def forecasting_residuals(
    series: pd.Series,
    time_index=None,
    lags: int = 24,
    train_ratio: float = 0.5,
    model_type: str = "ridge",
    use_calendar: bool = True,
) -> Dict:
    """
    예측모델을 train 구간에 적합 후 전체 구간을 예측 -> 잔차(error) 산출.
    lag 특성 + 시/요일 covariate를 사용한 회귀 기반 예측.

    반환: {residual(pd.Series), fitted, actual, error}
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    lags = max(2, min(lags, n // 4))

    # 달력 covariates (hour, dayofweek) — 시간 인덱스가 datetime일 때만
    covariates = None
    if use_calendar and time_index is not None:
        ti = pd.to_datetime(pd.Series(time_index), errors="coerce")
        if ti.notna().all():
            hour = ti.dt.hour.to_numpy()
            dow = ti.dt.dayofweek.to_numpy()
            covariates = np.column_stack([
                np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            ])

    X, y, idx = _build_lag_matrix(values, lags, covariates)
    if len(X) < 10:
        return {"err": "데이터가 부족하여 예측 모델을 적합할 수 없습니다."}

    n_train = max(5, int(len(X) * train_ratio))

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1)
    else:
        model = Ridge(alpha=1.0)

    model.fit(X[:n_train], y[:n_train])
    pred = model.predict(X)

    fitted = pd.Series(np.nan, index=series.index)
    fitted.iloc[idx] = pred
    error = pd.Series(np.nan, index=series.index)
    error.iloc[idx] = y - pred

    return {
        "fitted": fitted,
        "actual": series,
        "error": error,            # 부호 있는 잔차
        "residual": error,
        "n_train": n_train,
        "lags": lags,
    }


# ════════════════════════════════════════════════════════════
# Scorers
# ════════════════════════════════════════════════════════════

def norm_scorer(error: pd.Series) -> pd.Series:
    """NormScorer: 예측 오차의 절댓값 크기를 이상 점수로."""
    return error.abs()


def kmeans_scorer(error: pd.Series, window: int = 10, n_clusters: int = 4) -> pd.Series:
    """
    KMeansScorer: 오차의 슬라이딩 윈도우를 KMeans로 군집화하고
    가장 가까운 중심까지의 거리를 이상 점수로.
    """
    e = error.fillna(0.0).to_numpy()
    n = len(e)
    window = max(2, min(window, n // 3))
    windows, centers_idx = [], []
    for t in range(window, n + 1):
        windows.append(e[t - window:t])
        centers_idx.append(t - 1)          # 윈도우의 마지막 시점에 점수 부여
    W = np.array(windows)
    if len(W) < n_clusters:
        n_clusters = max(1, len(W))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    km.fit(W)
    dist = np.min(km.transform(W), axis=1)

    score = pd.Series(np.nan, index=error.index)
    score.iloc[centers_idx] = dist
    return score.bfill().ffill()


def wasserstein_scorer(series: pd.Series, window: int = 30) -> pd.Series:
    """
    WassersteinScorer: 각 슬라이딩 윈도우의 분포와 전체 분포 간
    Wasserstein 거리를 이상 점수로 (분포 변화 감지).
    """
    x = series.ffill().bfill().to_numpy(dtype=float)
    n = len(x)
    window = max(5, min(window, n // 3))
    ref = x
    score = pd.Series(np.nan, index=series.index)
    for t in range(window, n + 1):
        w = x[t - window:t]
        score.iloc[t - 1] = wasserstein_distance(w, ref)
    return score.bfill().ffill()


SCORERS = {
    "NormScorer": "예측 오차의 크기 — 간단하고 해석 쉬움 (점/전역 이상)",
    "KMeansScorer": "오차 패턴 클러스터링 — 복잡한 패턴 이상에 유리",
    "WassersteinScorer": "분포 간 거리 — 분포 변화(레짐 변화) 감지에 효과적",
}


def compute_score(
    series: pd.Series,
    scorer: str,
    time_index=None,
    forecasting: bool = True,
    lags: int = 24,
    train_ratio: float = 0.5,
    model_type: str = "ridge",
    use_calendar: bool = True,
    window: int = 10,
    n_clusters: int = 4,
) -> Dict:
    """
    하나의 변수 + 하나의 Scorer로 이상 점수 시계열을 계산.
    forecasting=True 면 예측모델 잔차에 Scorer 적용(AnomalyModel),
    False 면 원시 시계열에 직접 적용.
    """
    info = {"scorer": scorer, "forecasting": forecasting}

    if scorer == "WassersteinScorer":
        # 분포 변화는 원시 시계열에 직접 적용하는 것이 자연스러움
        score = wasserstein_scorer(series, window=window)
        info["score"] = score
        return info

    if forecasting:
        fc = forecasting_residuals(
            series, time_index=time_index, lags=lags,
            train_ratio=train_ratio, model_type=model_type, use_calendar=use_calendar,
        )
        if fc.get("err"):
            return {"error": fc["err"], "scorer": scorer}
        error = fc["error"]
        info.update({"fitted": fc["fitted"], "n_train": fc["n_train"], "lags": fc["lags"]})
    else:
        # 직접 적용: 평균 대비 편차를 '오차'로 간주
        error = series - series.rolling(max(2, window), min_periods=1, center=True).mean()

    if scorer == "NormScorer":
        score = norm_scorer(error)
    elif scorer == "KMeansScorer":
        score = kmeans_scorer(error, window=window, n_clusters=n_clusters)
    else:
        raise ValueError(f"알 수 없는 Scorer: {scorer}")

    info["score"] = score
    info["resid"] = error
    return info


# ════════════════════════════════════════════════════════════
# Detector: 이상 점수 -> 이진 라벨
# ════════════════════════════════════════════════════════════

def threshold_from_scores(score: pd.Series, method: str = "quantile",
                          q: float = 0.95, n_sigma: float = 3.0) -> float:
    s = score.dropna()
    if method == "quantile":
        return float(s.quantile(q))
    elif method == "sigma":
        return float(s.mean() + n_sigma * s.std())
    return float(s.quantile(q))


def detect(score: pd.Series, threshold: float) -> pd.Series:
    """ThresholdDetector: 점수 >= 임계값이면 1(이상)."""
    return (score >= threshold).astype(int)


# ════════════════════════════════════════════════════════════
# Aggregator: 다변량 이진 라벨 통합
# ════════════════════════════════════════════════════════════

def aggregate(binary_df: pd.DataFrame, method: str = "or", min_count: int = 2) -> pd.Series:
    """
    OrAggregator / AndAggregator / count.
    binary_df: 컬럼=변수, 값=0/1
    """
    counts = binary_df.sum(axis=1)
    if method == "or":
        return (counts >= 1).astype(int)
    elif method == "and":
        return (counts == binary_df.shape[1]).astype(int)
    elif method == "count":
        return (counts >= min_count).astype(int)
    return (counts >= 1).astype(int)


# ════════════════════════════════════════════════════════════
# 평가 지표 (eval_metric_from_scores 개념)
# ════════════════════════════════════════════════════════════

def evaluate_scores(score: pd.Series, labels: pd.Series) -> Dict:
    """라벨이 있을 때 AUC-ROC / AUC-PR 등 점수 기반 정량 평가."""
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    mask = score.notna() & labels.notna()
    s = score[mask].to_numpy()
    y = labels[mask].astype(int).to_numpy()
    out = {"n": int(mask.sum()), "n_pos": int(y.sum())}
    if y.sum() == 0 or y.sum() == len(y):
        out["error"] = "라벨에 이상/정상이 모두 존재해야 AUC를 계산할 수 있습니다."
        return out
    out["auc_roc"] = float(roc_auc_score(y, s))
    out["auc_pr"] = float(average_precision_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    out["roc"] = (fpr, tpr)
    out["pr"] = (rec, prec)
    return out


def classification_metrics(pred: pd.Series, labels: pd.Series) -> Dict:
    """임계값으로 이진화된 예측에 대한 정밀도/재현율/F1 + 혼동행렬."""
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    mask = pred.notna() & labels.notna()
    p = pred[mask].astype(int).to_numpy()
    y = labels[mask].astype(int).to_numpy()
    cm = confusion_matrix(y, p, labels=[0, 1])
    return {
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "confusion": cm,   # [[TN, FP], [FN, TP]]
    }


def threshold_sweep(score: pd.Series, quantiles=None) -> pd.DataFrame:
    """라벨이 없을 때: 임계값(분위수)에 따른 탐지 개수 민감도."""
    if quantiles is None:
        quantiles = np.round(np.arange(0.80, 0.995, 0.01), 3)
    s = score.dropna()
    rows = []
    for q in quantiles:
        thr = float(s.quantile(q))
        rows.append({"quantile": q, "threshold": thr, "n_anomalies": int((s >= thr).sum())})
    return pd.DataFrame(rows)
