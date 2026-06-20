"""
시각화 모듈 (Plotly 기반)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional


# ═══════════════════════════════════════════
# 공통 레이아웃 설정
# ═══════════════════════════════════════════

COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "success": "#10B981",
    "muted": "#6B7280",
    "bg": "#FFFFFF",
    "grid": "#F3F4F6"
}

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Pretendard, sans-serif", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="x unified"
)


def _apply_layout(fig, title: str = "", height: int = 400):
    fig.update_layout(**LAYOUT_DEFAULTS, title=title, height=height)
    return fig


# ═══════════════════════════════════════════
# Tab 1: 데이터 개요
# ═══════════════════════════════════════════

def plot_time_series(df: pd.DataFrame, time_col: str, value_cols: List[str], title: str = "시계열 데이터") -> go.Figure:
    """여러 변수의 시계열 라인 차트"""
    fig = make_subplots(
        rows=len(value_cols), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=value_cols
    )
    
    colors = px.colors.qualitative.Set2
    for i, col in enumerate(value_cols):
        fig.add_trace(
            go.Scatter(x=df[time_col], y=df[col], name=col,
                      line=dict(color=colors[i % len(colors)], width=1.5),
                      hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>"),
            row=i+1, col=1
        )
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=250 * len(value_cols),
        title=title,
        showlegend=False
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, columns: List[str]) -> go.Figure:
    """변수 간 상관관계 히트맵"""
    corr = df[columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr.values, 3),
        texttemplate="%{text}",
        textfont=dict(size=13),
        hoverongaps=False
    ))
    
    return _apply_layout(fig, "변수 간 상관관계", height=400)


def plot_missing_values(missing_info: pd.DataFrame) -> go.Figure:
    """결측치 현황 바 차트"""
    fig = go.Figure(go.Bar(
        x=missing_info["컬럼"],
        y=missing_info["결측치 수"],
        text=missing_info["결측률(%)"].apply(lambda x: f"{x}%"),
        textposition="outside",
        marker_color=COLORS["accent"]
    ))
    return _apply_layout(fig, "결측치 현황", height=300)


# ═══════════════════════════════════════════
# Tab 2: 전처리 & 분석
# ═══════════════════════════════════════════

def plot_before_after(original: pd.Series, processed: pd.Series, time_index, title: str = "") -> go.Figure:
    """전처리 전후 비교"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index, y=original, name="원본",
                            line=dict(color=COLORS["muted"], width=1), opacity=0.6))
    fig.add_trace(go.Scatter(x=time_index, y=processed, name="처리 후",
                            line=dict(color=COLORS["primary"], width=1.5)))
    return _apply_layout(fig, title, height=300)


def plot_stl_decomposition(stl_result: Dict, time_index) -> go.Figure:
    """STL 분해 결과 시각화 (원본, 추세, 계절성, 잔차)"""
    components = [
        ("원본 (Observed)", stl_result["observed"], COLORS["primary"]),
        ("추세 (Trend)", stl_result["trend"], COLORS["secondary"]),
        ("계절성 (Seasonal)", stl_result["seasonal"], COLORS["success"]),
        ("잔차 (Residual)", stl_result["residual"], COLORS["accent"])
    ]
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                       subplot_titles=[c[0] for c in components])
    
    for i, (name, data, color) in enumerate(components):
        idx = data.index if hasattr(data, "index") else time_index[:len(data)]
        fig.add_trace(
            go.Scatter(x=idx, y=data, name=name, line=dict(color=color, width=1.5),
                      showlegend=False),
            row=i+1, col=1
        )
    
    fig.update_layout(**LAYOUT_DEFAULTS, height=700, title="STL 시계열 분해")
    return fig


def plot_acf_pacf(acf_vals: np.ndarray, pacf_vals: np.ndarray, conf: float) -> go.Figure:
    """ACF/PACF 플롯"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF (자기상관함수)", "PACF (편자기상관함수)"])
    
    lags = list(range(len(acf_vals)))
    
    # ACF
    for lag, val in zip(lags, acf_vals):
        fig.add_trace(go.Scatter(x=[lag, lag], y=[0, val], mode="lines",
                                line=dict(color=COLORS["primary"], width=2),
                                showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=lags, y=acf_vals, mode="markers",
                            marker=dict(color=COLORS["primary"], size=5),
                            showlegend=False, name="ACF"), row=1, col=1)
    fig.add_hline(y=conf, line_dash="dash", line_color=COLORS["danger"], row=1, col=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color=COLORS["danger"], row=1, col=1)
    
    # PACF
    for lag, val in zip(lags[:len(pacf_vals)], pacf_vals):
        fig.add_trace(go.Scatter(x=[lag, lag], y=[0, val], mode="lines",
                                line=dict(color=COLORS["secondary"], width=2),
                                showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=lags[:len(pacf_vals)], y=pacf_vals, mode="markers",
                            marker=dict(color=COLORS["secondary"], size=5),
                            showlegend=False, name="PACF"), row=1, col=2)
    fig.add_hline(y=conf, line_dash="dash", line_color=COLORS["danger"], row=1, col=2)
    fig.add_hline(y=-conf, line_dash="dash", line_color=COLORS["danger"], row=1, col=2)
    
    fig.update_layout(**LAYOUT_DEFAULTS, height=350)
    return fig


# ═══════════════════════════════════════════
# Tab 3: 이상탐지
# ═══════════════════════════════════════════

def plot_anomaly_detection(
    series: pd.Series,
    anomaly_indices: List,
    time_index,
    title: str = "",
    bounds: Optional[Dict] = None
) -> go.Figure:
    """이상탐지 결과 시각화 - 원본 시계열 위에 이상치 마킹"""
    fig = go.Figure()
    
    # 원본 시계열
    fig.add_trace(go.Scatter(
        x=time_index, y=series,
        name="원본 데이터", mode="lines",
        line=dict(color=COLORS["primary"], width=1.5)
    ))
    
    # 이상치 포인트
    if len(anomaly_indices) > 0:
        anomaly_mask = series.index.isin(anomaly_indices)
        anom_x = [time_index[i] for i in range(len(series)) if anomaly_mask[i]]
        anom_y = [series.iloc[i] for i in range(len(series)) if anomaly_mask[i]]
        
        fig.add_trace(go.Scatter(
            x=anom_x, y=anom_y,
            name=f"이상치 ({len(anomaly_indices)}개)",
            mode="markers",
            marker=dict(color=COLORS["danger"], size=8, symbol="x",
                       line=dict(width=2, color=COLORS["danger"]))
        ))
    
    # 상/하한 밴드
    if bounds:
        if "upper_bound" in bounds:
            ub = bounds["upper_bound"]
            fig.add_trace(go.Scatter(
                x=time_index, y=ub if isinstance(ub, pd.Series) else [ub]*len(time_index),
                name="상한", mode="lines",
                line=dict(color=COLORS["danger"], width=1, dash="dash"), opacity=0.5
            ))
        if "lower_bound" in bounds:
            lb = bounds["lower_bound"]
            fig.add_trace(go.Scatter(
                x=time_index, y=lb if isinstance(lb, pd.Series) else [lb]*len(time_index),
                name="하한", mode="lines",
                line=dict(color=COLORS["danger"], width=1, dash="dash"), opacity=0.5,
                fill="tonexty", fillcolor="rgba(239,68,68,0.07)"
            ))
    
    return _apply_layout(fig, title, height=350)


def plot_residual_anomalies(
    residuals: pd.Series,
    anomaly_indices: List,
    time_index,
    threshold: float = None,
    title: str = "잔차 기반 이상탐지"
) -> go.Figure:
    """잔차 위에 이상치 마킹"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_index, y=residuals,
        name="잔차", mode="lines",
        line=dict(color=COLORS["muted"], width=1)
    ))
    
    if len(anomaly_indices) > 0:
        mask = residuals.index.isin(anomaly_indices)
        anom_x = [time_index[i] for i in range(len(residuals)) if mask[i]]
        anom_y = [residuals.iloc[i] for i in range(len(residuals)) if mask[i]]
        fig.add_trace(go.Scatter(
            x=anom_x, y=anom_y,
            name=f"이상치 ({len(anom_x)}개)",
            mode="markers",
            marker=dict(color=COLORS["danger"], size=8, symbol="x",
                       line=dict(width=2, color=COLORS["danger"]))
        ))
    
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color=COLORS["danger"], opacity=0.5,
                     annotation_text=f"+{threshold:.2f}")
        fig.add_hline(y=-threshold, line_dash="dash", line_color=COLORS["danger"], opacity=0.5,
                     annotation_text=f"-{threshold:.2f}")
    
    fig.add_hline(y=0, line_color=COLORS["muted"], line_width=0.5)
    
    return _apply_layout(fig, title, height=300)


def plot_forecast_vs_actual(
    actual: pd.Series,
    fitted: pd.Series,
    anomaly_indices: List,
    time_index,
    title: str = "예측 모형 기반 이상탐지"
) -> go.Figure:
    """실제값 vs 적합값 + 이상치"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=time_index, y=actual, name="실제값",
                            line=dict(color=COLORS["primary"], width=1.5)))
    fig.add_trace(go.Scatter(x=time_index, y=fitted, name="적합값",
                            line=dict(color=COLORS["success"], width=1.5, dash="dot")))
    
    if len(anomaly_indices) > 0:
        mask = actual.index.isin(anomaly_indices)
        anom_x = [time_index[i] for i in range(len(actual)) if mask[i]]
        anom_y = [actual.iloc[i] for i in range(len(actual)) if mask[i]]
        fig.add_trace(go.Scatter(
            x=anom_x, y=anom_y,
            name=f"이상치 ({len(anom_x)}개)",
            mode="markers",
            marker=dict(color=COLORS["danger"], size=8, symbol="x",
                       line=dict(width=2, color=COLORS["danger"]))
        ))
    
    return _apply_layout(fig, title, height=350)


# ═══════════════════════════════════════════
# Tab 4: 평가 대시보드
# ═══════════════════════════════════════════

def plot_method_comparison(method_counts: Dict[str, int]) -> go.Figure:
    """기법별 탐지 이상치 개수 비교"""
    methods = list(method_counts.keys())
    counts = list(method_counts.values())
    
    fig = go.Figure(go.Bar(
        x=methods, y=counts,
        text=counts, textposition="outside",
        marker_color=[COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                     COLORS["success"], COLORS["danger"], "#8B5CF6"][:len(methods)]
    ))
    return _apply_layout(fig, "기법별 탐지 이상치 수", height=350)


def plot_residual_histogram(residuals: pd.Series, title: str = "잔차 분포") -> go.Figure:
    """잔차 히스토그램 + 정규분포 곡선"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=40, name="잔차 분포",
        marker_color=COLORS["primary"], opacity=0.7,
        histnorm="probability density"
    ))
    
    # 정규분포 곡선 오버레이
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    from scipy.stats import norm
    mu, sigma = residuals.mean(), residuals.std()
    fig.add_trace(go.Scatter(
        x=x_range, y=norm.pdf(x_range, mu, sigma),
        name="정규분포", line=dict(color=COLORS["danger"], width=2)
    ))
    
    return _apply_layout(fig, title, height=300)


def plot_qq(residuals: pd.Series, title: str = "Q-Q Plot") -> go.Figure:
    """Q-Q 플롯"""
    from scipy.stats import probplot
    sorted_data = np.sort(residuals.dropna())
    theoretical = np.sort(np.random.normal(residuals.mean(), residuals.std(), len(sorted_data)))
    
    (osm, osr), (slope, intercept, r) = probplot(residuals.dropna(), dist="norm")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                            marker=dict(color=COLORS["primary"], size=4),
                            name="데이터"))
    
    line_x = np.array([osm.min(), osm.max()])
    line_y = slope * line_x + intercept
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines",
                            line=dict(color=COLORS["danger"], width=2, dash="dash"),
                            name="이론적 정규분포"))
    
    fig.update_layout(xaxis_title="이론적 분위수", yaxis_title="실제 분위수")
    return _apply_layout(fig, title, height=350)


def plot_cross_variable_anomalies(anomaly_df: pd.DataFrame, time_index) -> go.Figure:
    """다변량 교차 이상탐지 히트맵"""
    cols = [c for c in anomaly_df.columns if c != "anomaly_count"]
    
    z = anomaly_df[cols].astype(int).values.T
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=time_index,
        y=cols,
        colorscale=[[0, "#F3F4F6"], [1, COLORS["danger"]]],
        showscale=False,
        hoverongaps=False
    ))
    
    fig.update_layout(**LAYOUT_DEFAULTS, height=250, title="다변량 교차 이상탐지 (빨간색 = 이상)")
    return fig


def plot_anomaly_count_timeline(anomaly_df: pd.DataFrame, time_index) -> go.Figure:
    """시점별 이상 변수 개수 타임라인"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=time_index,
        y=anomaly_df["anomaly_count"],
        marker_color=[COLORS["danger"] if c >= 2 else COLORS["accent"] if c == 1 else COLORS["grid"]
                     for c in anomaly_df["anomaly_count"]],
        name="이상 변수 수"
    ))

    return _apply_layout(fig, "시점별 이상 탐지 변수 수", height=250)


# ═══════════════════════════════════════════
# Scorer / Detector / 평가 시각화
# ═══════════════════════════════════════════

def plot_series_with_score(series, score, anomaly_idx, time_index,
                           threshold=None, fitted=None, title="") -> go.Figure:
    """원본 시계열 + 이상점수 + 탐지결과를 3단으로 (show_anomalies_from_scores 개념)."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.42, 0.32, 0.26],
        subplot_titles=["시계열 (이상치 표시)", "이상 점수 (anomaly score)", "탐지 결과 (이진)"]
    )

    fig.add_trace(go.Scatter(x=time_index, y=series, name="시계열",
                             line=dict(color=COLORS["primary"], width=1.3)), row=1, col=1)
    if fitted is not None:
        fig.add_trace(go.Scatter(x=time_index, y=fitted, name="예측",
                                 line=dict(color=COLORS["success"], width=1, dash="dot"),
                                 opacity=0.7), row=1, col=1)
    if len(anomaly_idx) > 0:
        ax = [time_index[i] for i in anomaly_idx]
        ay = [series.iloc[i] for i in anomaly_idx]
        fig.add_trace(go.Scatter(x=ax, y=ay, name=f"이상치 ({len(anomaly_idx)})",
                                 mode="markers",
                                 marker=dict(color=COLORS["danger"], size=8, symbol="x",
                                             line=dict(width=2, color=COLORS["danger"]))),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=time_index, y=score, name="점수",
                             line=dict(color=COLORS["secondary"], width=1.2),
                             showlegend=False), row=2, col=1)
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color=COLORS["danger"],
                      opacity=0.6, row=2, col=1,
                      annotation_text=f"임계값 {threshold:.3g}")

    binary = [1 if i in set(anomaly_idx) else 0 for i in range(len(series))]
    fig.add_trace(go.Scatter(x=time_index, y=binary, name="탐지",
                             line=dict(color=COLORS["danger"], width=1), fill="tozeroy",
                             fillcolor="rgba(239,68,68,0.2)", showlegend=False), row=3, col=1)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["정상", "이상"], row=3, col=1)

    fig.update_layout(**LAYOUT_DEFAULTS, height=620, title=title)
    return fig


def plot_roc_curve(fpr, tpr, auc) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})",
                             line=dict(color=COLORS["primary"], width=2.5), fill="tozeroy",
                             fillcolor="rgba(37,99,235,0.1)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="무작위",
                             line=dict(color=COLORS["muted"], width=1, dash="dash")))
    fig.update_layout(xaxis_title="FPR (위양성률)", yaxis_title="TPR (재현율)")
    return _apply_layout(fig, "ROC 곡선", height=380)


def plot_pr_curve(recall, precision, ap) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (AP={ap:.3f})",
                             line=dict(color=COLORS["secondary"], width=2.5), fill="tozeroy",
                             fillcolor="rgba(124,58,237,0.1)"))
    fig.update_layout(xaxis_title="Recall (재현율)", yaxis_title="Precision (정밀도)")
    return _apply_layout(fig, "Precision-Recall 곡선", height=380)


def plot_confusion(cm) -> go.Figure:
    z = cm[::-1]  # 위쪽이 실제 이상이 되도록 뒤집기
    fig = go.Figure(data=go.Heatmap(
        z=z, x=["정상(예측)", "이상(예측)"], y=["이상(실제)", "정상(실제)"],
        colorscale="Blues", text=z, texttemplate="%{text}",
        textfont=dict(size=18), showscale=False))
    return _apply_layout(fig, "혼동행렬", height=340)


def plot_score_histogram(score, threshold=None, title="이상 점수 분포") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=score.dropna(), nbinsx=50,
                               marker_color=COLORS["primary"], opacity=0.75))
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color=COLORS["danger"],
                      annotation_text="임계값")
    fig.update_layout(xaxis_title="이상 점수", yaxis_title="빈도")
    return _apply_layout(fig, title, height=320)


def plot_threshold_sweep(sweep_df) -> go.Figure:
    """임계값(분위수)에 따른 탐지 개수 민감도."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep_df["quantile"], y=sweep_df["n_anomalies"],
                             mode="lines+markers", line=dict(color=COLORS["accent"], width=2),
                             marker=dict(size=5)))
    fig.update_layout(xaxis_title="임계값 분위수 (quantile)", yaxis_title="탐지된 이상치 수")
    return _apply_layout(fig, "임계값 민감도 곡선", height=340)


def plot_scorer_agreement(score_dict, time_index) -> go.Figure:
    """여러 Scorer의 정규화된 점수를 겹쳐 비교."""
    fig = go.Figure()
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"]]
    for i, (name, s) in enumerate(score_dict.items()):
        s = s.astype(float)
        rng = s.max() - s.min()
        norm = (s - s.min()) / rng if rng else s * 0
        fig.add_trace(go.Scatter(x=time_index, y=norm, name=name,
                                 line=dict(color=palette[i % len(palette)], width=1.3)))
    return _apply_layout(fig, "Scorer 간 이상 점수 비교 (0~1 정규화)", height=360)
