"""
다변량 시계열 이상탐지 웹앱

탭 구성
  ① 데이터 개요        : 업로드/샘플, 시각화, 통계, 상관, 결측
  ② 전처리 & 특성분석   : 결측/디노이징, ADF·Ljung-Box, ACF/PACF, STL
  ③ 이상탐지           : Scorer → Detector → Aggregator
  ④ 평가 대시보드       : AUC-ROC / 임계값 민감도 / Scorer 일치도
  ⑤ 결과 & 내보내기     : 이상치 상세표 + CSV 다운로드
"""
import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

from utils.preprocessing import handle_missing_values, apply_denoising, get_missing_info
from utils.analysis import compute_acf_pacf, stl_decompose, run_all_tests
from utils.scorers import (
    compute_score, threshold_from_scores, detect, aggregate,
    evaluate_scores, classification_metrics, threshold_sweep, SCORERS,
)
from utils.visualization import (
    plot_time_series, plot_correlation_heatmap, plot_missing_values,
    plot_before_after, plot_stl_decomposition, plot_acf_pacf,
    plot_cross_variable_anomalies, plot_anomaly_count_timeline,
    plot_series_with_score, plot_roc_curve, plot_pr_curve, plot_confusion,
    plot_score_histogram, plot_threshold_sweep, plot_scorer_agreement,
)

# ════════════════════════════════════════════
# 페이지 설정
# ════════════════════════════════════════════
st.set_page_config(page_title="시계열 이상탐지 웹앱", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; font-weight: 600; }
    div[data-testid="stMetric"] { background-color: #F8FAFC; padding: 12px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 다변량 시계열 이상탐지 웹앱")
st.caption("CSV를 업로드하면 자동으로 분석 후 Scorer→Detector→Aggregator 파이프라인으로 이상을 탐지합니다.")


# ════════════════════════════════════════════
# 데이터 로드
# ════════════════════════════════════════════
@st.cache_data
def load_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data
def generate_sample(choice: str) -> pd.DataFrame:
    np.random.seed(42)
    if choice == "공장 센서 데이터":
        n = 500
        dates = pd.date_range("2023-01-01", periods=n, freq="h")
        t = np.arange(n)
        temp = 25 + 0.005*t + 5*np.sin(2*np.pi*t/24) + np.random.normal(0, 1, n)
        vibration = 0.5 + 0.2*np.sin(2*np.pi*t/24) + np.random.normal(0, 0.1, n)
        pressure = 100 + 0.003*t + np.random.normal(0, 2, n)
        anom = [50, 51, 120, 200, 201, 202, 310, 400, 401]
        temp[anom] += np.random.choice([15, -15], size=len(anom))
        vibration[[50, 120, 200, 310, 400]] += np.random.uniform(1.5, 3.0, size=5)
        pressure[[51, 201, 202, 401]] += np.random.choice([30, -30], size=4)
        label = np.zeros(n, dtype=int); label[anom] = 1
        temp[75] = temp[76] = np.nan; vibration[150] = np.nan; pressure[300] = np.nan
        return pd.DataFrame({"timestamp": dates, "temperature": np.round(temp, 2),
                             "vibration": np.round(vibration, 4), "pressure": np.round(pressure, 2),
                             "is_anomaly": label})
    else:
        n = 720
        dates = pd.date_range("2023-06-01", periods=n, freq="h")
        t = np.arange(n)
        daily = 20*np.sin(2*np.pi*t/24 - np.pi/2); weekly = 5*np.sin(2*np.pi*t/168)
        cpu = np.clip(45 + daily + weekly + np.random.normal(0, 5, n), 5, 100)
        memory = np.clip(60 + 0.01*t + 10*np.sin(2*np.pi*t/24) + np.random.normal(0, 3, n), 10, 100)
        network = np.clip(500 + 200*np.sin(2*np.pi*t/24) + np.random.normal(0, 50, n), 0, None)
        spike = [100, 101, 250, 251, 252, 450, 600, 601]
        cpu[spike] = np.random.uniform(90, 100, len(spike))
        memory[[100, 250, 251, 450, 600]] += 25
        network[[101, 252, 601]] *= 3
        label = np.zeros(n, dtype=int); label[spike] = 1
        cpu[180] = np.nan; memory[360] = np.nan
        return pd.DataFrame({"timestamp": dates, "cpu_usage": np.round(cpu, 2),
                             "memory_usage": np.round(memory, 2), "network_traffic": np.round(network, 2),
                             "is_anomaly": label})


# ════════════════════════════════════════════
# 사이드바 — 데이터 선택
# ════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 설정")
    uploaded = st.file_uploader("📂 CSV 파일 업로드", type=["csv"])
    use_sample = st.checkbox("샘플 데이터 사용", value=uploaded is None)
    sample_choice = None
    if use_sample:
        sample_choice = st.selectbox("샘플 데이터 선택", ["공장 센서 데이터", "서버 모니터링 데이터"])

# 파일 변경 감지용 키 (파일이 바뀌면 다운스트림 캐시/결과가 새로 계산됨)
if uploaded is not None:
    file_bytes = uploaded.getvalue()
    data_key = hashlib.md5(file_bytes).hexdigest()
    df = load_data(file_bytes)
elif use_sample:
    data_key = f"sample::{sample_choice}"
    df = generate_sample(sample_choice)
else:
    df = None
    data_key = None

if df is None:
    st.info("👈 사이드바에서 CSV를 업로드하거나 샘플 데이터를 선택하세요.")
    st.stop()

# 새 데이터가 들어오면 이전 탐지 결과 초기화
if st.session_state.get("data_key") != data_key:
    st.session_state["data_key"] = data_key
    st.session_state.pop("detection", None)


# ════════════════════════════════════════════
# 사이드바 — 컬럼 자동 감지 & 파라미터
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown("---")

    time_candidates = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col]); time_candidates.append(col)
        except (ValueError, TypeError):
            pass
    time_col = st.selectbox("⏰ 시간 컬럼", time_candidates or df.columns.tolist())
    try:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
    except Exception:
        st.warning("시간 컬럼 변환 실패 — 인덱스를 시간축으로 사용합니다.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 라벨(정답) 컬럼 자동 후보: 0/1 만 가진 컬럼
    label_candidates = [c for c in numeric_cols
                        if set(pd.Series(df[c]).dropna().unique()).issubset({0, 1})]
    label_col = st.selectbox("🏷️ 정답(이상) 라벨 컬럼 (선택)",
                             ["(없음)"] + label_candidates,
                             help="0/1 라벨이 있으면 AUC-ROC 등 정량 평가가 가능합니다.")
    label_col = None if label_col == "(없음)" else label_col

    feature_cols = [c for c in numeric_cols if c != label_col]
    if not feature_cols:
        st.error("분석할 수치형 변수가 없습니다."); st.stop()

    selected_cols = st.multiselect("📊 분석 대상 변수", feature_cols,
                                   default=feature_cols[:min(3, len(feature_cols))])
    if not selected_cols:
        st.warning("최소 1개 변수를 선택하세요."); st.stop()

    st.markdown("---")
    st.subheader("🔧 공통 파라미터")
    seasonal_period = st.number_input("계절 주기 (sp)", 2, 365, 24, 1,
                                      help="시간별=24, 월별=12 등")

time_idx = df[time_col].values
labels = df[label_col] if label_col else None


# ════════════════════════════════════════════
# 탭
# ════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 데이터 개요", "🔧 전처리 & 분석", "🎯 이상탐지", "📊 평가 대시보드", "📁 결과 & 내보내기"
])


# ─── Tab 1: 데이터 개요 ───────────────────────
with tab1:
    st.header("📋 데이터 개요")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행 수", f"{len(df):,}")
    c2.metric("분석 변수", f"{len(selected_cols)}")
    c3.metric("결측치", f"{df[selected_cols].isna().sum().sum()}")
    c4.metric("정답 라벨", "있음" if label_col else "없음")

    st.subheader("시계열 시각화")
    st.plotly_chart(plot_time_series(df, time_col, selected_cols), use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("기초 통계량")
        st.dataframe(df[selected_cols].describe().round(3), use_container_width=True)
    with cc2:
        if len(selected_cols) >= 2:
            st.subheader("상관관계")
            st.plotly_chart(plot_correlation_heatmap(df, selected_cols), use_container_width=True)

    missing_info = get_missing_info(df[selected_cols])
    if missing_info["결측치 수"].sum() > 0:
        st.subheader("결측치 현황")
        st.plotly_chart(plot_missing_values(missing_info), use_container_width=True)

    with st.expander("📄 데이터 프리뷰"):
        st.dataframe(df.head(50), use_container_width=True)


# ─── Tab 2: 전처리 & 분석 ─────────────────────
with tab2:
    st.header("🔧 전처리 & 시계열 분석")
    st.info("💡 이상치는 탐지 대상이므로 제거하지 않고, 결측치·노이즈만 처리합니다.")

    p1, p2 = st.columns(2)
    with p1:
        missing_method = st.selectbox("결측치 처리", ["linear", "ffill", "bfill", "moving_average"],
                                      format_func=lambda x: {"linear": "선형보간", "ffill": "LOCF(직전값)",
                                                             "bfill": "NOCB(직후값)", "moving_average": "이동평균"}[x])
    with p2:
        denoise_method = st.selectbox("디노이징", ["none", "sma", "ema"],
                                      format_func=lambda x: {"none": "적용 안 함", "sma": "단순이동평균(SMA)",
                                                             "ema": "지수이동평균(EMA)"}[x])
    dparams = {}
    if denoise_method == "sma":
        dparams["window"] = st.slider("SMA 윈도우", 2, 20, 5)
    elif denoise_method == "ema":
        dparams["alpha"] = st.slider("EMA 평활계수 α", 0.1, 0.9, 0.3, 0.1)

    df_proc = handle_missing_values(df, method=missing_method)
    df_proc = apply_denoising(df_proc, method=denoise_method, **dparams)
    st.session_state["df_processed"] = df_proc

    if denoise_method != "none" or df[selected_cols].isna().sum().sum() > 0:
        cc = st.selectbox("전후 비교 변수", selected_cols, key="cmp")
        st.plotly_chart(plot_before_after(df[cc], df_proc[cc], time_idx,
                                          title=f"{cc} — 전처리 전후"), use_container_width=True)

    st.markdown("---")
    st.subheader("시계열 특성 분석")
    st.markdown("#### 정상성 · 백색잡음 검정")
    st.dataframe(run_all_tests(df_proc, selected_cols), use_container_width=True, hide_index=True)

    st.markdown("#### ACF / PACF")
    acol = st.selectbox("변수", selected_cols, key="acf")
    try:
        a, p, conf = compute_acf_pacf(df_proc[acol], nlags=min(40, len(df_proc)//4))
        st.plotly_chart(plot_acf_pacf(a, p, conf), use_container_width=True)
    except Exception as e:
        st.warning(f"ACF/PACF 오류: {e}")

    st.markdown("#### STL 분해")
    scol = st.selectbox("변수", selected_cols, key="stl")
    stl_r = stl_decompose(df_proc[scol], period=seasonal_period)
    if stl_r:
        st.plotly_chart(plot_stl_decomposition(stl_r, time_idx), use_container_width=True)
    else:
        st.warning(f"STL 분해 실패: 최소 {2*seasonal_period+1}개 데이터 필요")


# ─── Tab 3: 이상탐지 ──────────────────────────
with tab3:
    st.header("🎯 이상탐지 — Scorer → Detector → Aggregator")
    df_proc = st.session_state.get("df_processed", df)

    with st.expander("ℹ️ 이상탐지 파이프라인 설명", expanded=False):
        st.markdown(
            "- **Scorer**: 시계열에 이상 점수 부여 — NormScorer(오차 크기) / KMeansScorer(오차 패턴) / WassersteinScorer(분포 변화)\n"
            "- **AnomalyModel(예측 기반)**: 예측모델 잔차에 Scorer 적용\n"
            "- **Detector**: 점수 → 이진 (분위수/시그마 임계값)\n"
            "- **Aggregator**: 다변량 이진 결과를 OR/AND/count 로 통합")

    cset1, cset2, cset3 = st.columns(3)
    with cset1:
        chosen_scorers = st.multiselect("Scorer 선택", list(SCORERS.keys()),
                                        default=["NormScorer", "KMeansScorer"])
        forecasting = st.checkbox("예측모델 기반(AnomalyModel)", value=True,
                                  help="끄면 Scorer를 시계열에 직접 적용")
    with cset2:
        det_method = st.radio("Detector 임계값", ["quantile", "sigma"],
                              format_func=lambda x: {"quantile": "분위수", "sigma": "σ(시그마)"}[x], horizontal=True)
        det_q = st.slider("분위수 q", 0.80, 0.999, 0.95, 0.005) if det_method == "quantile" \
            else st.slider("σ 배수", 1.5, 5.0, 3.0, 0.5)
    with cset3:
        model_type = st.selectbox("예측모델", ["ridge", "rf"],
                                  format_func=lambda x: {"ridge": "Ridge 회귀", "rf": "랜덤포레스트"}[x])
        window = st.slider("윈도우 크기 (KMeans/Wasserstein)", 5, 60, 20)

    if not chosen_scorers:
        st.warning("최소 1개 Scorer를 선택하세요."); st.stop()

    if st.button("🚀 이상탐지 실행", type="primary"):
        detection = {"per_var": {}, "scorers": chosen_scorers,
                     "det_method": det_method, "det_q": det_q}
        prog = st.progress(0.0)
        total = len(selected_cols) * len(chosen_scorers)
        done = 0
        for col in selected_cols:
            series = df_proc[col].reset_index(drop=True)
            detection["per_var"][col] = {}
            for sc in chosen_scorers:
                res = compute_score(series, sc, time_index=time_idx, forecasting=forecasting,
                                    lags=seasonal_period, model_type=model_type, window=window)
                done += 1; prog.progress(done / total)
                if res.get("error"):
                    detection["per_var"][col][sc] = {"error": res["error"]}
                    continue
                score = res["score"]
                if det_method == "quantile":
                    thr = threshold_from_scores(score, "quantile", q=det_q)
                else:
                    thr = threshold_from_scores(score, "sigma", n_sigma=det_q)
                binary = detect(score, thr)
                anom_idx = [i for i in range(len(binary)) if binary.iloc[i] == 1]
                detection["per_var"][col][sc] = {
                    "score": score, "threshold": thr, "binary": binary,
                    "anomaly_idx": anom_idx, "fitted": res.get("fitted"),
                }
        prog.empty()
        st.session_state["detection"] = detection
        st.success("✅ 탐지 완료 — 아래 결과 및 평가 대시보드를 확인하세요.")

    detection = st.session_state.get("detection")
    if not detection:
        st.info("위 설정을 고른 뒤 **이상탐지 실행** 버튼을 누르세요.")
        st.stop()

    # 결과 시각화
    for col in selected_cols:
        if col not in detection["per_var"]:
            continue
        st.subheader(f"📌 {col}")
        for sc, r in detection["per_var"][col].items():
            if r.get("error"):
                st.warning(f"{sc}: {r['error']}"); continue
            series = df_proc[col].reset_index(drop=True)
            st.plotly_chart(
                plot_series_with_score(series, r["score"], r["anomaly_idx"], time_idx,
                                       threshold=r["threshold"], fitted=r.get("fitted"),
                                       title=f"{sc} — {col} ({len(r['anomaly_idx'])}개 탐지)"),
                use_container_width=True)
        st.markdown("---")

    # Aggregator (다변량 통합)
    st.subheader("🔗 Aggregator — 변수 간 이상 통합")
    agg_method = st.radio("통합 방식", ["or", "count", "and"],
                          format_func=lambda x: {"or": "OR(하나라도)", "count": "N개 이상",
                                                 "and": "AND(모두)"}[x], horizontal=True)
    min_count = st.slider("count 기준 N", 2, max(2, len(selected_cols)), 2) if agg_method == "count" else 2

    # 변수별: 모든 Scorer 합집합 → 이진
    var_binary = {}
    for col in selected_cols:
        merged = np.zeros(len(df_proc), dtype=int)
        for sc, r in detection["per_var"].get(col, {}).items():
            if not r.get("error"):
                merged |= r["binary"].to_numpy()
        var_binary[col] = merged
    binary_df = pd.DataFrame(var_binary)
    agg_series = aggregate(binary_df, method=agg_method, min_count=min_count)

    cross_df = binary_df.copy().astype(bool)
    cross_df["anomaly_count"] = binary_df.sum(axis=1)
    st.plotly_chart(plot_cross_variable_anomalies(cross_df, time_idx), use_container_width=True)
    st.plotly_chart(plot_anomaly_count_timeline(cross_df, time_idx), use_container_width=True)
    st.success(f"통합 결과: 총 {int(agg_series.sum())}개 시점이 이상으로 판정되었습니다.")

    st.session_state["agg"] = {"binary_df": binary_df, "agg_series": agg_series,
                               "agg_method": agg_method}


# ─── Tab 4: 평가 대시보드 ─────────────────────
with tab4:
    st.header("📊 평가 대시보드")
    detection = st.session_state.get("detection")
    if not detection:
        st.info("🎯 이상탐지 탭에서 먼저 분석을 실행하세요."); st.stop()

    eval_col = st.selectbox("평가할 변수", selected_cols, key="eval")
    var_res = detection["per_var"].get(eval_col, {})
    valid = {sc: r for sc, r in var_res.items() if not r.get("error")}
    if not valid:
        st.warning("이 변수에 유효한 탐지 결과가 없습니다."); st.stop()

    if labels is not None:
        # ── 라벨 있음: 정량 평가 (AUC-ROC 등) ──
        st.subheader("✅ 정량 평가 (정답 라벨 기반)")
        st.caption("점수 기반 정량 평가 — AUC-ROC / AUC-PR / F1")

        rows = []
        for sc, r in valid.items():
            ev = evaluate_scores(r["score"], labels)
            cm = classification_metrics(r["binary"], labels)
            rows.append({"Scorer": sc,
                         "AUC-ROC": round(ev.get("auc_roc", float("nan")), 4),
                         "AUC-PR": round(ev.get("auc_pr", float("nan")), 4),
                         "Precision": round(cm["precision"], 4),
                         "Recall": round(cm["recall"], 4),
                         "F1": round(cm["f1"], 4),
                         "탐지수": len(r["anomaly_idx"])})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        best = st.selectbox("곡선/혼동행렬을 볼 Scorer", list(valid.keys()), key="best")
        r = valid[best]
        ev = evaluate_scores(r["score"], labels)
        cm = classification_metrics(r["binary"], labels)
        if ev.get("error"):
            st.warning(ev["error"])
        else:
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(plot_roc_curve(*ev["roc"], ev["auc_roc"]), use_container_width=True)
            with g2:
                st.plotly_chart(plot_pr_curve(*ev["pr"], ev["auc_pr"]), use_container_width=True)
        st.plotly_chart(plot_confusion(cm["confusion"]), use_container_width=True)
    else:
        # ── 라벨 없음: 비지도 진단 ──
        st.subheader("🔎 비지도 진단 (정답 라벨 없음)")
        st.caption("정답이 없을 때 탐지가 적절한지 판단하기 위한 보조 지표들입니다.")

        diag_sc = st.selectbox("Scorer", list(valid.keys()), key="diag")
        r = valid[diag_sc]
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(plot_score_histogram(r["score"], r["threshold"]), use_container_width=True)
        with d2:
            st.plotly_chart(plot_threshold_sweep(threshold_sweep(r["score"])), use_container_width=True)

    st.markdown("---")
    st.subheader("Scorer 간 일치도")
    st.caption("여러 Scorer가 같은 시점을 이상으로 보는지 비교 — 일치할수록 신뢰도가 높습니다.")
    score_dict = {sc: r["score"] for sc, r in valid.items()}
    st.plotly_chart(plot_scorer_agreement(score_dict, time_idx), use_container_width=True)
    if len(valid) >= 2:
        sets = [set(r["anomaly_idx"]) for r in valid.values()]
        inter = set.intersection(*sets)
        union = set.union(*sets)
        jac = len(inter) / max(len(union), 1)
        m1, m2, m3 = st.columns(3)
        m1.metric("모든 Scorer 공통 탐지", f"{len(inter)}개")
        m2.metric("합집합 탐지", f"{len(union)}개")
        m3.metric("일치도 (Jaccard)", f"{jac:.2%}")


# ─── Tab 5: 결과 & 내보내기 ───────────────────
with tab5:
    st.header("📁 결과 & 내보내기")
    detection = st.session_state.get("detection")
    if not detection:
        st.info("🎯 이상탐지 탭에서 먼저 분석을 실행하세요."); st.stop()

    rows = []
    for col in selected_cols:
        for sc, r in detection["per_var"].get(col, {}).items():
            if r.get("error"):
                continue
            for i in r["anomaly_idx"]:
                rows.append({"시점": df[time_col].iloc[i] if i < len(df) else i,
                             "인덱스": i, "변수": col,
                             "값": round(float(df_proc[col].iloc[i]), 4),
                             "Scorer": sc, "이상점수": round(float(r["score"].iloc[i]), 4)})
    if not rows:
        st.info("탐지된 이상치가 없습니다."); st.stop()

    detail = pd.DataFrame(rows)
    # 시점·변수별로 어느 Scorer가 잡았는지 집계
    pivot = (detail.groupby(["인덱스", "시점", "변수"])
             .agg(값=("값", "first"),
                  탐지_Scorer=("Scorer", lambda x: ", ".join(sorted(set(x)))),
                  Scorer수=("Scorer", lambda x: len(set(x))))
             .reset_index().sort_values(["Scorer수", "인덱스"], ascending=[False, True]))

    if labels is not None:
        pivot["정답라벨"] = pivot["인덱스"].map(lambda i: int(labels.iloc[i]) if i < len(labels) else None)

    c1, c2, c3 = st.columns(3)
    c1.metric("이상 시점·변수 건수", f"{len(pivot)}")
    c2.metric("이상 시점 수(중복제거)", f"{pivot['인덱스'].nunique()}")
    c3.metric("2개+ Scorer 합의", f"{int((pivot['Scorer수'] >= 2).sum())}")

    st.dataframe(pivot, use_container_width=True, hide_index=True)

    buf = io.StringIO()
    pivot.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button("📥 이상치 목록 CSV 다운로드", buf.getvalue(),
                       file_name="anomalies.csv", mime="text/csv")
