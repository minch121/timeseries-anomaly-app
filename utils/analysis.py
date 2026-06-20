"""
시계열 특성 분석 모듈
- 정상성 검정 (ADF, Ljung-Box)
- ACF/PACF 계산
- STL 분해
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from typing import Dict, Tuple, Optional


def adf_test(series: pd.Series) -> Dict:
    """
    ADF 검정 (Augmented Dickey-Fuller)
    H₀: 단위근 존재 → 비정상성
    H₁: 단위근 미존재 → 정상성
    """
    clean = series.dropna()
    if len(clean) < 20:
        return {"statistic": None, "p_value": None, "result": "데이터 부족", "is_stationary": None}
    
    try:
        result = adfuller(clean, autolag="AIC")
        p_value = result[1]
        return {
            "statistic": round(result[0], 4),
            "p_value": round(p_value, 4),
            "result": "정상성 시계열" if p_value < 0.05 else "비정상성 시계열",
            "is_stationary": p_value < 0.05
        }
    except Exception as e:
        return {"statistic": None, "p_value": None, "result": f"오류: {str(e)}", "is_stationary": None}


def ljungbox_test(series: pd.Series, lags: int = 1) -> Dict:
    """
    Ljung-Box 검정
    H₀: 자기상관 없음 (백색잡음)
    H₁: 자기상관 있음
    """
    clean = series.dropna()
    if len(clean) < 20:
        return {"statistic": None, "p_value": None, "result": "데이터 부족", "is_white_noise": None}
    
    try:
        result = acorr_ljungbox(clean, lags=[lags], return_df=True)
        p_value = result["lb_pvalue"].iloc[0]
        return {
            "statistic": round(result["lb_stat"].iloc[0], 4),
            "p_value": round(p_value, 4),
            "result": "백색잡음 아님 (모형 개선 가능)" if p_value < 0.05 else "백색잡음 (추가 모형 불필요)",
            "is_white_noise": p_value >= 0.05
        }
    except Exception as e:
        return {"statistic": None, "p_value": None, "result": f"오류: {str(e)}", "is_white_noise": None}


def compute_acf_pacf(series: pd.Series, nlags: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ACF/PACF 계산"""
    clean = series.dropna()
    nlags = min(nlags, len(clean) // 2 - 1)
    
    acf_values = acf(clean, nlags=nlags, fft=True)
    pacf_values = pacf(clean, nlags=nlags)
    conf_interval = 1.96 / np.sqrt(len(clean))
    
    return acf_values, pacf_values, conf_interval


def stl_decompose(series: pd.Series, period: int = 12) -> Optional[Dict]:
    """
    STL 분해: 추세 + 계절성 + 잔차
    """
    clean = series.dropna()
    if len(clean) < 2 * period + 1:
        return None
    
    try:
        stl = STL(clean, period=period, robust=True)
        result = stl.fit()
        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
            "observed": clean
        }
    except Exception as e:
        return None


def run_all_tests(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """선택된 모든 컬럼에 대해 ADF + Ljung-Box 검정 수행"""
    results = []
    for col in columns:
        adf = adf_test(df[col])
        lb = ljungbox_test(df[col])
        results.append({
            "변수": col,
            "ADF 통계량": adf["statistic"],
            "ADF p-value": adf["p_value"],
            "ADF 판정": adf["result"],
            "LB 통계량": lb["statistic"],
            "LB p-value": lb["p_value"],
            "LB 판정": lb["result"]
        })
    return pd.DataFrame(results)
