"""
시계열 데이터 전처리 모듈
- 결측치 처리 (LOCF, NOCB, 이동평균, 선형보간)
- 디노이징 (SMA, EMA)
※ 이상치는 탐지 대상이므로 전처리에서 제거하지 않음
"""
import pandas as pd
import numpy as np


def handle_missing_values(df: pd.DataFrame, method: str = "linear", window: int = 6) -> pd.DataFrame:
    """
    결측치 처리
    - linear: 선형보간
    - ffill: LOCF (Last Observation Carried Forward)
    - bfill: NOCB (Next Observation Carried Backward)
    - moving_average: 이동평균
    """
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if method == "linear":
        df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method="linear", limit_direction="both")
    elif method == "ffill":
        df_filled[numeric_cols] = df_filled[numeric_cols].ffill()
        df_filled[numeric_cols] = df_filled[numeric_cols].bfill()  # 맨 앞 결측 처리
    elif method == "bfill":
        df_filled[numeric_cols] = df_filled[numeric_cols].bfill()
        df_filled[numeric_cols] = df_filled[numeric_cols].ffill()  # 맨 뒤 결측 처리
    elif method == "moving_average":
        for col in numeric_cols:
            ma = df_filled[col].rolling(window=window, min_periods=1).mean()
            df_filled[col] = df_filled[col].fillna(ma)
            df_filled[col] = df_filled[col].interpolate(method="linear", limit_direction="both")
    
    return df_filled


def denoise_sma(series: pd.Series, window: int = 3) -> pd.Series:
    """단순이동평균(SMA) 디노이징"""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def denoise_ema(series: pd.Series, alpha: float = 0.3) -> pd.Series:
    """지수이동평균(EMA) 디노이징"""
    return series.ewm(alpha=alpha, adjust=False).mean()


def apply_denoising(df: pd.DataFrame, method: str = "none", **kwargs) -> pd.DataFrame:
    """데이터프레임 전체에 디노이징 적용"""
    if method == "none":
        return df.copy()
    
    df_denoised = df.copy()
    numeric_cols = df_denoised.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == "sma":
            df_denoised[col] = denoise_sma(df_denoised[col], window=kwargs.get("window", 3))
        elif method == "ema":
            df_denoised[col] = denoise_ema(df_denoised[col], alpha=kwargs.get("alpha", 0.3))
    
    return df_denoised


def get_missing_info(df: pd.DataFrame) -> pd.DataFrame:
    """결측치 정보 요약"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    info = pd.DataFrame({
        "컬럼": numeric_cols,
        "결측치 수": [df[col].isna().sum() for col in numeric_cols],
        "결측률(%)": [round(df[col].isna().mean() * 100, 2) for col in numeric_cols]
    })
    return info
