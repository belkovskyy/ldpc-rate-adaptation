"""Признаки для прогноза QBER: лаги и скользящие статистики по временному ряду.

Кадры квантового канала упорядочены во времени (block_id, frame_idx). QBER
меняется плавно, поэтому лаги и скользящие среднее/отклонение предыдущих кадров —
сильные признаки для прогноза текущего уровня ошибок.
"""
from __future__ import annotations

from typing import List

import pandas as pd

from policy import ALLOWED_R


def add_lags(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    """Значения col на L кадров назад (сдвиг по времени)."""
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolls(df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
    """Скользящие среднее и отклонение по окну предыдущих кадров."""
    for w in windows:
        df[f"{col}_roll{w}_mean"] = df[col].rolling(w, min_periods=1).mean()
        df[f"{col}_roll{w}_std"] = df[col].rolling(w, min_periods=1).std().fillna(0.0)
    return df


def nearest_R(x: float) -> float:
    """Ближайшая допустимая скорость кода из сетки ALLOWED_R."""
    return min(ALLOWED_R, key=lambda r: abs(r - x))
