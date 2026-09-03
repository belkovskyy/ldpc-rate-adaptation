"""Политика выбора параметров LDPC-кода {R, s, p} по оценке QBER.

Информационная реконсиляция в квантовом распределении ключей (QKD): по текущему
уровню ошибок канала (QBER) нужно выбрать скорость LDPC-кода R и разбиение
служебных бит на shortening/puncturing (s, p, s+p=d) так, чтобы коррекция прошла,
а раскрытой информации было как можно меньше — то есть R максимально возможным.
"""
from __future__ import annotations

import math

import numpy as np

# Константы кодирования (из условия задачи).
ALLOWED_R = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
N = 32000          # длина блока
D = 4800           # s + p (служебные биты)
F_EC = 1.15        # коэффициент эффективности реконсиляции
EMA_ALPHA = 0.33   # сглаживание оценки QBER


def h2(p: float) -> float:
    """Бинарная энтропия с клэмпами для численной устойчивости."""
    x = float(np.clip(p, 1e-6, 0.5 - 1e-6))
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def choose_rsp(qber: float, f_ec: float = F_EC, n: int = N, d: int = D):
    """(R, s, p) с максимально возможным R при данном QBER и f_ec.

    Целевая эффективная скорость r_cand = 1 - f_ec * h2(qber); идём по ALLOWED_R
    сверху вниз и берём наибольший R, при котором puncturing/shortening укладывается
    в [0, d]. Fail-safe — минимальный R с клэмпом.
    """
    r_cand = 1.0 - f_ec * h2(qber)
    for R in sorted(ALLOWED_R, reverse=True):
        p = math.ceil((1.0 - R) * n - (1.0 - r_cand) * (n - d))
        s = d - p
        if 0 <= p <= d and 0 <= s <= d:
            return round(R, 2), int(s), int(p)

    R = min(ALLOWED_R)
    p = int(np.clip(math.ceil((1.0 - R) * n - (1.0 - r_cand) * (n - d)), 0, d))
    return round(R, 2), int(d - p), int(p)


def ema_series(values, alpha: float = EMA_ALPHA) -> np.ndarray:
    """Онлайновая EMA: на шаге t сглаживание по значениям QBER до t включительно."""
    out, prev = [], None
    for v in values:
        prev = float(v) if prev is None else alpha * float(v) + (1.0 - alpha) * prev
        out.append(prev)
    return np.array(out, dtype=float)
