"""Прогноз QBER квантильной регрессией (LightGBM + Optuna).

Оцениваем не только медиану QBER (alpha=0.5), но и верхний квантиль (alpha=0.8):
верхняя оценка задаёт запас для fail-safe в политике выбора кода — лучше взять
скорость чуть консервативнее, чем не пройти коррекцию.

Валидация — TimeSeriesSplit с зазором (gap).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


def _cv_splits(x, n_splits: int = 5, test_size: int = 400, gap: int = 2):
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    yield from tscv.split(x)


def tune_lgbm_quantile(x: pd.DataFrame, y: pd.Series, alpha: float = 0.8,
                       n_trials: int = 60) -> Dict:
    """Подобрать гиперпараметры LightGBM-квантили по MAE на TimeSeriesSplit."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "quantile",
            "alpha": alpha,
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1800),
            "random_state": 42,
            "verbosity": -1,
        }
        maes = []
        for tr, va in _cv_splits(x):
            model = LGBMRegressor(**params)
            model.fit(x.iloc[tr], y.iloc[tr])
            maes.append(mean_absolute_error(y.iloc[va], model.predict(x.iloc[va])))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best.update({"objective": "quantile", "alpha": alpha, "random_state": 42, "verbosity": -1})
    return best


def fit_quantile_heads(x: pd.DataFrame, y: pd.Series,
                       alpha_hi: float = 0.8) -> Tuple[LGBMRegressor, LGBMRegressor]:
    """Обучить две головы: медиану (0.5) и верхний квантиль (alpha_hi)."""
    q50 = LGBMRegressor(**tune_lgbm_quantile(x, y, alpha=0.5, n_trials=40)).fit(x, y)
    q_hi = LGBMRegressor(**tune_lgbm_quantile(x, y, alpha=alpha_hi, n_trials=40)).fit(x, y)
    return q50, q_hi
