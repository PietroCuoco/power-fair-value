"""Tests for the LightGBM model, quantile intervals, and the DM test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from power_fv import validate as V
from power_fv.models import LightGBMModel, RidgeModel


def _nonlinear_market(days: int = 250) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic data with a steep merit-order kink and an hour x load
    interaction - structure a linear model cannot capture but trees can."""
    idx = pd.date_range("2023-01-01", periods=days * 24, freq="h", tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(7)
    residual = rng.normal(30_000, 8_000, n)
    lag = rng.normal(60, 20, n)
    peak = ((idx.hour >= 17) & (idx.hour <= 20)).astype(float)
    kink = np.clip(residual - 38_000, 0, None)  # convex: steep slope above threshold
    price = (
        15
        + 0.0005 * residual
        + 0.005 * kink  # steep merit-order kink (nonlinear)
        + 0.00015 * peak * residual  # hour x load interaction (nonlinear)
        + 0.25 * lag
        + rng.normal(0, 3, n)
    )
    X = pd.DataFrame(
        {
            "residual_load_fc": residual,
            "price_lag_168h": lag,
            "hour": idx.hour,
            "dow": idx.dayofweek,
            "month": idx.month,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
            "is_holiday": np.zeros(n, dtype=int),
        },
        index=idx,
    )
    return X, pd.Series(price, index=idx, name="price")


def test_lightgbm_beats_ridge_on_nonlinear_signal():
    X, y = _nonlinear_market()
    cut = int(len(X) * 0.7)
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y.iloc[:cut], y.iloc[cut:]
    lgbm_mae = V.mae(yte, LightGBMModel(params={"n_estimators": 200}).fit(Xtr, ytr).predict(Xte))
    ridge_mae = V.mae(yte, RidgeModel().fit(Xtr, ytr).predict(Xte))
    assert lgbm_mae < ridge_mae  # trees capture the convex/threshold structure


def test_quantile_backtest_monotone_and_covers():
    X, y = _nonlinear_market(days=220)
    splitter = V.WalkForwardSplitter(initial_train_days=150, step_days=40)
    preds, actual = V.run_quantile_backtest(
        X, y, splitter, quantiles=(0.05, 0.5, 0.95), params={"n_estimators": 120}
    )
    # Monotonicity after anti-crossing sort.
    assert (preds["q05"] <= preds["q50"]).all()
    assert (preds["q50"] <= preds["q95"]).all()
    # 90% interval should cover roughly 90% (wide tolerance for a small sample).
    cov = V.interval_coverage(actual, preds["q05"], preds["q95"])
    assert 0.80 <= cov <= 0.99


def test_interval_coverage_simple():
    actual = pd.Series([1.0, 5.0, 9.0])
    lower = pd.Series([0.0, 0.0, 0.0])
    upper = pd.Series([2.0, 2.0, 10.0])  # second point (5) is outside
    assert abs(V.interval_coverage(actual, lower, upper) - 2.0 / 3.0) < 1e-9


def test_dm_detects_better_model():
    rng = np.random.default_rng(0)
    n = 2000
    e1 = rng.normal(0, 5, n)  # worse model: larger errors
    e2 = rng.normal(0, 2, n)  # better model: smaller errors
    stat, p = V.diebold_mariano(pd.Series(e1), pd.Series(e2), loss="abs")
    assert stat > 0  # model 1 has higher loss -> model 2 better
    assert p < 0.05  # difference is significant


def test_dm_identical_errors_not_significant():
    rng = np.random.default_rng(1)
    e = pd.Series(rng.normal(0, 3, 500))
    stat, p = V.diebold_mariano(e, e, loss="abs")
    assert stat == 0.0
    assert p == 1.0
