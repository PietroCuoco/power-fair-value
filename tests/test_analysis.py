"""Tests for conformal recalibration, ablation, and error breakdown."""

from __future__ import annotations

import numpy as np
import pandas as pd

from power_fv import analysis as A
from power_fv import validate as V


def _market_with_forecasts(days: int = 220) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2023-01-01", periods=days * 24, freq="h", tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(3)
    residual = rng.normal(30_000, 8_000, n)
    lag = rng.normal(60, 20, n)
    price = 10 + 0.0012 * residual + 0.25 * lag + rng.normal(0, 5, n)
    X = pd.DataFrame(
        {
            "residual_load_fc": residual,
            "fc_load_total": residual + rng.normal(20_000, 500, n),
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


def test_cqr_coverage_near_nominal():
    X, y = _market_with_forecasts()
    splitter = V.WalkForwardSplitter(initial_train_days=150, step_days=40)
    preds, actual = V.run_conformal_quantile_backtest(
        X, y, splitter, alpha=0.10, calib_days=30, params={"n_estimators": 120}
    )
    cov = V.interval_coverage(actual, preds["q_lo"], preds["q_hi"])
    assert 0.85 <= cov <= 0.98  # conformal guarantee -> close to nominal 90%


def test_ablation_forecasts_help():
    X, y = _market_with_forecasts()
    splitter = V.WalkForwardSplitter(initial_train_days=150, step_days=40)
    res = A.forecast_feature_ablation(X, y, splitter, params={"n_estimators": 120})
    assert res["full"] < res["no_forecasts"]  # removing fundamentals hurts accuracy


def test_error_breakdown_regimes():
    idx = pd.date_range("2023-01-01", periods=200, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    actual = pd.Series(rng.normal(50, 10, 200), index=idx)
    actual.iloc[:10] = -20.0  # negative regime
    actual.iloc[10:15] = 500.0  # spike regime
    pred = actual + rng.normal(0, 1, 200)
    table, by_hour, spike_level = A.error_breakdown(actual, pred)
    assert set(table.index) <= {"negative", "spike", "normal"}
    assert table.loc["negative", "n"] == 10
    assert len(by_hour) <= 24
    assert spike_level > 50  # 95th percentile sits above the mean


def test_shap_importance_ranks_dominant_driver_top():
    X, y = _market_with_forecasts(days=120)
    # residual_load_fc has the strongest coefficient in _market_with_forecasts,
    # so it (or its near-duplicate fc_load_total) should rank among the top.
    imp = A.shap_importance(X, y, sample_size=1500, params={"n_estimators": 120})
    assert imp.index[0] in {"residual_load_fc", "fc_load_total", "price_lag_168h"}
    assert (imp >= 0).all()
