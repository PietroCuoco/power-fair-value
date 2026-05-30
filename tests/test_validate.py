"""Tests for walk-forward validation, metrics, and baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from power_fv import validate as V
from power_fv.models import RidgeModel, SeasonalNaive


def _index(days: int = 300) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", periods=days * 24, freq="h", tz="UTC")


def test_splitter_train_always_precedes_test():
    idx = _index(300)
    splitter = V.WalkForwardSplitter(initial_train_days=180, step_days=30)
    folds = list(splitter.split(idx))
    assert len(folds) >= 1
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()  # no peeking into the future


def test_splitter_is_expanding_and_non_overlapping():
    idx = _index(300)
    splitter = V.WalkForwardSplitter(initial_train_days=180, step_days=30)
    folds = list(splitter.split(idx))
    train_sizes = [len(tr) for tr, _ in folds]
    assert train_sizes == sorted(train_sizes)  # training window grows
    # Test blocks do not overlap.
    seen = pd.DatetimeIndex([])
    for _, test_idx in folds:
        assert seen.intersection(test_idx).empty
        seen = seen.union(test_idx)


def test_metrics_known_values():
    y = pd.Series([10.0, 20.0, 30.0])
    p = pd.Series([12.0, 18.0, 33.0])  # errors 2, 2, 3
    assert abs(V.mae(y, p) - 7.0 / 3.0) < 1e-9
    assert abs(V.rmse(y, p) - np.sqrt((4 + 4 + 9) / 3)) < 1e-9


def test_skill_score_sign():
    assert V.skill_score(8.0, 10.0) > 0  # model better than baseline
    assert V.skill_score(12.0, 10.0) < 0  # model worse


def test_seasonal_naive_returns_lag_feature():
    X = pd.DataFrame({"price_lag_168h": [50.0, 60.0]})
    pred = SeasonalNaive().fit(X, pd.Series([0, 0])).predict(X)
    assert list(pred) == [50.0, 60.0]


def test_ridge_fits_and_predicts():
    idx = _index(60)
    rng = np.random.default_rng(0)
    n = len(idx)
    X = pd.DataFrame(
        {
            "residual_load_fc": rng.normal(30_000, 5_000, n),
            "price_lag_168h": rng.normal(50, 10, n),
            "hour": idx.hour,
            "dow": idx.dayofweek,
            "month": idx.month,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
            "is_holiday": np.zeros(n, dtype=int),
        },
        index=idx,
    )
    y = 0.001 * X["residual_load_fc"] + 0.5 * X["price_lag_168h"]
    model = RidgeModel(alpha=1.0).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (n,)
    assert V.mae(y, pred) < y.std()  # captures real signal


def test_run_backtest_covers_test_period():
    idx = _index(300)
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"price_lag_168h": rng.normal(50, 10, len(idx))}, index=idx)
    y = pd.Series(rng.normal(50, 10, len(idx)), index=idx)
    splitter = V.WalkForwardSplitter(initial_train_days=180, step_days=30)
    pred, actual = V.run_backtest(X, y, SeasonalNaive(), splitter)
    assert len(pred) == len(actual)
    assert pred.index.equals(actual.index)
    assert pred.index.is_monotonic_increasing
