"""Tests for the trading translation module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from power_fv import trade as T

PEAK = list(range(8, 20))  # 08-20 CET


def test_to_daily_products_baseload_and_peak():
    # Two full Berlin-local days, value = local hour -> baseload 11.5, peak 13.5.
    idx = pd.date_range("2024-06-03 00:00", periods=48, freq="h", tz=T.BERLIN)
    s = pd.Series(idx.hour.astype(float), index=idx)
    daily = T.to_daily_products(s, PEAK)
    assert np.allclose(daily["baseload"], 11.5)
    assert np.allclose(daily["peakload"], 13.5)  # mean of hours 8..19


def test_forward_proxy_is_point_in_time():
    days = pd.date_range("2024-01-01", periods=10, freq="D")
    base = pd.Series(np.arange(10.0), index=days)
    proxy = T.forward_proxy(base, window=3)
    # Day index 2 lacks 3 prior days -> NaN. Day 3 uses days 0,1,2 (mean 1.0),
    # day 4 uses days 1,2,3 (mean 2.0) - always strictly past days, never day t.
    assert np.isnan(proxy.iloc[2])
    assert abs(proxy.iloc[3] - 1.0) < 1e-9
    assert abs(proxy.iloc[4] - 2.0) < 1e-9


def test_backtest_pnl_logic():
    days = pd.date_range("2024-01-01", periods=3, freq="D")
    model = pd.Series([60.0, 40.0, 50.0], index=days)
    realized = pd.Series([58.0, 42.0, 51.0], index=days)
    proxy = pd.Series([50.0, 50.0, 50.0], index=days)
    res = T.backtest_signal(model, realized, proxy, threshold=5.0, premium=0.0, cost=1.0)
    # Day1 edge +10>5 -> long, pnl = (58-50)-1 = 7. Day2 edge -10<-5 -> short, pnl = -(42-50)-1 = 7.
    # Day3 edge 0 -> flat, pnl 0.
    assert list(res["position"]) == [1, -1, 0]
    assert abs(res["pnl"].iloc[0] - 7.0) < 1e-9
    assert abs(res["pnl"].iloc[1] - 7.0) < 1e-9
    assert res["pnl"].iloc[2] == 0.0


def test_perfect_foresight_is_profitable():
    # If the model equals realized, the rule positions correctly and makes money.
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-01-01", periods=200, freq="D")
    realized = pd.Series(rng.normal(50, 10, 200), index=days)
    proxy = pd.Series(rng.normal(50, 10, 200), index=days)
    res = T.backtest_signal(realized, realized, proxy, threshold=0.0, premium=0.0, cost=0.0)
    stats = T.summarize_backtest(res)
    assert stats["total_pnl"] > 0
    assert stats["hit_rate"] > 0.95


def test_summarize_counts():
    days = pd.date_range("2024-01-01", periods=4, freq="D")
    res = pd.DataFrame(
        {"position": [1, -1, 0, 1], "pnl": [2.0, -1.0, 0.0, 3.0]},
        index=days,
    )
    res["edge"] = 0.0
    res["settle"] = 0.0
    stats = T.summarize_backtest(res)
    assert stats["n_trades"] == 3
    assert stats["n_long"] == 2
    assert stats["n_short"] == 1
    assert abs(stats["hit_rate"] - 2.0 / 3.0) < 1e-9


def test_confidence_filter_zeros_low_confidence_days():
    days = pd.date_range("2024-01-01", periods=4, freq="D")
    res = pd.DataFrame(
        {"position": [1, -1, 1, -1], "pnl": [2.0, 3.0, -1.0, 4.0]},
        index=days,
    )
    res["edge"] = 0.0
    res["settle"] = 0.0
    mask = pd.Series([True, False, True, False], index=days)
    out = T.apply_confidence_filter(res, mask)
    assert list(out["position"]) == [1, 0, 1, 0]
    assert list(out["pnl"]) == [2.0, 0.0, -1.0, 0.0]
