"""Tests for point-in-time features and the leakage guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from power_fv import features as F


def _synthetic_market(hours: int = 24 * 90) -> pd.DataFrame:
    """A 90-day hourly UTC frame with all 12 ingested columns."""
    idx = pd.date_range("2024-06-01", periods=hours, freq="h", tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(1)
    t = np.arange(n)
    load = 55_000 + 6_000 * np.sin(t * 2 * np.pi / 24) + rng.normal(0, 300, n)
    wind_on = np.clip(9_000 + rng.normal(0, 1_500, n), 0, None)
    wind_off = np.clip(2_500 + rng.normal(0, 400, n), 0, None)
    pv = np.clip(8_000 * np.sin(t * 2 * np.pi / 24), 0, None)
    residual = load - wind_on - wind_off - pv
    price = 35 + residual / 1_200 + rng.normal(0, 4, n)
    return pd.DataFrame(
        {
            "price_da": price,
            "load_actual": load,
            "residual_load_actual": residual,
            "gen_wind_onshore_actual": wind_on,
            "gen_wind_offshore_actual": wind_off,
            "gen_pv_actual": pv,
            "fc_gen_total": load + rng.normal(0, 500, n),
            "fc_gen_wind_pv": wind_on + wind_off + pv + rng.normal(0, 600, n),
            "fc_gen_wind_onshore": wind_on + rng.normal(0, 400, n),
            "fc_gen_wind_offshore": wind_off + rng.normal(0, 150, n),
            "fc_gen_pv": pv + rng.normal(0, 300, n),
            "fc_load_total": load + rng.normal(0, 400, n),
        },
        index=idx,
    )


def test_residual_forecast_is_load_minus_wind_pv():
    df = F.add_constructed(_synthetic_market())
    expected = df["fc_load_total"] - df["fc_gen_wind_pv"]
    assert np.allclose(df["residual_load_fc"], expected)


def test_modeling_frame_drops_warmup_nans():
    X, y, _ = F.make_modeling_frame(_synthetic_market())
    assert not X.isna().any().any()
    assert len(X) == len(y)
    # 168h warm-up for the longest lag should be dropped.
    assert len(X) == 24 * 90 - 168


def test_leakage_guard_passes_on_real_features():
    df = _synthetic_market()
    _, _, meta = F.build_features(df)
    sample = df.index[-14 * 24:]  # last two weeks, all hours
    F.assert_no_leakage(meta, sample)  # must not raise


def test_guard_catches_contemporaneous_label():
    df = _synthetic_market()
    bad = {"price_now": ("target", 0)}
    with pytest.raises(F.LeakageError, match="label"):
        F.assert_no_leakage(bad, df.index[-48:])


def test_guard_catches_lag24_actual_in_afternoon():
    df = _synthetic_market()
    bad = {"load_lag_24h": ("actual", 24)}
    # Include afternoon target hours, where a lag-24h actual post-dates the gate.
    sample = df.index[-48:]
    with pytest.raises(F.LeakageError, match="leaks"):
        F.assert_no_leakage(bad, sample)


def test_lag48_actual_is_safe():
    df = _synthetic_market()
    ok = {"load_lag_48h": ("actual", 48)}
    F.assert_no_leakage(ok, df.index[-48:])  # must not raise


def test_forecast_aligned_is_safe():
    df = _synthetic_market()
    ok = {"fc_load_total": ("forecast", 0)}
    F.assert_no_leakage(ok, df.index[-48:])  # known at gate, must not raise
