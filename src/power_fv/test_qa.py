"""Unit tests for ingest parsers and QA checks (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from power_fv import ingest, qa


def _synthetic_frame() -> pd.DataFrame:
    """Hourly UTC frame spanning Germany's Oct-2024 DST fall-back (25h local day)."""
    idx = pd.date_range("2024-10-25", "2024-10-29", freq="h", tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(0)
    load = 50_000 + 5_000 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 200, n)
    wind_on = np.clip(8_000 + rng.normal(0, 1_000, n), 0, None)
    wind_off = np.clip(2_000 + rng.normal(0, 300, n), 0, None)
    pv = np.clip(3_000 * np.sin(np.arange(n) * 2 * np.pi / 24), 0, None)
    residual = load - wind_on - wind_off - pv
    price = 30 + residual / 1_000 + rng.normal(0, 5, n)

    df = pd.DataFrame(
        {
            "price_da": price,
            "load_actual": load,
            "residual_load_actual": residual,
            "gen_wind_onshore_actual": wind_on,
            "gen_wind_offshore_actual": wind_off,
            "gen_pv_actual": pv,
        },
        index=idx,
    )
    df.index.name = "timestamp_utc"
    # Inject two negative prices and one extreme spike (must be preserved).
    df.iloc[10, df.columns.get_loc("price_da")] = -45.0
    df.iloc[11, df.columns.get_loc("price_da")] = -12.0
    df.iloc[40, df.columns.get_loc("price_da")] = 3000.0
    return df


def test_dst_detects_25h_day():
    rep = qa.check_dst(_synthetic_frame())
    assert rep["n_long_days_25h"] == 1
    assert "2024-10-27" in rep["long_days_sample"]


def test_negative_prices_preserved_and_counted():
    rep = qa.check_negative_prices(_synthetic_frame()["price_da"])
    assert rep["n_negative"] == 2
    assert rep["min_price"] == -45.0


def test_spike_is_flagged():
    rep = qa.check_spikes(_synthetic_frame()["price_da"], threshold=6.0)
    assert rep["n_spikes"] >= 1
    assert 3000.0 in rep["top_spike_values"]


def test_residual_consistency_near_zero():
    rep = qa.check_residual_consistency(_synthetic_frame())
    assert rep["checked"] is True
    assert rep["median_abs_diff_mw"] < 1.0  # built to reconcile exactly


def test_gap_fill_respects_limit():
    df = _synthetic_frame()
    df.iloc[20:22, df.columns.get_loc("load_actual")] = np.nan  # 2h gap -> filled
    df.iloc[60:66, df.columns.get_loc("load_actual")] = np.nan  # 6h gap -> not filled
    clean, rep = qa.run_qa(df, {"qa": {"max_gap_hours_interpolate": 3}})
    assert rep["gap_fill"]["values_filled_per_col"]["load_actual"] == 2
    assert rep["remaining_nans_per_col"]["load_actual"] == 6


def test_dst_ignores_partial_boundary_day():
    # Data starting at UTC midnight in winter -> first Berlin day has 23h (01:00-23:00).
    # That partial boundary day must NOT be flagged as a spring-forward transition.
    idx = pd.date_range("2024-01-01", "2024-01-05", freq="h", tz="UTC")
    df = pd.DataFrame({"price_da": range(len(idx))}, index=idx, dtype="float64")
    rep = qa.check_dst(df)
    assert rep["n_short_days_23h"] == 0
    assert rep["n_long_days_25h"] == 0


def test_index_check_counts_missing_hours():
    df = _synthetic_frame()
    df = df.drop(df.index[30])  # remove one hour
    rep = qa.check_index(df)
    assert rep["n_missing_hours"] == 1
    assert rep["is_monotonic"] is True


# --- ingest parser tests ----------------------------------------------------


def test_parse_index_sorts():
    assert ingest.parse_index({"timestamps": [300, 100, 200]}) == [100, 200, 300]


def test_parse_block_handles_nulls():
    payload = {"series": [[1_700_000_000_000, 42.0], [1_700_003_600_000, None]]}
    s = ingest.parse_block(payload)
    assert len(s) == 2
    assert s.iloc[0] == 42.0
    assert np.isnan(s.iloc[1])
    assert str(s.index.tz) == "UTC"


def test_select_blocks_includes_start_block():
    starts = [0, 100, 200, 300, 400]
    assert ingest._select_blocks(starts, start_ms=150, end_ms=350) == [100, 200, 300]
