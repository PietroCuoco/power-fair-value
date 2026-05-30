"""Point-in-time feature engineering for next-day hourly price forecasting.

Every feature must be knowable at the day-ahead gate (12:00 Berlin on D-1) for
delivery day D. This module builds the feature matrix and provides a *leakage
guard* (:func:`assert_no_leakage`) that mechanically verifies the information
timeline, so leakage cannot creep in unnoticed as features are added.

Information timeline
--------------------
* gate(D) = (D-1) 12:00 Berlin.
* A day-ahead price for delivery day X is known at gate(X) = (X-1) 12:00.
  Hence the target series is admissible only at lag >= 24h; its own value is
  the label.
* Forecast series (``fc_*``) for day D are admissible (the day-ahead
  fundamentals available to a bidder at the gate).
* Actual (realised) series are known only as they occur and are admissible
  only at lag >= 48h: a lag-24h actual is still unknown for the afternoon
  hours of D-1 at a 12:00 gate.
* Calendar features are deterministic and always admissible.

Lags are expressed in hours on the contiguous UTC grid. Across the two annual
DST transitions a 24h-UTC lag drifts by one local hour relative to "same local
hour yesterday"; this affects only two days per year and is documented rather
than special-cased in this prototype.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

BERLIN = "Europe/Berlin"
GATE_HOUR = 12
TARGET = "price_da"

PRICE_LAGS = (24, 48, 168)  # D-1, D-2, D-7 same hour
ACTUAL_LAGS = (48, 168)  # >= 48h only (see module docstring)

FORECAST_COLS = (
    "fc_load_total",
    "fc_gen_total",
    "fc_gen_wind_pv",
    "fc_gen_wind_onshore",
    "fc_gen_wind_offshore",
    "fc_gen_pv",
    "residual_load_fc",  # constructed below
)
ACTUAL_LAG_COLS = ("residual_load_actual", "load_actual")


class LeakageError(AssertionError):
    """Raised when a feature would use information unavailable at the gate."""


def add_constructed(df: pd.DataFrame) -> pd.DataFrame:
    """Add the constructed day-ahead residual-load forecast.

    residual_load_fc = forecast load - forecast (wind + PV). This is the
    merit-order driver built from validated forecast inputs (filters 411 and
    5097); we deliberately do not use SMARD filter 413, which failed magnitude
    validation against the Bundesnetzagentur residual-load figure.
    """
    out = df.copy()
    out["residual_load_fc"] = out["fc_load_total"] - out["fc_gen_wind_pv"]
    return out


def _calendar_features(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    loc = index.tz_convert(BERLIN)
    years = range(loc.year.min(), loc.year.max() + 1)
    de_holidays = holidays.Germany(years=years)
    is_hol = np.array([d.date() in de_holidays for d in loc], dtype=int)
    return {
        "hour": pd.Series(loc.hour, index=index, dtype="int16"),
        "dow": pd.Series(loc.dayofweek, index=index, dtype="int16"),
        "month": pd.Series(loc.month, index=index, dtype="int16"),
        "is_weekend": pd.Series((loc.dayofweek >= 5).astype(int), index=index),
        "is_holiday": pd.Series(is_hol, index=index),
        "hour_sin": pd.Series(np.sin(2 * np.pi * loc.hour / 24), index=index),
        "hour_cos": pd.Series(np.cos(2 * np.pi * loc.hour / 24), index=index),
    }


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, tuple[str, int]]]:
    """Build the feature matrix, target, and per-feature provenance metadata.

    Returns
    -------
    X, y, meta
        ``meta`` maps each feature column to ``(kind, lag_hours)`` where kind is
        one of {"forecast", "target", "actual", "calendar"}. It is the input to
        :func:`assert_no_leakage`.
    """
    df = add_constructed(df)
    feats: dict[str, pd.Series] = {}
    meta: dict[str, tuple[str, int]] = {}

    # Day-ahead forecasts for delivery day D (aligned, lag 0).
    for col in FORECAST_COLS:
        feats[col] = df[col]
        meta[col] = ("forecast", 0)

    # Lagged day-ahead prices (target series), admissible at lag >= 24h.
    for lag in PRICE_LAGS:
        name = f"price_lag_{lag}h"
        feats[name] = df[TARGET].shift(lag)
        meta[name] = ("target", lag)

    # Rolling price statistics over the prior 7 days, shifted 24h to stay known.
    roll = df[TARGET].shift(24).rolling(168, min_periods=24)
    feats["price_roll7_mean"] = roll.mean()
    feats["price_roll7_std"] = roll.std()
    meta["price_roll7_mean"] = ("target", 24)
    meta["price_roll7_std"] = ("target", 24)

    # Lagged actuals, admissible at lag >= 48h.
    for base in ACTUAL_LAG_COLS:
        for lag in ACTUAL_LAGS:
            name = f"{base}_lag_{lag}h"
            feats[name] = df[base].shift(lag)
            meta[name] = ("actual", lag)

    # Deterministic calendar features.
    for name, series in _calendar_features(df.index).items():
        feats[name] = series
        meta[name] = ("calendar", 0)

    X = pd.DataFrame(feats)
    y = df[TARGET]
    return X, y, meta


def make_modeling_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, tuple[str, int]]]:
    """Build features and drop the warm-up rows that contain lag NaNs."""
    X, y, meta = build_features(df)
    valid = X.dropna().index.intersection(y.dropna().index)
    return X.loc[valid], y.loc[valid], meta


# --- Leakage guard ----------------------------------------------------------


def gate_time(target_ts: pd.Timestamp) -> pd.Timestamp:
    """The day-ahead gate (D-1 12:00 Berlin) for a delivery timestamp on day D."""
    local_day = target_ts.tz_convert(BERLIN).normalize()
    return local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=GATE_HOUR)


def information_time(kind: str, source_ts: pd.Timestamp) -> pd.Timestamp:
    """The earliest clock time at which a source value becomes known."""
    if kind == "calendar":
        return pd.Timestamp("1900-01-01", tz=BERLIN)
    if kind in ("forecast", "target"):
        # Known at the gate of the source value's own delivery day.
        return gate_time(source_ts)
    if kind == "actual":
        # Realised, known at the end of its delivery hour.
        return source_ts.tz_convert(BERLIN) + pd.Timedelta(hours=1)
    raise ValueError(f"Unknown feature kind: {kind}")


def assert_no_leakage(meta: dict[str, tuple[str, int]], sample_index: pd.DatetimeIndex) -> None:
    """Verify every feature is knowable at the gate for every sampled target.

    Two rules are enforced:

    1. The target series may only be used at lag >= 24h (its own value is the
       label, known exactly at the gate).
    2. For every feature and every sampled delivery timestamp, the information
       time of the source value must not exceed the target's gate time.
    """
    for col, (kind, lag) in meta.items():
        if kind == "target" and lag < 24:
            raise LeakageError(
                f"'{col}': target series used at lag {lag}h (< 24h) - that is the label."
            )

    for target_ts in sample_index:
        gate = gate_time(target_ts)
        for col, (kind, lag) in meta.items():
            source_ts = target_ts - pd.Timedelta(hours=lag)
            info = information_time(kind, source_ts)
            if info > gate:
                raise LeakageError(
                    f"'{col}' ({kind}, lag {lag}h) leaks at target {target_ts}: "
                    f"info_time {info} > gate {gate}."
                )
