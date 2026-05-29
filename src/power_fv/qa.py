"""Power-specific data-quality checks.

These checks reflect failure modes that matter for electricity data rather than
generic null counting:

* DST transitions produce 23- and 25-hour local days. We verify these and
  report them (they are expected, not errors).
* Negative prices are economically real (must-run plant + renewables) and are
  preserved and reported, never deleted.
* Price spikes are flagged with a robust (MAD-based) z-score and preserved.
* Short gaps are time-interpolated up to a configurable limit; longer gaps are
  flagged for inspection.

Each check is a pure function operating on a DataFrame/Series so it can be unit
tested without any network access. ``run_qa`` orchestrates IO and report output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BERLIN = "Europe/Berlin"


# --- Individual checks (pure) ----------------------------------------------


def check_index(df: pd.DataFrame) -> dict[str, Any]:
    """Monotonicity, duplicates, and hourly-continuity of the UTC index."""
    idx = df.index
    full = pd.date_range(idx.min(), idx.max(), freq="h", tz="UTC")
    missing = full.difference(idx)
    return {
        "n_rows": int(len(df)),
        "is_monotonic": bool(idx.is_monotonic_increasing),
        "n_duplicate_timestamps": int(idx.duplicated().sum()),
        "expected_hours": int(len(full)),
        "n_missing_hours": int(len(missing)),
    }


def check_dst(df: pd.DataFrame) -> dict[str, Any]:
    """Find local days with 23 or 25 hours (spring-forward / fall-back)."""
    local_dates = df.index.tz_convert(BERLIN).date
    counts = pd.Series(local_dates).value_counts()
    short = sorted(str(d) for d, c in counts.items() if c == 23)
    long = sorted(str(d) for d, c in counts.items() if c == 25)
    return {
        "n_short_days_23h": len(short),
        "n_long_days_25h": len(long),
        "short_days_sample": short[:5],
        "long_days_sample": long[:5],
    }


def check_negative_prices(price: pd.Series) -> dict[str, Any]:
    """Count and characterise negative prices (preserved, not removed)."""
    valid = price.dropna()
    neg = valid[valid < 0]
    return {
        "n_negative": int(len(neg)),
        "pct_negative": round(100 * len(neg) / max(len(valid), 1), 3),
        "min_price": None if valid.empty else round(float(valid.min()), 2),
    }


def robust_zscore(series: pd.Series) -> pd.Series:
    """MAD-based z-score, robust to the heavy tails of power prices."""
    x = series.astype("float64")
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (x - med) / mad


def check_spikes(price: pd.Series, threshold: float) -> dict[str, Any]:
    """Flag (do not remove) extreme prices via robust z-score."""
    z = robust_zscore(price.dropna())
    flagged = z[z.abs() > threshold]
    top = price.loc[flagged.index].abs().sort_values(ascending=False).head(5)
    return {
        "threshold": threshold,
        "n_spikes": int(len(flagged)),
        "top_spike_values": [round(float(price.loc[i]), 2) for i in top.index],
    }


def check_ranges(df: pd.DataFrame) -> dict[str, Any]:
    """Sanity bounds: non-price quantities should be non-negative."""
    violations: dict[str, int] = {}
    for col in df.columns:
        if col == "price_da":
            continue
        series = df[col].dropna()
        n_bad = int((series < 0).sum())
        if n_bad:
            violations[col] = n_bad
    return {"negative_value_violations": violations}


def check_residual_consistency(df: pd.DataFrame) -> dict[str, Any]:
    """Cross-check: residual load ~= load - wind - solar (where all present)."""
    needed = {"residual_load_actual", "load_actual",
              "gen_wind_onshore_actual", "gen_wind_offshore_actual", "gen_pv_actual"}
    if not needed.issubset(df.columns):
        return {"checked": False}
    recon = (
        df["load_actual"]
        - df["gen_wind_onshore_actual"]
        - df["gen_wind_offshore_actual"]
        - df["gen_pv_actual"]
    )
    diff = (df["residual_load_actual"] - recon).dropna()
    return {
        "checked": True,
        "median_abs_diff_mw": None if diff.empty else round(float(diff.abs().median()), 1),
        "p95_abs_diff_mw": None if diff.empty else round(float(diff.abs().quantile(0.95)), 1),
    }


# --- Cleaning ---------------------------------------------------------------


def fill_short_gaps(df: pd.DataFrame, max_gap_hours: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reindex to a complete hourly grid and interpolate only *short* gap runs.

    A NaN run is filled only if it is interior (bounded by real values on both
    sides) and its full length is <= ``max_gap_hours``. Longer runs are left as
    NaN so a partially-filled long gap can never masquerade as clean data.
    """
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    out = df.reindex(full)
    out.index.name = df.index.name

    filled: dict[str, int] = {}
    for col in out.columns:
        s = out[col]
        interp = s.interpolate(method="time", limit_area="inside").to_numpy()
        vals = s.to_numpy(copy=True)
        isna = s.isna().to_numpy()
        n_filled, i, n = 0, 0, len(s)
        while i < n:
            if not isna[i]:
                i += 1
                continue
            j = i
            while j < n and isna[j]:
                j += 1
            run_len = j - i
            interior = i > 0 and j < n
            if interior and run_len <= max_gap_hours and not np.isnan(interp[i]):
                vals[i:j] = interp[i:j]
                n_filled += run_len
            i = j
        out[col] = vals
        filled[col] = n_filled
    return out, filled


# --- Orchestrator -----------------------------------------------------------


def run_qa(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run all checks, clean short gaps, and return (clean_df, report)."""
    qa_cfg = cfg.get("qa", {})
    threshold = float(qa_cfg.get("spike_zscore_threshold", 6.0))
    max_gap = int(qa_cfg.get("max_gap_hours_interpolate", 3))

    report: dict[str, Any] = {
        "index": check_index(df),
        "dst": check_dst(df),
        "ranges": check_ranges(df),
        "residual_consistency": check_residual_consistency(df),
    }
    if "price_da" in df.columns:
        report["negative_prices"] = check_negative_prices(df["price_da"])
        report["spikes"] = check_spikes(df["price_da"], threshold)

    clean, filled = fill_short_gaps(df, max_gap)
    report["gap_fill"] = {"max_gap_hours": max_gap, "values_filled_per_col": filled}
    report["remaining_nans_per_col"] = {c: int(clean[c].isna().sum()) for c in clean.columns}
    return clean, report


def _to_markdown(report: dict[str, Any]) -> str:
    idx, dst = report["index"], report["dst"]
    lines = [
        "# Data QA Report",
        "",
        "## Index integrity",
        f"- Rows: {idx['n_rows']:,}  |  Monotonic: {idx['is_monotonic']}  "
        f"|  Duplicates: {idx['n_duplicate_timestamps']}",
        f"- Expected hours: {idx['expected_hours']:,}  |  Missing: {idx['n_missing_hours']}",
        "",
        "## DST transitions (expected, not errors)",
        f"- 23-hour days: {dst['n_short_days_23h']} (e.g. {dst['short_days_sample']})",
        f"- 25-hour days: {dst['n_long_days_25h']} (e.g. {dst['long_days_sample']})",
        "",
    ]
    if "negative_prices" in report:
        n = report["negative_prices"]
        s = report["spikes"]
        lines += [
            "## Price diagnostics",
            f"- Negative-price hours: {n['n_negative']} ({n['pct_negative']}%), "
            f"min {n['min_price']} EUR/MWh (preserved)",
            f"- Spikes (|robust z| > {s['threshold']}): {s['n_spikes']} flagged, "
            f"top |values| {s['top_spike_values']} (preserved)",
            "",
        ]
    rc = report["residual_consistency"]
    if rc.get("checked"):
        lines += [
            "## Cross-series consistency",
            f"- |residual_load - (load - wind - solar)| median "
            f"{rc['median_abs_diff_mw']} MW, p95 {rc['p95_abs_diff_mw']} MW",
            "",
        ]
    gf = report["gap_fill"]
    lines += [
        "## Gap handling",
        f"- Short gaps (<= {gf['max_gap_hours']}h) time-interpolated.",
        f"- Values filled: {gf['values_filled_per_col']}",
        f"- Remaining NaNs: {report['remaining_nans_per_col']}",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], out_dir: str | Path) -> None:
    """Write qa_report.json and qa_report.md to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "qa_report.md").write_text(_to_markdown(report), encoding="utf-8")
