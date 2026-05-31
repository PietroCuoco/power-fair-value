"""Trading translation: from hourly fair value to a prompt-curve view.

The model produces hourly day-ahead fair values. A desk trades the *prompt
curve* - the near-dated forward (front-week baseload). This module bridges the
two and turns the forecast into a falsifiable, backtestable view.

Pipeline of reasoning:

1. Aggregate hourly fair values into daily tradable products: ``baseload``
   (mean of 24 hours) and ``peakload`` (mean of peak hours, 08-20 CET).
2. Compare the model's baseload fair value to a **forward proxy**. We have real
   day-ahead prices but no free forward feed, so the proxy is the trailing
   realized baseload (power forwards anchor to recent realized levels and
   seasonality). It is a documented stand-in, passed as an argument so a real
   EEX forward series can be dropped in unchanged.
3. Require the edge to clear two hurdles before trading: an empirically
   estimated **risk premium** (the systematic forward-vs-realized wedge) and a
   **noise buffer** sized from the model's own daily forecast-error volatility.
   This stops us trading on forecast noise.
4. Backtest a long/short/flat rule with a per-trade transaction cost.

Every estimate (proxy, premium, threshold) is point-in-time: it uses only past
information (shifted, trailing windows), so the backtest contains no look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BERLIN = "Europe/Berlin"


def to_daily_products(series: pd.Series, peak_hours: list[int]) -> pd.DataFrame:
    """Aggregate an hourly (UTC-indexed) price/forecast into daily products.

    Returns a day-indexed frame with ``baseload`` (mean of all hours) and
    ``peakload`` (mean of the configured peak hours), using Berlin local days.
    """
    loc = series.index.tz_convert(BERLIN)
    frame = pd.DataFrame(
        {"v": series.to_numpy(), "day": loc.normalize(), "hour": loc.hour}
    )
    base = frame.groupby("day")["v"].mean()
    peak = frame[frame["hour"].isin(list(peak_hours))].groupby("day")["v"].mean()
    out = pd.DataFrame({"baseload": base, "peakload": peak})
    out.index = pd.DatetimeIndex(out.index, name="day")
    return out


def forward_proxy(realized_baseload: pd.Series, window: int = 7) -> pd.Series:
    """Forward stand-in: trailing mean of realized baseload, known pre-delivery.

    Shifted by one day so the value for day D uses only days strictly before D.
    """
    return realized_baseload.shift(1).rolling(window, min_periods=window).mean()


def rolling_premium(
    realized_baseload: pd.Series, proxy: pd.Series, window: int = 60
) -> pd.Series:
    """Point-in-time risk premium: trailing mean of (realized - proxy).

    A positive value means realized baseload has tended to exceed the proxy, so
    the proxy systematically under-prices; we de-bias the edge by it.
    """
    return (realized_baseload - proxy).shift(1).rolling(window, min_periods=window).mean()


def rolling_threshold(
    model_baseload: pd.Series, realized_baseload: pd.Series, window: int = 30, k: float = 1.0
) -> pd.Series:
    """Noise buffer: k * trailing std of the model's daily forecast error.

    Sized from how volatile the model's own misses have been recently; we only
    trade when the edge is large relative to that. Shifted to stay point-in-time.
    """
    err = model_baseload - realized_baseload
    return err.shift(1).rolling(window, min_periods=window).std() * k


def backtest_signal(
    model_baseload: pd.Series,
    realized_baseload: pd.Series,
    proxy: pd.Series,
    threshold: pd.Series | float,
    premium: pd.Series | float = 0.0,
    cost: float = 0.5,
) -> pd.DataFrame:
    """Long/short/flat baseload vs the forward proxy; cost-aware P&L per MWh.

    Decision edge = model fair value - proxy - premium. Go long if edge >
    threshold, short if edge < -threshold, else flat. A long that is held to
    delivery earns (realized - proxy); a short earns the negative; each trade
    pays ``cost`` EUR/MWh.
    """
    df = pd.DataFrame({"model": model_baseload, "realized": realized_baseload, "proxy": proxy})
    df["threshold"] = threshold
    df["premium"] = premium
    df = df.dropna()

    edge = df["model"] - df["proxy"] - df["premium"]
    pos = np.where(edge > df["threshold"], 1, np.where(edge < -df["threshold"], -1, 0))
    settle = df["realized"] - df["proxy"]
    pnl = pos * settle - np.abs(pos) * cost
    return pd.DataFrame(
        {"edge": edge, "position": pos, "settle": settle, "pnl": pnl}, index=df.index
    )


def summarize_backtest(res: pd.DataFrame) -> dict[str, float]:
    """Headline stats: activity, hit rate, P&L, and a per-trade info ratio."""
    traded = res[res["position"] != 0]
    n = int(len(traded))
    pnl_std = float(traded["pnl"].std()) if n > 1 else float("nan")
    has_var = n > 1 and pnl_std == pnl_std and pnl_std > 0  # pnl_std==pnl_std excludes NaN
    info_ratio = float(traded["pnl"].mean() / pnl_std) if has_var else float("nan")
    stats = {
        "n_days": int(len(res)),
        "n_trades": n,
        "n_long": int((res["position"] == 1).sum()),
        "n_short": int((res["position"] == -1).sum()),
        "hit_rate": float((traded["pnl"] > 0).mean()) if n else float("nan"),
        "avg_pnl_per_trade": float(traded["pnl"].mean()) if n else float("nan"),
        "total_pnl": float(res["pnl"].sum()),
        "pnl_std_per_trade": pnl_std,
        "info_ratio_per_trade": info_ratio,
    }
    return stats
