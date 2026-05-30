"""Walk-forward validation and metrics.

Time-series evaluation must never use future information, so we use an
expanding-window walk-forward protocol rather than random k-fold:

* Train on all history up to a cutoff, predict the next ``step_days`` block,
  then advance the cutoff by ``step_days`` and refit. This mimics a desk that
  retrains periodically rather than intraday.
* Within every fold, all training targets precede all test targets, so there is
  no fold-level leakage. (Feature-level leakage is handled separately by the
  guard in :mod:`power_fv.features`.)

Metrics are reported in EUR/MWh. We headline MAE and RMSE, break MAE down by
hour of day (where power-price error concentrates), and report a skill score
against the seasonal-naive baseline. sMAPE is deliberately avoided because power
prices cross zero and go negative, which makes percentage errors meaningless.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Model(Protocol):
    """Minimal fit/predict interface used by the backtest runner."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Model: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class WalkForwardSplitter:
    """Expanding-window splitter over a sorted, tz-aware hourly index."""

    initial_train_days: int
    step_days: int

    def split(self, index: pd.DatetimeIndex) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        index = index.sort_values()
        start_day = index.min().normalize()
        last_day = index.max().normalize()
        step = pd.Timedelta(days=self.step_days)
        test_start = start_day + pd.Timedelta(days=self.initial_train_days)

        while test_start <= last_day:
            test_end = test_start + step
            train_idx = index[index < test_start]
            test_idx = index[(index >= test_start) & (index < test_end)]
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx
            test_start = test_end


# --- Metrics ----------------------------------------------------------------


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true.to_numpy() - np.asarray(y_pred))))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true.to_numpy() - np.asarray(y_pred)) ** 2)))


def skill_score(mae_model: float, mae_baseline: float) -> float:
    """1 - MAE_model / MAE_baseline. Positive means better than the baseline."""
    return 1.0 - mae_model / mae_baseline


def per_hour_mae(y_true: pd.Series, y_pred: pd.Series, hours: pd.Series) -> pd.Series:
    """MAE grouped by hour of day (hours aligned to y_true's index)."""
    err = (y_true - pd.Series(np.asarray(y_pred), index=y_true.index)).abs()
    return err.groupby(hours.loc[y_true.index]).mean().rename("mae_by_hour")


def summarize(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "n": int(len(y_true))}


# --- Backtest runner --------------------------------------------------------


def run_backtest(
    X: pd.DataFrame, y: pd.Series, model: Model, splitter: WalkForwardSplitter
) -> tuple[pd.Series, pd.Series]:
    """Walk-forward backtest. Returns (predictions, aligned actuals)."""
    preds: list[pd.Series] = []
    for train_idx, test_idx in splitter.split(X.index):
        model.fit(X.loc[train_idx], y.loc[train_idx])
        yhat = model.predict(X.loc[test_idx])
        preds.append(pd.Series(np.asarray(yhat), index=test_idx))
    if not preds:
        raise ValueError("No folds produced - check splitter parameters vs data span.")
    pred = pd.concat(preds).sort_index()
    actual = y.loc[pred.index]
    return pred, actual


# --- Quantile backtest and interval coverage --------------------------------


def run_quantile_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: WalkForwardSplitter,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
    params: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Walk-forward backtest fitting one LightGBM per quantile per fold.

    Returns a DataFrame with one column per quantile (e.g. q05, q50, q95) and
    the aligned actuals. Quantiles are sorted row-wise to remove any quantile
    crossing (a separately-fit q95 occasionally dips below q50).
    """
    from power_fv.models import LightGBMModel

    cols: dict[float, list[pd.Series]] = {q: [] for q in quantiles}
    for train_idx, test_idx in splitter.split(X.index):
        for q in quantiles:
            model = LightGBMModel(quantile=q, params=params).fit(
                X.loc[train_idx], y.loc[train_idx]
            )
            cols[q].append(pd.Series(model.predict(X.loc[test_idx]), index=test_idx))

    preds = pd.DataFrame(
        {f"q{int(q * 100):02d}": pd.concat(s).sort_index() for q, s in cols.items()}
    )
    # Enforce monotonicity across quantiles (anti-crossing).
    preds.iloc[:, :] = np.sort(preds.to_numpy(), axis=1)
    actual = y.loc[preds.index]
    return preds, actual


def interval_coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Empirical fraction of actuals falling within [lower, upper]."""
    inside = (actual.to_numpy() >= lower.to_numpy()) & (actual.to_numpy() <= upper.to_numpy())
    return float(np.mean(inside))


# --- Diebold-Mariano test ---------------------------------------------------


def diebold_mariano(
    error1: pd.Series, error2: pd.Series, loss: str = "abs", h: int = 24
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Parameters
    ----------
    error1, error2 : forecast errors (actual - prediction) of model 1 and 2.
    loss : "abs" (compares MAE) or "sq" (compares MSE).
    h : Newey-West truncation lag. Forecast errors are autocorrelated (daily
        structure in hourly data), so we use a HAC variance with a Bartlett
        kernel. Default 24 covers one day of intraday autocorrelation.

    Returns
    -------
    (dm_stat, p_value)
        The loss differential is d = loss(error1) - loss(error2). A positive
        dm_stat means model 1 has the higher loss, i.e. model 2 is more
        accurate; the p-value is two-sided (standard normal).
    """
    from scipy.stats import norm

    e1 = np.asarray(error1, dtype="float64")
    e2 = np.asarray(error2, dtype="float64")
    if loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "sq":
        d = e1**2 - e2**2
    else:
        raise ValueError("loss must be 'abs' or 'sq'")

    n = len(d)
    d_bar = d.mean()
    d_dem = d - d_bar

    # Newey-West HAC long-run variance with Bartlett weights.
    gamma0 = np.sum(d_dem * d_dem) / n
    var = gamma0
    for lag in range(1, h):
        if lag >= n:
            break
        cov = np.sum(d_dem[lag:] * d_dem[:-lag]) / n
        weight = 1.0 - lag / h
        var += 2.0 * weight * cov

    if var <= 0:  # identical error series -> no difference to detect
        return 0.0, 1.0
    dm_stat = d_bar / np.sqrt(var / n)
    p_value = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)

# --- Conformalized quantile regression (CQR) --------------------------------


def run_conformal_quantile_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: WalkForwardSplitter,
    quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
    alpha: float = 0.10,
    calib_days: int = 90,
    params: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Conformalized quantile regression (Romano et al., 2019).

    Raw quantile intervals under-cover because they capture conditional spread
    but not full predictive uncertainty. CQR fixes this with a finite-sample
    coverage guarantee: within each fold the most recent ``calib_days`` of the
    training window are held out as a calibration set; the conformity score
    E = max(q_lo - y, y - q_hi) measures how far reality spills outside the
    predicted band; its (1-alpha) empirical quantile Q then widens the test
    interval to [q_lo - Q, q_hi + Q].

    Returns a DataFrame (q_lo, q_mid, q_hi) and the aligned actuals.
    """
    from power_fv.models import LightGBMModel

    q_lo, q_mid, q_hi = quantiles
    blocks: list[pd.DataFrame] = []
    for train_idx, test_idx in splitter.split(X.index):
        cutoff = train_idx.max().normalize() - pd.Timedelta(days=calib_days)
        fit_idx = train_idx[train_idx < cutoff]
        calib_idx = train_idx[train_idx >= cutoff]
        if len(fit_idx) < 24 * 30 or len(calib_idx) < 24 * 7:
            fit_idx, calib_idx = train_idx, train_idx[-24 * 30:]

        models = {
            q: LightGBMModel(quantile=q, params=params).fit(X.loc[fit_idx], y.loc[fit_idx])
            for q in (q_lo, q_mid, q_hi)
        }
        lo_c = models[q_lo].predict(X.loc[calib_idx])
        hi_c = models[q_hi].predict(X.loc[calib_idx])
        yc = y.loc[calib_idx].to_numpy()
        scores = np.maximum(lo_c - yc, yc - hi_c)

        n = len(scores)
        k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        q_adj = np.sort(scores)[k - 1]

        block = pd.DataFrame(
            {
                "q_lo": models[q_lo].predict(X.loc[test_idx]) - q_adj,
                "q_mid": models[q_mid].predict(X.loc[test_idx]),
                "q_hi": models[q_hi].predict(X.loc[test_idx]) + q_adj,
            },
            index=test_idx,
        )
        blocks.append(block)

    preds = pd.concat(blocks).sort_index()
    preds.iloc[:, :] = np.sort(preds.to_numpy(), axis=1)  # anti-crossing
    actual = y.loc[preds.index]
    return preds, actual
