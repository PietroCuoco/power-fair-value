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
