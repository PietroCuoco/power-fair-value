"""Research-quality analysis: ablation and error breakdown.

These tools answer two questions a reviewer will ask:

* *Where does the skill come from?* The ablation retrains the model with the
  forecast features removed. If accuracy collapses, the skill is genuine
  fundamentals, not an artifact.
* *Where does the model fail?* The breakdown splits error by hour of day and by
  price regime (negative-price hours, spike hours, normal hours), locating the
  conditions under which the forecast - and any trading signal built on it -
  should be trusted less.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from power_fv.models import LightGBMModel
from power_fv.validate import WalkForwardSplitter, mae, run_backtest

BERLIN = "Europe/Berlin"
FORECAST_PREFIXES = ("fc_",)
CONSTRUCTED_FORECASTS = ("residual_load_fc",)


def _forecast_columns(X: pd.DataFrame) -> list[str]:
    cols = [c for c in X.columns if c.startswith(FORECAST_PREFIXES)]
    cols += [c for c in CONSTRUCTED_FORECASTS if c in X.columns]
    return sorted(set(cols))


def forecast_feature_ablation(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: WalkForwardSplitter,
    params: dict | None = None,
) -> dict[str, float]:
    """Walk-forward MAE for the full model and two ablated variants.

    Variants: full feature set; all forecast features removed; only the
    constructed residual-load forecast removed. Uses the median (point) model.
    """
    fc_cols = _forecast_columns(X)
    variants = {
        "full": X,
        "no_forecasts": X.drop(columns=fc_cols),
    }
    if "residual_load_fc" in X.columns:
        variants["no_residual_load_fc"] = X.drop(columns=["residual_load_fc"])

    results: dict[str, float] = {}
    for name, X_variant in variants.items():
        pred, actual = run_backtest(
            X_variant, y, LightGBMModel(quantile=0.5, params=params), splitter
        )
        results[name] = mae(actual, pred)
    return results


def error_breakdown(
    actual: pd.Series,
    pred: pd.Series,
    negative_threshold: float = 0.0,
    spike_quantile: float = 0.95,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """MAE by price regime and by hour of day.

    Regimes: 'negative' (price < threshold), 'spike' (price >= the
    spike_quantile of realised prices), else 'normal'.

    Returns (regime_table, mae_by_hour, spike_level).
    """
    pred = pd.Series(pred.to_numpy(), index=actual.index)
    err = (actual - pred).abs()

    spike_level = float(actual.quantile(spike_quantile))
    regime = pd.Series("normal", index=actual.index)
    regime[actual < negative_threshold] = "negative"
    regime[actual >= spike_level] = "spike"

    regime_table = pd.DataFrame(
        {"mae": err.groupby(regime).mean(), "n": err.groupby(regime).size()}
    )
    hours = pd.Series(actual.index.tz_convert(BERLIN).hour, index=actual.index)
    mae_by_hour = err.groupby(hours).mean().rename("mae_by_hour")
    return regime_table, mae_by_hour, spike_level


def shap_importance(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int = 3000,
    params: dict | None = None,
    seed: int = 42,
) -> pd.Series:
    """Mean absolute SHAP value per feature (global importance).

    Fits the median LightGBM on the full data (this is an attribution exercise,
    not an out-of-sample skill measurement), then explains a random sample of
    rows with a fast TreeExplainer. Returns features ranked by mean |SHAP|,
    i.e. each feature's average contribution magnitude to the prediction.
    """
    import shap

    model = LightGBMModel(quantile=0.5, params=params).fit(X, y)
    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        positions = rng.choice(len(X), size=sample_size, replace=False)
        sample = X.iloc[np.sort(positions)]
    else:
        sample = X

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(sample)
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
    return importance.sort_values(ascending=False)
