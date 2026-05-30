"""Baseline forecasting models.

Two honest baselines that the improved model must beat:

* :class:`SeasonalNaive` - price(D, h) = price(D-7, h). For power this is a
  strong baseline: it matches both hour of day and day of week, capturing the
  dominant weekly-and-daily seasonality. It is simply the ``price_lag_168h``
  feature, which the leakage guard already certifies as point-in-time safe.
* :class:`RidgeModel` - a regularized linear model on the fundamentals and
  calendar features. It doubles as an interpretable sanity check: the
  coefficient on residual-load forecast should be positive (merit-order logic).

Both expose a fit/predict interface compatible with the walk-forward runner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["hour", "dow", "month", "is_weekend", "is_holiday"]


class SeasonalNaive:
    """price(D, h) = price(D-7, h), read directly from the lag-168h feature."""

    column = "price_lag_168h"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaive:
        if self.column not in X.columns:
            raise KeyError(f"SeasonalNaive needs the '{self.column}' feature.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy()


class RidgeModel:
    """Regularized linear baseline with one-hot calendar and scaled numerics."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.pipe: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeModel:
        categorical = [c for c in CATEGORICAL if c in X.columns]
        numeric = [c for c in X.columns if c not in categorical]
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("num", StandardScaler(), numeric),
            ]
        )
        self.pipe = Pipeline([("pre", pre), ("ridge", Ridge(alpha=self.alpha))])
        self.pipe.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError("RidgeModel must be fit before predict.")
        return self.pipe.predict(X)


class LightGBMModel:
    """Gradient-boosted trees: the improved model.

    Trees capture what the linear baseline cannot - the nonlinear, convex
    merit-order relationship (price rises steeply once residual load pushes into
    expensive peaking plant) and sharp spike/negative-price regimes.

    If ``quantile`` is None the model predicts the conditional median-like point
    forecast with an L1 (MAE-aligned, spike-robust) objective. If ``quantile``
    is set, it fits LightGBM's pinball-loss quantile objective at that level,
    used to build prediction intervals.

    Calendar columns (hour, dow, month) are passed as categorical so the trees
    split on them as unordered categories. They are kept as non-negative
    integers (not pandas ``category`` dtype) so the value->category mapping is
    identical across walk-forward folds.
    """

    CATEGORICAL = ["hour", "dow", "month"]

    def __init__(self, quantile: float | None = None, params: dict | None = None) -> None:
        self.quantile = quantile
        base = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if quantile is None:
            base["objective"] = "regression_l1"
        else:
            base["objective"] = "quantile"
            base["alpha"] = quantile
        if params:
            base.update(params)
        self.params = base
        self.model: object | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMModel:
        import lightgbm as lgb

        cat = [c for c in self.CATEGORICAL if c in X.columns]
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y, categorical_feature=cat)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBMModel must be fit before predict.")
        return self.model.predict(X)
