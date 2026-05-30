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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":
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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
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
