# Logbook — Day 3: The Improved Model, Uncertainty, and Interpretability

**Project:** European Power Fair Value — German Day-Ahead Forecast & Prompt-Curve View
**Author:** Pietro Cuoco
**Scope of Day 3:** Build the gradient-boosted model, prove its improvement is
statistically real, quantify and calibrate its uncertainty, and understand
*where* and *why* it works — using the validation harness from Day 2.

Written to be understood: every method is explained before its result.

---

## 1. The improved model: why trees, and how it forecasts

Ridge (Day 2) can only draw straight-line relationships. But the price-vs-load
relationship is **nonlinear and convex**: as residual load rises and cheap plant
runs out, price climbs ever more steeply (the "merit order"), and spike and
negative-price regimes are sharply nonlinear. **LightGBM** (gradient-boosted
decision trees) captures exactly that kind of structure: kinks, thresholds, and
interactions between features.

The point forecast is the model's **median** (a quantile-0.5 / pinball objective,
equivalent to optimizing MAE). The median is deliberately robust to spikes — it
predicts the typical outcome rather than being dragged by extreme tail prices.
Calendar features (hour, day-of-week, month) are passed as categoricals so the
tree splits on them as unordered categories.

It runs through the **same walk-forward harness** as the baselines, so its score
is directly comparable.

---

## 2. Files built

- `src/power_fv/models.py` — added `LightGBMModel` (point + quantile).
- `src/power_fv/validate.py` — added the **Diebold-Mariano test**, the
  **quantile backtest** + interval coverage, and **conformalized quantile
  regression (CQR)**.
- `src/power_fv/analysis.py` — **forecast ablation**, **error breakdown**, and
  **SHAP importance**.
- `tests/test_model.py`, `tests/test_analysis.py` — unit tests for all of it.
- `scripts/run_pipeline.py` — stages: `model`, `conformal`, `ablation`,
  `breakdown`, `shap`.

---

## 3. Headline results (real data, 16,729 out-of-sample hours)

| Model | MAE (EUR/MWh) | RMSE (EUR/MWh) | Skill vs naive |
|---|---|---|---|
| Seasonal-naive | 35.13 | 56.54 | — |
| Ridge | 16.42 | 28.21 | +53.2% |
| **LightGBM** | **14.23** | **27.33** | **+59.5%** |

LightGBM also beats Ridge by +13.3% MAE. MAE improved more than RMSE because the
median objective sharpens typical-hour accuracy without chasing spikes (RMSE is
spike-dominated).

---

## 4. Is the improvement real? The Diebold-Mariano test

A lower MAE could in principle be luck. The **Diebold-Mariano (DM) test** checks
whether the difference in forecast accuracy between two models is statistically
significant. It looks at the per-hour loss differential (here, the difference in
absolute errors), and tests whether its mean is distinguishable from zero, using
a HAC (Newey-West) variance that accounts for the autocorrelation of hourly
errors.

Results (positive stat => LightGBM more accurate):
- DM vs seasonal-naive: stat **+22.11**, p ~ 0.
- DM vs Ridge: stat **+10.14**, p ~ 0.

A stat of +10 is roughly a ten-sigma result: the chance it is luck is
vanishingly small. So **LightGBM significantly beats both** — the nonlinearity
genuinely matters. (Caveat to test in robustness: the exact stat depends on the
HAC lag, set to 24h; the *significance* is robust to that choice, but we will
show stability across a few lags so it cannot be challenged.)

---

## 5. Uncertainty: quantile intervals and the conformal fix

We don't just want a point forecast; we want a range. LightGBM trained at the
5% and 95% quantiles gives a 90% **prediction interval**. The test of an interval
is **coverage**: does the 90% band actually contain ~90% of outcomes?

It did not — raw coverage was only **71.8%**. The intervals were too narrow: raw
quantile regression captures conditional spread but not full predictive
uncertainty (model error, regime shifts, fat spike tails).

The fix is **conformalized quantile regression (CQR)**: within each fold we hold
out the most recent slice of training data as a calibration set, measure how far
reality spills outside the predicted band there, and widen the band by exactly
that amount. This carries a finite-sample coverage guarantee.

Result: coverage **71.8% -> 88.9%**, interval width 38.6 -> 56.0 EUR/MWh. Close
to the 90% target; the small shortfall is honest, reflecting that power prices
are non-stationary (CQR's guarantee assumes exchangeability). We now have
calibrated uncertainty to size trades with on Day 4.

---

## 6. Where does the skill come from? The ablation

We retrain LightGBM with features removed:

| Variant | MAE | vs full |
|---|---|---|
| full | 14.24 | — |
| no forecasts | 23.69 | +9.45 |
| no residual_load_fc only | 14.19 | -0.05 |

Removing **all** forecast features costs +9.45 MAE (a 66% increase) — proving the
skill is genuine fundamentals, not price autocorrelation. (Even without
forecasts, MAE 23.69 still beats naive's 35.13, so skill has two sources: ~half
from price-history/calendar structure, ~half from fundamentals forecasts.)

Removing **only** `residual_load_fc` costs essentially nothing, because its
components (`fc_load_total`, `fc_gen_wind_pv`) remain and the model recovers the
same information from them. The feature is informative but *redundant given its
inputs*.

---

## 7. Where does the model fail? The error breakdown

| Regime | MAE (EUR/MWh) | Hours |
|---|---|---|
| normal | 11.87 | 14,854 |
| negative | 15.60 | 1,038 |
| spike (>= 163 EUR/MWh) | 54.39 | 837 |

Internal-consistency check: the hour-weighted average of these is exactly
14.23 — the headline MAE — confirming the breakdown partitions the whole
out-of-sample set correctly. The model is strong on normal hours and struggles
on spikes (intrinsically hard, and our median objective does not chase them).
The worst hours are **17-19 local (evening peak)**, when residual load peaks and
prices are most volatile. This is the honest "trust the signal less here" map for
the trading layer.

---

## 8. SHAP, and why it appears to contradict the ablation

**SHAP** attributes each prediction to its features. Global importance
(mean |SHAP|, EUR/MWh of average influence):

| Feature | mean |SHAP| |
|---|---|
| residual_load_fc | 21.80 |
| price_lag_24h | 6.46 |
| price_lag_168h | 4.58 |
| fc_gen_wind_pv | 4.49 |
| price_lag_48h | 2.69 |
| price_roll7_mean | 2.13 |
| fc_gen_total | 1.94 |
| month | 1.92 |

`residual_load_fc` dominates — yet the ablation said removing it costs nothing.
**This is not a contradiction; the two methods measure different things:**

* **SHAP** measures *how much the fitted model relies on a feature*, holding the
  model fixed. The tree prefers `residual_load_fc` as its split variable because
  it is the single cleanest encoding of the merit-order signal, so SHAP gives it
  large attribution.
* **Refit-ablation** measures *how much accuracy is lost if the feature is
  removed and the model relearns*. Because the components remain, the model
  substitutes seamlessly, so the loss is ~0.

Under redundant (correlated) features these diverge legitimately: "what does the
model use?" (SHAP: residual load, overwhelmingly) is a different question from
"what unique information would I lose?" (ablation: none). A permutation test
would agree with SHAP. Reported together, the three tools give one coherent
picture: residual load is the dominant driver; the skill is real fundamentals;
the engineered feature is informative but redundant.

---

## 9. Incidents and test status

- A unit test fixture initially had nonlinearity too weak relative to noise, so
  LightGBM did not beat Ridge on it; the fixture (not the model) was fixed to
  contain a realistic merit-order kink and interaction. Lesson: a "model A beats
  B" test is only meaningful if the data contains the structure being tested.
- The stricter local ruff flagged line-length again; lines split. CI green.
- **Tests: 34 passed.** Lint clean.

---

## 10. Open items carried to Day 4 (the high-value day)

- **Trading translation (Requirement 3):** aggregate hourly fair values into a
  front-week baseload view, estimate the day-ahead-vs-forward risk premium,
  define a falsifiable signal with invalidation conditions, and run a crude
  cost-aware backtest. Use the calibrated CQR intervals for sizing and the error
  breakdown to flag low-trust regimes.
- **LLM component (Requirement 4):** news/outage -> structured features via a
  schema-validated, logged LLM call.
- Figures (Day 5) will visualize all Day 3 results: forecast-vs-actual week,
  merit-order scatter, model comparison, MAE by hour/regime, coverage before/
  after CQR, ablation, and SHAP.
