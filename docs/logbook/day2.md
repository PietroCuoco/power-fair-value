# Logbook — Day 2: Features, Leakage Guard, and the Validation Harness

**Project:** European Power Fair Value — German Day-Ahead Forecast & Prompt-Curve View
**Author:** Pietro Cuoco
**Scope of Day 2:** Turn the clean dataset into a *point-in-time* feature set,
prove mechanically that it contains no look-ahead leakage, and build the
honest scoring harness (baselines + walk-forward validation + metrics) that
every later model is judged against.

This entry is written to be understood, not just recorded. Each concept is
explained before the result that uses it.

---

## 1. The core problem Day 2 solves: "what did we know, and when?"

A day-ahead price forecast is only meaningful if it uses information a trader
would actually have had at decision time. The German day-ahead auction for
delivery day **D** closes at **12:00 Berlin time on D−1** ("the gate"). After
that, prices for all 24 hours of D are fixed.

So the rule for every feature is simple to state and easy to get wrong:

> A feature used to predict the price of day D may only use information that
> exists at or before 12:00 on D−1.

Most forecasting mistakes in this domain are violations of this rule, called
**leakage** — accidentally feeding the model information from the future. A
model that leaks looks brilliant in testing and fails in production. Day 2 is
built around making leakage structurally impossible, and *proving* it.

---

## 2. What we built (files)

- `src/power_fv/features.py` — builds the feature matrix, constructs the
  residual-load forecast, and contains the **leakage guard**.
- `src/power_fv/validate.py` — the **walk-forward splitter**, the **metrics**,
  and the backtest runner.
- `src/power_fv/models.py` — the two **baselines** (seasonal-naïve and Ridge).
- `tests/test_features.py`, `tests/test_validate.py` — unit tests, including
  tests that the leakage guard actually *catches* leaks.
- `scripts/run_pipeline.py` — extended with `features` and `baselines` stages.

---

## 3. The features, and why each is admissible

We predict the hourly price of day D using 23 features, grouped by *when their
information becomes available*.

**Day-ahead forecasts for day D (7 features).** SMARD publishes day-ahead
forecasts of load, wind, solar, and total generation. A bidder has these before
the gate, so they are admissible. From them we **construct** the single most
important driver:

> `residual_load_fc = forecast load − forecast (wind + PV)`

Residual load is the demand that must be met by *dispatchable* plants (gas,
coal, etc.). Because those plants are stacked cheapest-first (the "merit
order"), the price is set by the marginal plant needed to cover residual load —
so residual load is the dominant fundamental price driver. We built it ourselves
from validated inputs rather than trusting SMARD's filter 413, which we rejected
on Day 1.

**Lagged day-ahead prices (5 features).** The price for any day X is known at
*its* gate, (X−1) 12:00. So the price from 24h, 48h, and 168h (7 days) ago is
fully known by the time we forecast day D. We also add 7-day rolling mean and
standard deviation of price (shifted to stay known). These capture the recent
price level and volatility regime.

**Lagged actuals (4 features).** Realised load and residual load are known only
*as they happen*. At the 12:00 gate on D−1, we do **not** yet know D−1's
afternoon actuals. So realised series are admissible only at lag ≥ 48h. We use
48h and 168h lags of actual load and actual residual load.

**Calendar features (7).** Hour, day-of-week, month, weekend flag, German public
holiday flag, and a sine/cosine encoding of the hour. These are deterministic
(we always know what day it is), so always admissible. They let the model learn
the daily shape (overnight trough, morning ramp, midday solar dip, evening peak)
and weekly pattern.

Total: 7 + 5 + 4 + 7 = **23 features**.

---

## 4. The leakage guard: proving the rule holds

Asserting "no leakage" in a sentence is what most candidates do. We instead
*prove* it mechanically, so it cannot silently break as features are added.

The idea is to give every value two timestamps:

- **gate(D)** — the deadline for predicting day D = (D−1) 12:00 Berlin.
- **information_time(value)** — the earliest moment that value is knowable:
  - a day-ahead *price* for day X: known at gate(X) = (X−1) 12:00;
  - a *forecast* for day X: assumed available at gate(X);
  - a realised *actual* at hour t: known at the end of hour t;
  - a *calendar* value: always known.

A feature is safe if, for every target it feeds,
`information_time(source) ≤ gate(target)`.

On top of that single inequality the guard enforces one extra rule: the **target
price series may only be used at lag ≥ 24h**, because the price of day D itself
is the *label* we are trying to predict — using it would be circular.

Why this catches the subtle case: a realised value at (D−1, 13:00) has
information_time ≈ 14:00 on D−1, which is *after* the 12:00 gate. So a 24-hour
lag of an actual leaks for the afternoon hours — and the guard flags exactly
that. This is why actuals are restricted to lag ≥ 48h while prices are allowed
at lag ≥ 24h.

The guard is enforced in two places: as **unit tests** (including two tests that
plant deliberate leaks — a contemporaneous label and a 24h actual lag — and
confirm the guard raises an error), and as a **pipeline gate** that runs on 2,880
timestamps (120 days, all hours) every time features are built. If a future edit
introduces leakage, the pipeline stops.

---

## 5. Walk-forward validation: why not the usual cross-validation

Standard k-fold cross-validation shuffles data into random folds. For time
series this is invalid: it would train on 2026 data to predict 2024, using the
future to explain the past. The reported score would be meaningless.

Instead we use **expanding-window walk-forward** validation, which mimics how a
desk actually operates:

1. Train on all history up to a cutoff.
2. Predict the next block of days.
3. Move the cutoff forward, add the newly-seen data, and refit.
4. Repeat to the end of the data.

Our settings: an initial training window of **540 days**, then refit every **30
days** and forecast the following 30-day block. This produces ~23 folds. A unit
test verifies the key property: in every fold, *all* training timestamps come
strictly before *all* test timestamps — no peeking. Refitting every 30 days
(rather than every day) reflects a realistic periodic-retrain cadence and is
still leak-free, because training data is always in the past relative to the
block being predicted.

---

## 6. The metrics, and how to read them

All errors are in **EUR/MWh**, the unit of the price.

- **MAE (mean absolute error)** — the average size of the miss. Robust and
  directly interpretable: "on average we are off by X €/MWh."
- **RMSE (root mean squared error)** — squares errors before averaging, so big
  misses count disproportionately. When **RMSE is much larger than MAE**, it
  tells you the error distribution has heavy tails — i.e. a few large misses
  (price spikes) dominate. That is exactly the signature of power prices.
- **Skill score = 1 − MAE_model / MAE_baseline** — improvement over the
  baseline. 0 means "no better than naïve"; +0.5 means "half the error of
  naïve"; 1 would be perfect.

We deliberately **do not use sMAPE** or other percentage errors: power prices
cross zero and go negative, which makes "percent error" explode or become
meaningless.

---

## 7. The baselines, and why they are the right bar

A model is only impressive relative to an honest reference. We use two:

**Seasonal-naïve: price(D, h) = price(D−7, h).** "The same hour, one week ago."
This is a *strong* baseline for power because it matches both the hour of day
and the day of week, capturing the dominant daily-and-weekly seasonality for
free. Beating it requires real fundamental information. (Mechanically it is just
our `price_lag_168h` feature, which the leakage guard already certifies as
safe.)

**Ridge regression** — a linear model on the fundamentals and calendar, with
one-hot calendar encodings, standardized numeric inputs, and L2 regularization.
It is the simplest model that can actually *use* the fundamentals, and it
doubles as an interpretability check: the coefficient on residual-load forecast
should be positive (more residual load → higher price), matching merit-order
logic. This is essentially the "LEAR" model family from the electricity-price
forecasting literature.

---

## 8. Results (validated)

**Feature build.** Matrix shape **29,689 × 23** — exactly the 29,857 clean hours
minus the 168-hour warm-up needed by the longest lag. The constructed
residual-load forecast averages **29,964 MW**, which we validated two ways:
it sits correctly *below* the Bundesnetzagentur winter-quarter figure of ~34 GW
(full-year average must be lower because summer has low demand and high solar),
and the implied wind+PV forecast (53,229 − 29,964 = 23,265 MW) matches Germany's
known ~23 GW average renewable output. Two independent anchors agree, so the
construction is correct.

**Backtest (16,729 out-of-sample hours, ~mid-2024 to May 2026):**

| Model | MAE (€/MWh) | RMSE (€/MWh) | Skill vs naïve |
|---|---|---|---|
| Seasonal-naïve (D−7) | 35.13 | 56.54 | — |
| Ridge (fundamentals + calendar) | 16.42 | 28.21 | **+53.2%** |

**Reading these numbers.** The naïve MAE of €35 reflects how volatile German
day-ahead prices remain in this period; the RMSE/MAE ratio of ~1.6 confirms
heavy spike-driven tails. Ridge more than halves the error.

---

## 9. Is +53% too good? (the validation that matters)

A large skill jump is exactly when to suspect leakage, so we checked rather than
celebrated:

1. **Every Ridge feature already passed the mechanical leakage guard** — no
   contemporaneous price, prices only at lag ≥ 24h, actuals only at lag ≥ 48h.
2. **The walk-forward protocol is tested** to ensure training always precedes
   testing, so there is no fold-level leakage either.
3. **The skill has a legitimate source.** Ridge conditions on day D's
   *fundamentals forecast* (residual load above all), whereas naïve just assumes
   last week repeats. Since residual load drives price via the merit order,
   beating a calendar-only baseline by a wide margin is expected, not anomalous.
4. **It matches the literature.** Regularized-linear models with fundamental
   inputs (the LEAR family in the Lago et al. open benchmark) routinely beat a
   weekly-naïve by 40–60% MAE on German data. +53% is squarely in that range.
5. **The residual error is real.** A €16 MAE (not €2) reflects genuine
   unexplained variance — we have no gas/coal/CO₂ features yet — which confirms
   the model is not somehow seeing the answer.

Conclusion: the result is **strong and legitimate**, and it validates that the
point-in-time feature set is genuinely informative. Day 3 will confirm it
formally with (a) a Diebold–Mariano test for statistical significance of the
improvement and (b) an ablation that removes the forecast features — if skill
collapses, the skill was real fundamentals.

---

## 10. Incident and test status

**Incident — ruff version discrepancy.** Local ruff (0.15) enforced two rules
the sandbox's default config had not: a 100-char line limit (E501) and removal
of redundant quotes on self-referential type annotations (UP037). Fixed by
splitting the long lines and unquoting the annotations; the sandbox now mirrors
the project's `pyproject.toml` so the check is identical going forward. CI green.

**Tests:** **25 passed** (1 smoke + 10 QA/ingest + 7 feature/leakage + 7
validation), including the two positive leak-detection tests. Lint clean.

---

## 11. Open items carried to Day 3

- LightGBM model to capture the *nonlinear, convex* merit-order relationship and
  spike regimes that a linear model cannot.
- Diebold–Mariano test of significance vs. both baselines.
- Quantile prediction intervals (0.05 / 0.50 / 0.95) and a coverage-calibration
  check, for uncertainty quantification and later trade sizing.
- SHAP feature attributions for interpretability.
- Forecast-feature ablation to confirm the source of the Ridge/LightGBM skill.
- Per-hour and per-regime (spike / negative-price) error breakdown.
