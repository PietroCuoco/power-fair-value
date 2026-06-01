# European Power Fair Value — German Day-Ahead Forecast & Prompt-Curve View

**Pietro Cuoco** · `pietrocuoco10@gmail.com` · June 2026

## Objective

Build a daily fair-value view for German (DE/LU) day-ahead (DA) power, validate it, and translate it into a tradable prompt-curve view — with one programmatic LLM component. The emphasis throughout is on leakage-safe validation and claims that are measurabe rather than flattering.

## Data and quality assurance

Twelve hourly SMARD series (Bundesnetzagentur, CC BY 4.0) spanning Jan 2023 – May 2026 (29,857 rows): the DA price, actual load and wind/solar generation, and their published day-ahead forecasts. QA confirmed zero missing hours after
DST-aware alignment (4 spring-forward and 3 fall-back days handled), a negative price share of 5.2% with a floor at −500 EUR/MWh (the EPEX limit — a sign the
data is genuine), and 109 price spikes, all preserved rather than winsorised. The residual-load forecast is constructed as load-forecast minus wind-and-solar
forecast, after a candidate raw series was rejected on a magnitude check.

## Forecasting

Features are fundamentals forecasts (residual load, wind/solar, load), price autoregressors (24h/48h/168h lags, 7-day rolling stats), and calendar terms.
A **leakage guard** attaches an information timestamp to every feature and asserts that nothing known only after the D-1 12:00 Berlin gate enters the model
for delivery day D; it runs both as a pipeline check and as unit tests.

Models are evaluated by **expanding-window walk-forward** (no shuffling), producing 16,729 out-of-sample hourly forecasts (~mid-2024 to May 2026).

| Model | MAE | RMSE | Skill vs naive |
|---|---|---|---|
| Seasonal-naive | 35.13 | 56.54 | — |
| Ridge | 16.42 | 28.21 | +53.2% |
| **LightGBM** | **14.23** | **27.33** | **+59.5%** |

The nonlinearity is warranted by the merit-order structure (Fig. 1) and is statistically real: Diebold–Mariano tests give +22.1 vs naive and +10.1 vs Ridge
(p ≈ 0). Figure 2 shows the accuracy gap.

![Fig 1. Merit-order structure](reports/figures/01_merit_order.png)

![Fig 2. Model comparison](reports/figures/02_model_comparison.png)

## Calibrated uncertainty

Raw quantile-regression intervals were over-confident (71.8% empirical coverage for a nominal 90%). **Conformalized quantile regression** restores coverage to 88.9% at the honest cost of a wider band (38.6 → 56.0 EUR/MWh), Fig. 3. These calibrated intervals feed position sizing and the trading invalidation rule.

![Fig 3. Coverage calibration](reports/figures/03_coverage_calibration.png)

## Where the skill comes from, and where it fails

A feature ablation (Fig. 4) shows that removing all fundamentals forecasts costs +9.45 MAE, while removing the (redundant) constructed residual-load feature
costs essentially nothing — the skill is genuine fundamentals, not price autocorrelation. SHAP attributes most of the model's output to residual load;
the apparent tension with the ablation is resolved by feature redundancy (SHAP measures model reliance on a fixed model; ablation measures marginal value after
refit). By regime, MAE is 11.9 on normal hours, 15.6 on negative-price hours, and 54.4 on spikes (intrinsically hard); the worst hours are the 17:00–19:00
evening peak.

![Fig 4. Feature ablation](reports/figures/04_ablation.png)

## From forecast to a prompt-curve view

The hourly fair value is aggregated to a daily baseload view and compared to a forward price. With no free forward-curve feed available, the forward enters as
a **swappable input**; we proxy it two ways. Against a *backward-looking* anchor (trailing realised baseload) the signal posts a 95% hit rate — reported only as
a **diagnostic** that such an anchor is not a tradable forward, since any good forecaster beats it trivially. Against a *forward-looking* consensus (a market
that prices like the Ridge model), the incremental nonlinear edge is **real but modest**: 61.9% hit rate over 134 trades — ≈2.7σ above chance by a permutation
test (Fig. 5) — at +EUR 4.18 per trade after a conservative EUR 0.50/MWh cost (the edge survives costs up to ~EUR 4.7/MWh). A confidence filter that trades
only on narrow-interval days roughly **doubles the risk-adjusted information ratio** (0.22 → 0.43), improving returns by cutting P&L volatility rather than raising the hit rate.

![Fig 5. Trading significance (permutation test)](reports/figures/05_trading_significance.png)

**Use / invalidation.** Trade only when fair value beats the forward by more than the estimated risk premium plus a noise buffer sized from the model's own forecast-error volatility; size by the calibrated interval width; stand down when the edge compresses, when the predicted interval widens, in spike-prone evening-peak hours, or when the residual-load forecast revises against the view.

This is a **mechanism demonstration, not deployable alpha**. A real P&L requires a licensed forward series — the code accepts one as a drop-in — and against a
truly efficient market the edge would be smaller still. The model's primary value is accurate fair-value estimation with calibrated uncertainty.

## LLM component

A provider-agnostic, schema-constrained LLM (Gemini by default) converts free-text outage/news into validated structured supply-disruption records (asset, fuel, capacity, window, price direction, confidence). On a synthetic sample it extracted five events from six items and correctly emitted nothing for a "no change" non-event — i.e. it does not hallucinate a record per line. Every
call is logged; the stage degrades gracefully with no API key, and tests use a mocked client so the suite passes offline. It is presented as a capability —
the route from unstructured text to a model-ready feature — rather than wired into the historical backtest, for which no time-aligned news history exists.

## Engineering and reproducibility

One CLI (`run_pipeline.py --stage ...`) drives every stage; 
`make install && make run` reproduces the pipeline end to end from a clean clone, 
`make test` runs 45 unit tests (including the leakage guard), and CI enforces ruff + pytest.
All results are persisted; `submission.csv` ships hourly forecasts with the calibrated 90% interval.

## Limitations

No free forward-curve feed (trading is a mechanism demo, as above); the LLM component is demonstrated on synthetic text; spike-hour accuracy is limited by design (a median objective does not chase tails). Each is a deliberate,
documented scope choice rather than an oversight.
