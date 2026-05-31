# Logbook — Day 4: Trading Translation & the LLM Component

**Project:** European Power Fair Value — German Day-Ahead Forecast & Prompt-Curve View
**Author:** Pietro Cuoco
**Scope of Day 4:** Turn the forecast into a tradable prompt-curve view
(Requirement 3), and build the programmatic LLM component (Requirement 4). This
is the day where intellectual honesty matters most: the headline lesson is that
**forecast accuracy is not the same thing as tradeable alpha**, and the work is
built to show that distinction rather than hide it.

---

## Part A — From fair value to a tradable view (Requirement 3)

### A.1 The bridge

The model outputs an hourly day-ahead fair value. A desk trades the *prompt
curve* — the near-dated forward (front-week baseload). We bridge the two in four
steps, all point-in-time:

1. **Aggregate** hourly fair values into daily products: `baseload` = mean of
   the 24 hours, `peakload` = mean of peak hours (08–20 CET).
2. **Compare** the model's baseload fair value to a forward price to form an
   edge: `edge = fair_value − forward`.
3. **Clear two hurdles** before acting: an empirically estimated **risk premium**
   (the systematic forward-vs-realized wedge) and a **noise buffer** sized from
   the model's own recent daily forecast-error volatility. We only trade when the
   edge beats both — i.e. when the signal is larger than our own uncertainty.
4. **Backtest** a long/short/flat rule with a transaction cost.

Every input to a day-D decision is built from data strictly before D (shifted,
trailing windows), and the position settles on realised D afterward — exactly
how a real trade resolves. No look-ahead.

### A.2 The forward-price problem, and the swappable-input design

We have real day-ahead prices but **no free forward feed** (EEX forward
settlements are licensed). The "forward" is therefore passed in as an argument,
so a real series can be dropped in with one line and the whole backtest re-runs
for real. By default we substitute a proxy — and *which* proxy we choose turns
out to be the entire story.

### A.3 The honest three-tier result

| Backtest | Trades | Hit rate | Avg P&L/trade (EUR/MWh) | Info ratio/trade |
|---|---|---|---|---|
| Anchor (trailing realised — strawman) | 409 | 95.1% | +29.87 | +1.13 |
| Consensus (vs Ridge — forward-looking) | 134 | 61.9% | +4.18 | +0.22 |
| Consensus + confidence filter | 59 | 59.3% | +4.58 | +0.43 |

**The 95% was a red flag, not a triumph — and chasing it down is the most
important thing on this day.** It is *not* code leakage; the timing is clean. It
is the consequence of a **strawman forward proxy**. Using "last week's average
realised baseload" as the price you transact at is unrealistic: no such forward
exists, and because our model is an accurate forecaster (`model ≈ realised`),
the quantity we trade on (`model − proxy`) is almost identical to the quantity
we get paid on (`realised − proxy`). We were effectively using the forecast to
predict itself. The estimated risk premium was +0.07 EUR/MWh and the
"always-long vs proxy" benchmark only +6.2 total, confirming the anchor has no
directional content of its own — so the entire P&L was forecast skill beating a
benchmark no real market would price.

**The realistic test** replaces the anchor with a *forward-looking consensus*:
assume the market prices like our simple linear **Ridge** model (which uses the
same public forecasts), so LightGBM only earns on its *incremental, nonlinear*
edge. That collapses the result to **61.9% hit rate, +4.18 EUR/MWh per trade** —
modest, but real: with 134 trades, 61.9% is ≈2.7σ above a coin flip
(binomial p ≈ 0.006). And it is *selective* — only 134 of 631 days traded, i.e.
the model bets only when it strongly disagrees with the consensus.

### A.4 The invalidation rule, and what it actually buys

The confidence filter only acts on **high-confidence days** — where the model's
*predicted* interval (q95−q05) is narrow relative to its recent norm — and stands
down otherwise. It is strictly point-in-time (predicted width is known at the
gate; the cutoff is a trailing, shifted median).

Its effect is subtle and worth stating precisely: trades roughly halve
(134→59), hit rate barely moves (61.9%→59.3%), but the **info ratio doubles
(0.22→0.43)**. Backing out the numbers, per-trade P&L volatility fell from ≈€19
to ≈€11. So **the invalidation rule works by cutting risk, not by being more
often right** — which is exactly what a width-based confidence signal *should*
do: a narrow predicted interval means low expected dispersion, hence less
volatile P&L. Honest caveat: on the 59-trade subset the 59.3% hit rate is only
≈1.4σ above chance, so the *directional* edge there is not independently
significant; the gain is risk-adjustment.

### A.5 The principle: forecast accuracy ≠ tradeable alpha

You are not paid for being accurate; you are paid for being **more right than
the market, net of costs and the risk premium**. A backward-looking anchor
ignores public forecasts, so any competent forecaster beats it trivially — that
is the 95%. A real forward already embeds those forecasts, so the genuine edge is
only the *residual* the market misses (a fundamental shift the forward hasn't yet
priced). That residual is small and noisy, which is why the consensus edge is
+€4.18, not +€29.87. Reporting the collapse openly is the centrepiece of the
trading section.

### A.6 Why there is no "real" P&L (a data limit, not a conceptual one)

A real backtest needs two real tradeable prices: one to **enter** at and one to
**settle** at. For a stock, both are the same freely-available price series, so
the entry price is baked into the data. For our DA-to-curve view the two are
*different instruments*: we have the settlement (the DA auction price, free) but
not the entry (the forward, licensed). To take a *financial* position on the DA
level you trade the future that settles against it — and that future's price is
precisely what we lack. **With a real forward series the swappable input would
produce a fully real P&L, unchanged code.** The limitation is free-data scope,
not method.

### A.7 Use / invalidation guidance (for the report)

*Use:* hourly fair value → daily baseload/peak → front-week view; trade only when
fair value beats the forward by more than risk premium + a noise buffer; size by
the CQR interval width. *Invalidate / stand down when:* the edge falls below
threshold; the predicted interval widens beyond its norm (empirically where
risk-adjusted returns degrade); the day is in a low-trust regime (spike-prone
evening-peak, MAE ~€54 vs ~€12); or the residual-load forecast revises against
the view before the gate. *Claim:* the mechanism plus a statistically significant
but modest incremental edge — an upper bound on deployable alpha. Primary value
is accurate fair value with calibrated uncertainty.

### A.8 Cost assumption

A flat **€0.50/MWh per position** (`cost` parameter). This is conservative —
liquid EEX front baseload spreads run ~€0.05–0.20/MWh — so it does not flatter
results. The consensus +€4.18 is *after* this cost; gross ≈ +€4.68, so the edge
survives transaction costs up to ~€4.68/MWh and is not a cost artifact. A
cost-sensitivity sweep (0 / 0.5 / 1.0 / 2.0) will go in the robustness section.

---

## Part B — The programmatic LLM component (Requirement 4)

### B.1 What it does

Power-relevant information often arrives as *text* — outage notices, REMIT
messages, news — that isn't in the numeric feeds. The component reads such free
text and returns **typed, schema-validated records**: `{asset, fuel_type,
capacity_mw, start, end, direction (bullish/bearish/neutral), confidence}`. The
`direction` field classifies price impact in the project's merit-order frame
(supply lost / demand up → bullish; capacity returns / extra wind → bearish), so
the output speaks the same language as the price model.

### B.2 Design choices (production-grade, not a toy)

* **Strict schema** — a Pydantic model is the response schema, so output is
  validated JSON, not prose parsed by fragile heuristics.
* **Provider-agnostic & mockable** — Gemini (default, `gemini-2.5-flash` via the
  unified `google-genai` SDK) or Groq, chosen in config; the network call is
  isolated so tests inject a stub. The whole suite and CI therefore pass **with
  no API key**.
* **Audited** — every call logs prompt, raw response, status, and parsed result
  to `logs/llm/`.
* **Graceful degradation** — no key or no network → it returns zero events and
  logs the reason, instead of crashing.

### B.3 Validation on the synthetic sample

Six synthetic items (fictional plants/dates — reproducible and copyright-free)
produced **five events**:

| Input | Extracted | Correct? |
|---|---|---|
| Grohnde-Nord nuclear offline, 1400 MW | bullish, nuclear, 1400 MW | ✓ |
| Cold snap lifts demand | bullish, demand | ✓ |
| Weisweiler-3 lignite returns, 600 MW | bearish, lignite, 600 MW | ✓ |
| 400 kV line works limit wind export | bullish, transmission | ✓ (ambiguous) |
| Offshore wind above norms | bearish, wind | ✓ |
| "No change to Irsching gas schedule" | (nothing) | ✓ |

The standout is the **sixth item: a non-event correctly produced no record** —
the negative control passes, so the extractor is not blindly emitting a row per
line. Directions all map to merit-order logic; stated capacities (1400, 600 MW)
are captured and the rest correctly left null. The transmission item is the one
genuine ambiguity (bullish for the whole DE-LU price, arguably bearish locally) —
exactly where the `confidence` field and human review matter.

### B.4 Honest scope

We **demonstrate the capability** (text → validated, logged features,
reproducibly) and **describe** how it would feed the model in production as a
time-aligned "supply-disruption" feature gated at the forecast cutoff. We do
**not** bolt it onto the historical backtest, because we have no timestamped
historical text feed and a fabricated feature would either be made-up or leak.
Same discipline as the trading section: claim only what the data supports.

---

## Incidents and lessons

* **Caught our own too-good backtest.** A 95% hit rate triggered a stop-and-
  investigate; the diagnosis (strawman proxy, not leakage) reshaped the whole
  trading section. Catching this is more valuable than a clean profit curve.
* **Secret hygiene.** An API key briefly landed in `.env.example` (a committed
  template) instead of `.env` (gitignored); GitHub push protection blocked it,
  the key was revoked and rotated, and the commit was rewritten clean. The
  safety net worked as intended. Lesson: secrets live only in `.env`.

## Test status

**45 tests passing**, lint clean. New: `test_trade.py` (aggregation, point-in-
time proxy, P&L logic, perfect-foresight sanity, confidence filter) and
`test_llm.py` (schema validation, mocked extraction, graceful no-key, audit
logging) — all offline.

## Open items → Day 5 (assembly)

Figures (`plots.py` + `figures` stage), the 1–3 page report, README headline,
`submission.csv`, clean-clone reproducibility (one documented command), and a
trading-desk-oriented dashboard summarising the work.
