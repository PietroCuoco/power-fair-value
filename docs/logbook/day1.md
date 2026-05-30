# Logbook — Day 1: Data Ingestion & Quality Assurance

**Project:** European Power Fair Value — German Day-Ahead Forecast & Prompt-Curve View
**Author:** Pietro Cuoco
**Scope of Day 1:** Establish a reproducible, validated data foundation for the
German (DE/LU) day-ahead power market: one-command ingestion of hourly prices
and fundamental drivers, plus power-specific quality assurance.

---

## 1. Objective

Deliver a clean, point-in-time-aware hourly dataset for the DE/LU bidding zone
covering day-ahead prices and the fundamental drivers that set them, with
documented provenance and automated quality checks. The dataset must be
reproducible from a single command and trustworthy enough that every downstream
forecasting and trading result rests on solid ground.

---

## 2. Key decisions and rationale

| Decision | Choice | Rationale |
|---|---|---|
| Market | Germany (DE/LU bidding zone) | Deepest liquidity, highest renewable penetration, most frequent negative prices — the richest setting to demonstrate fundamentals-driven pricing and domain-correct data handling. |
| Forecast target | Next-day hourly day-ahead price (Option A) | Lets the model capture daily price *shape* (morning ramp, midday solar trough, evening peak); 24 forecasts/day give validation statistical power. Bridged to a front-week baseload view for the trading section. |
| Primary data source | SMARD.de (Bundesnetzagentur) | Token-free, CC BY 4.0, stable JSON API. Natively carries day-ahead forecast series, which are exactly what a leakage-free design needs. Not blocked by registration latency. |
| Secondary source | ENTSO-E Transparency | Requested in parallel (token granted within ~3 working days) for cross-source QA reconciliation; the project is never blocked on it. |
| Price floor sanity anchor | EPEX technical floor −500 €/MWh | Used as an external validation anchor for the price series. |
| Residual-load forecast | Constructed: load_fc − (wind+PV)_fc | The merit-order driver, built from validated SMARD forecast inputs rather than an unverified single series (see Incident 3). |

**Point-in-time philosophy (the spine of the whole project):** the day-ahead
auction for delivery day D closes at 12:00 Berlin on D−1. Every future feature
must respect that information boundary. Day 1 secured the inputs needed to honour
it: SMARD day-ahead *forecast* series (load, wind, solar) rather than realised
actuals, which would constitute look-ahead leakage.

---

## 3. Data sources and provenance

| Logical series | SMARD filter | Region | Kind | Use |
|---|---|---|---|---|
| price_da | 4169 | DE-LU | actual | Target (day-ahead clearing price) |
| load_actual | 410 | DE | actual | EDA, lagged feature, oracle ablation |
| residual_load_actual | 4359 | DE | actual | Cross-series consistency check |
| gen_wind_onshore_actual | 4067 | DE | actual | Oracle ablation |
| gen_wind_offshore_actual | 1225 | DE | actual | Oracle ablation |
| gen_pv_actual | 4068 | DE | actual | Oracle ablation |
| fc_gen_total | 122 | DE | forecast | Feature |
| fc_gen_wind_pv | 5097 | DE | forecast | Key renewable driver; residual-load construction |
| fc_gen_wind_onshore | 123 | DE | forecast | Feature |
| fc_gen_wind_offshore | 3791 | DE | forecast | Feature |
| fc_gen_pv | 125 | DE | forecast | Feature |
| fc_load_total | 411 | DE | forecast | Day-ahead load forecast (validated, see Incident 3) |

Filter IDs were verified against the SMARD OpenAPI specification
(`bundesAPI/smard-api`) rather than memory. Endpoint shape: a per-series weekly
index of UTC millisecond timestamps, then weekly JSON blocks of `[ms, value]`
pairs; `null` → NaN. All timestamps stored in UTC internally.

---

## 4. Files created (Day 1)

**Scaffold:**
- `README.md` — storefront: quickstart, data-source table, point-in-time note.
- `requirements.txt`, `requirements.lock.txt` — dependencies and frozen versions.
- `.gitignore`, `.env.example` — hygiene and secret template.
- `pyproject.toml` — package metadata, ruff/black/pytest configuration.
- `config/config.yaml` — single source of truth (market, dates, paths, thresholds).
- `src/power_fv/__init__.py` — package marker.
- `tests/test_smoke.py` — trivial test so CI is green from the first commit.
- `.github/workflows/ci.yml` — ruff + pytest on every push.
- `figures/.gitkeep` — preserves the committed-figures directory.

**Ingestion and QA:**
- `src/power_fv/config.py` — config + `.env` loader, path resolution.
- `src/power_fv/ingest.py` — SMARD client, series registry, range fetch with
  weekly-block stitching, and a `discover` probe for confirming filter IDs.
- `src/power_fv/qa.py` — power-specific checks and `qa_report` writer.
- `tests/test_qa.py` — 10 tests covering QA logic and ingest parsers.
- `scripts/run_pipeline.py` — entrypoint wiring the ingest and qa stages.

**Generated locally (gitignored):** `data/raw/smard_raw.parquet` (29,857 × 12),
`data/processed/dataset_clean.parquet`, `data/processed/qa_report.{md,json}`.

---

## 5. Data validation results

Dataset: **29,857 hourly rows**, 2023-01-01 → 2026-05-24, **0 missing hours**
(UTC grid). Hour count is arithmetically consistent (~3 years 5 months).

**DST transitions (after the boundary fix, Incident 2):**
- 4 spring-forward (23-hour) days: 2023-03-26, 2024-03-31, 2025-03-30, 2026-03-29.
- 3 fall-back (25-hour) days: 2023-10-29, 2024-10-27, 2025-10-26.
- These match the German calendar exactly.

**Price diagnostics:**
- Negative-price hours: **1,562 (5.23%)**, minimum **−500 €/MWh** — exactly the
  EPEX technical floor, a near-definitive confirmation of genuine day-ahead
  auction data. Negatives preserved (economically real), never deleted.
- Spikes (|robust MAD z| > 6): **109 flagged and preserved**; top |values|
  936, 820, 819, 805, 674 €/MWh — plausible for the post-2022 period.

**Cross-series consistency:** residual_load_actual reconciles to
(load − wind_on − wind_off − PV) with median and p95 absolute difference of
**0.0 MW**. Confirms the actual series are internally consistent; noted that
residual_load_actual is therefore collinear with its components and will not be
used as an independent feature.

**Forecast coverage:** all forecast columns have 0 NaN across the full window —
dense day-ahead forecast availability for point-in-time features.

**Forecast-load ID validation (Incident 3):** filter 411 confirmed as total load
forecast (mean 53,229 MW, consistent with the Bundesnetzagentur Q1-2026 Netzlast
figure). Filter 413 rejected (mean 91.8 MW vs. an expected ~34,000 MW residual
load) — it is some near-zero signed quantity, not residual load.

---

## 6. Incidents and resolutions

**Incident 1 — ENTSO-E token latency.** The Transparency Platform now grants API
access only after an email request, within ~3 working days — incompatible with a
5-day build. *Resolution:* adopted SMARD.de as the primary source (token-free,
and arguably better suited because it carries the forecast series natively);
ENTSO-E demoted to a parallel cross-check. No quality loss.

**Incident 2 — DST boundary false positive.** The first QA report flagged 5
spring-forward days, including 2023-01-01, which is not a DST date. Cause: the
data window starts at 00:00 UTC = 01:00 Berlin in winter, so the first local day
has only 23 hours — a partial-boundary artifact. *Resolution:* `check_dst` now
excludes the first and last (partial) local days; a regression test
(`test_dst_ignores_partial_boundary_day`) prevents recurrence. Count corrected to
4 / 3, matching the calendar.

**Incident 3 — unvalidated forecast-load filter.** The SMARD spec did not
unambiguously expose a day-ahead *load* forecast ID. A `discover` probe found two
live candidates (411, 413). Magnitude validation against the Bundesnetzagentur
residual-load figure (~34 GW) confirmed 411 as total load and rejected 413
(~0.09 GW). *Resolution:* dropped 413; the residual-load forecast is constructed
as load_fc − (wind+PV)_fc from validated inputs.

**Incident 4 — gap-fill partial-fill bug.** A unit test caught that pandas'
`interpolate(limit=...)` partially fills long gaps (3 of a 6-hour gap) rather than
skipping them. *Resolution:* rewrote `fill_short_gaps` to be run-length aware —
an interior NaN run is filled only if its full length is within the limit.

**Incident 5 — file misplacement and CI failure.** During a file update,
`config.py` was deleted and `test_qa.py` landed in `src/` instead of `tests/`,
breaking imports and CI. *Resolution:* restored layout via explicit commands and
adopted an open-file-and-paste workflow for future edits. CI green thereafter.

---

## 7. Reproducibility

```
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -e . && pip install -r requirements.txt
python scripts/run_pipeline.py --stage discover   # confirm live filter IDs
python scripts/run_pipeline.py --stage all         # ingest + qa
ruff check . && pytest -q                          # 11 passing tests
```

**Commit history (Day 1):**
1. `chore: project scaffold (config, CI, tooling, package skeleton)`
2. `feat: SMARD ingest + power-specific QA with tests`
3. `fix: exclude partial boundary days from DST check; add forecast load series`
4. `data: reject unvalidated filter 413; residual-load forecast from 411-5097`

End-of-day test status: **11 passed** (1 smoke + 10 QA/ingest-parser). CI green.

---

## 8. Open items carried to Day 2

- Validate the constructed `residual_load_fc` mean against the ~34 GW benchmark
  (expected; to confirm when the features stage runs).
- Build the point-in-time feature layer and the mechanical leakage guard.
- Implement baselines (seasonal-naïve, regularised linear) and the
  expanding-window walk-forward splitter.
- Fold ENTSO-E into a cross-source QA reconciliation if/when the token arrives.
