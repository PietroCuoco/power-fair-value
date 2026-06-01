"""End-to-end pipeline entrypoint.

Stages (wired incrementally over Days 1-5):
    ingest   : pull all SMARD series -> data/raw/smard_raw.parquet
    qa       : run QA checks -> data/processed/dataset_clean.parquet + reports
    discover : probe candidate filter ids for the day-ahead load forecast
    all      : ingest + qa

Usage:
    python scripts/run_pipeline.py --stage ingest
    python scripts/run_pipeline.py --stage qa
    python scripts/run_pipeline.py --stage discover
    python scripts/run_pipeline.py --stage all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from power_fv import analysis as anl
from power_fv import features as feat
from power_fv import ingest, plots, qa
from power_fv import llm as llm_mod
from power_fv import trade as trd
from power_fv import validate as val
from power_fv.config import load_config
from power_fv.models import RidgeModel, SeasonalNaive


def _stage_ingest(cfg: dict) -> pd.DataFrame:
    print("[ingest] pulling SMARD series ...")
    return ingest.build_dataset(cfg, save=True)


def _stage_qa(cfg: dict) -> None:
    raw_path = Path(cfg["data"]["raw_dir"]) / "smard_raw.parquet"
    if not raw_path.exists():
        raise SystemExit(f"[qa] {raw_path} not found - run --stage ingest first.")
    df = pd.read_parquet(raw_path)
    clean, report = qa.run_qa(df, cfg)
    out_dir = Path(cfg["data"]["processed_dir"])
    clean.to_parquet(out_dir / "dataset_clean.parquet")
    reports_dir = out_dir.parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    qa.write_reports(report, reports_dir)
    print(f"[qa] clean dataset -> {out_dir}; qa_report -> {reports_dir}")


def _stage_features(cfg: dict) -> None:
    out_dir = Path(cfg["data"]["processed_dir"])
    clean_path = out_dir / "dataset_clean.parquet"
    if not clean_path.exists():
        raise SystemExit(f"[features] {clean_path} not found - run --stage qa first.")
    df = pd.read_parquet(clean_path)
    X, y, meta = feat.make_modeling_frame(df)

    # Enforce the leakage guard as a pipeline gate (recent 120 days, all hours).
    sample = X.index[-120 * 24:]
    feat.assert_no_leakage(meta, sample)

    # Magnitude validation of the constructed residual-load forecast.
    rl_mean = (df["fc_load_total"] - df["fc_gen_wind_pv"]).mean()
    print(f"[features] leakage guard passed on {len(sample):,} timestamps")
    print(f"[features] residual_load_fc mean = {rl_mean:,.0f} MW (expect ~34,000)")
    print(f"[features] X shape = {X.shape}, {len(meta)} features")

    X.to_parquet(out_dir / "features_X.parquet")
    y.to_frame("price_da").to_parquet(out_dir / "target_y.parquet")
    print(f"[features] saved features_X.parquet and target_y.parquet to {out_dir}")


def _stage_baselines(cfg: dict) -> None:
    out_dir = Path(cfg["data"]["processed_dir"])
    xp, yp = out_dir / "features_X.parquet", out_dir / "target_y.parquet"
    if not (xp.exists() and yp.exists()):
        raise SystemExit("[baselines] features not found - run --stage features first.")
    X = pd.read_parquet(xp)
    y = pd.read_parquet(yp)["price_da"]

    wf = cfg["model"]["walk_forward"]
    splitter = val.WalkForwardSplitter(wf["initial_train_days"], wf["step_days"])

    naive_pred, actual = val.run_backtest(X, y, SeasonalNaive(), splitter)
    ridge_pred, _ = val.run_backtest(X, y, RidgeModel(), splitter)

    naive_mae = val.mae(actual, naive_pred)
    ridge_mae = val.mae(actual, ridge_pred)

    print(f"[baselines] out-of-sample hours: {len(actual):,}")
    print(
        f"[baselines] seasonal-naive  MAE {naive_mae:6.2f}  "
        f"RMSE {val.rmse(actual, naive_pred):6.2f}"
    )
    print(
        f"[baselines] ridge           MAE {ridge_mae:6.2f}  "
        f"RMSE {val.rmse(actual, ridge_pred):6.2f}"
    )
    print(f"[baselines] ridge skill vs naive: {val.skill_score(ridge_mae, naive_mae):+.1%}")

    preds = pd.DataFrame(
        {"actual": actual, "seasonal_naive": naive_pred, "ridge": ridge_pred}
    )
    preds.to_parquet(out_dir / "preds_baselines.parquet")
    print(f"[baselines] saved preds_baselines.parquet to {out_dir}")


def _stage_model(cfg: dict) -> None:
    out_dir = Path(cfg["data"]["processed_dir"])
    xp, yp = out_dir / "features_X.parquet", out_dir / "target_y.parquet"
    bp = out_dir / "preds_baselines.parquet"
    if not (xp.exists() and yp.exists() and bp.exists()):
        raise SystemExit("[model] need features and baselines - run those stages first.")
    X = pd.read_parquet(xp)
    y = pd.read_parquet(yp)["price_da"]
    base = pd.read_parquet(bp)

    wf = cfg["model"]["walk_forward"]
    splitter = val.WalkForwardSplitter(wf["initial_train_days"], wf["step_days"])
    quantiles = tuple(cfg["model"].get("quantiles", [0.05, 0.5, 0.95]))

    print("[model] running LightGBM quantile walk-forward (this takes a few minutes) ...")
    preds, actual = val.run_quantile_backtest(X, y, splitter, quantiles=quantiles)
    base = base.reindex(actual.index)
    point = preds["q50"]

    mae_l, rmse_l = val.mae(actual, point), val.rmse(actual, point)
    mae_n = val.mae(actual, base["seasonal_naive"])
    mae_r = val.mae(actual, base["ridge"])

    e_lgbm = actual - point
    dm_n_stat, dm_n_p = val.diebold_mariano(actual - base["seasonal_naive"], e_lgbm)
    dm_r_stat, dm_r_p = val.diebold_mariano(actual - base["ridge"], e_lgbm)
    cov = val.interval_coverage(actual, preds["q05"], preds["q95"])

    print(f"[model] out-of-sample hours: {len(actual):,}")
    print(f"[model] LightGBM   MAE {mae_l:6.2f}  RMSE {rmse_l:6.2f}")
    print(f"[model] skill vs naive: {val.skill_score(mae_l, mae_n):+.1%}  "
          f"| skill vs ridge: {val.skill_score(mae_l, mae_r):+.1%}")
    print(f"[model] DM vs naive: stat {dm_n_stat:+.2f}, p {dm_n_p:.2e}  "
          f"(positive stat => LightGBM more accurate)")
    print(f"[model] DM vs ridge: stat {dm_r_stat:+.2f}, p {dm_r_p:.2e}")
    print(f"[model] 90% interval coverage: {cov:.1%}  (target 90%)")

    out = preds.copy()
    out["actual"] = actual
    out.to_parquet(out_dir / "preds_model.parquet")
    print(f"[model] saved preds_model.parquet to {out_dir}")


def _stage_conformal(cfg: dict) -> None:
    out_dir = Path(cfg["data"]["processed_dir"])
    xp, yp = out_dir / "features_X.parquet", out_dir / "target_y.parquet"
    mp = out_dir / "preds_model.parquet"
    if not (xp.exists() and yp.exists()):
        raise SystemExit("[conformal] need features - run --stage features first.")
    X = pd.read_parquet(xp)
    y = pd.read_parquet(yp)["price_da"]

    wf = cfg["model"]["walk_forward"]
    splitter = val.WalkForwardSplitter(wf["initial_train_days"], wf["step_days"])

    print("[conformal] running CQR walk-forward (this takes a few minutes) ...")
    preds, actual = val.run_conformal_quantile_backtest(X, y, splitter, alpha=0.10)
    cov_after = val.interval_coverage(actual, preds["q_lo"], preds["q_hi"])
    width_after = float((preds["q_hi"] - preds["q_lo"]).mean())

    if mp.exists():
        raw = pd.read_parquet(mp).reindex(actual.index)
        cov_before = val.interval_coverage(actual, raw["q05"], raw["q95"])
        width_before = float((raw["q95"] - raw["q05"]).mean())
        print(f"[conformal] coverage before (raw):  {cov_before:.1%}, width {width_before:.1f}")
    print(
        f"[conformal] coverage after (CQR):   {cov_after:.1%}, "
        f"width {width_after:.1f}  (target 90%)"
    )

    out = preds.copy()
    out["actual"] = actual
    out.to_parquet(out_dir / "preds_conformal.parquet")
    print(f"[conformal] saved preds_conformal.parquet to {out_dir}")


def _stage_ablation(cfg: dict) -> None:
    out_dir = Path(cfg["data"]["processed_dir"])
    xp, yp = out_dir / "features_X.parquet", out_dir / "target_y.parquet"
    if not (xp.exists() and yp.exists()):
        raise SystemExit("[ablation] need features - run --stage features first.")
    X = pd.read_parquet(xp)
    y = pd.read_parquet(yp)["price_da"]

    wf = cfg["model"]["walk_forward"]
    splitter = val.WalkForwardSplitter(wf["initial_train_days"], wf["step_days"])

    print("[ablation] retraining LightGBM under feature ablations (a few minutes) ...")
    res = anl.forecast_feature_ablation(X, y, splitter)
    full = res["full"]
    print(f"[ablation] full feature set       MAE {full:6.2f}")
    for name, value in res.items():
        if name == "full":
            continue
        print(f"[ablation] {name:24s} MAE {value:6.2f}  (+{value - full:5.2f} vs full)")
    pd.Series(res, name="mae").to_frame().to_parquet(out_dir / "ablation.parquet")
    print(f"[ablation] saved ablation.parquet to {out_dir}")


def _stage_breakdown(cfg: dict) -> None:
    out_dir = Path(cfg["data"]["processed_dir"])
    mp = out_dir / "preds_model.parquet"
    if not mp.exists():
        raise SystemExit("[breakdown] need preds_model - run --stage model first.")
    preds = pd.read_parquet(mp)
    table, by_hour, spike_level = anl.error_breakdown(preds["actual"], preds["q50"])

    print(f"[breakdown] spike threshold (95th pct): {spike_level:.1f} EUR/MWh")
    print("[breakdown] MAE by regime:")
    for regime, row in table.iterrows():
        print(f"[breakdown]   {regime:9s} MAE {row['mae']:6.2f}  (n={int(row['n']):,})")
    worst = by_hour.sort_values(ascending=False).head(3)
    print(f"[breakdown] worst hours (local): {[int(h) for h in worst.index]} "
          f"with MAE {[round(float(v), 1) for v in worst.values]}")
    by_hour.to_frame().to_parquet(out_dir / "mae_by_hour.parquet")
    table.to_parquet(out_dir / "mae_by_regime.parquet")
    print(f"[breakdown] saved mae_by_hour.parquet and mae_by_regime.parquet to {out_dir}")


def _stage_shap(cfg: dict) -> None:
    out_dir = Path(cfg["data"]["processed_dir"])
    xp, yp = out_dir / "features_X.parquet", out_dir / "target_y.parquet"
    if not (xp.exists() and yp.exists()):
        raise SystemExit("[shap] need features - run --stage features first.")
    X = pd.read_parquet(xp)
    y = pd.read_parquet(yp)["price_da"]

    print("[shap] fitting model and computing SHAP attributions ...")
    imp = anl.shap_importance(X, y)
    print("[shap] top features by mean |SHAP| (EUR/MWh contribution to prediction):")
    for name, value in imp.head(12).items():
        print(f"[shap]   {name:26s} {value:7.3f}")
    imp.to_frame("mean_abs_shap").to_parquet(out_dir / "shap_importance.parquet")
    print(f"[shap] saved shap_importance.parquet to {out_dir}")


def _stage_trade(cfg: dict) -> None:
    out_dir = Path(cfg["data"]["processed_dir"])
    mp = out_dir / "preds_model.parquet"
    bp = out_dir / "preds_baselines.parquet"
    if not (mp.exists() and bp.exists()):
        raise SystemExit("[trade] need preds_model and preds_baselines - run those stages first.")
    preds = pd.read_parquet(mp)
    base = pd.read_parquet(bp)
    peak = cfg["trading"]["peak_hours"]

    model_daily = trd.to_daily_products(preds["q50"], peak)["baseload"]
    real_daily = trd.to_daily_products(preds["actual"], peak)["baseload"]
    ridge_daily = trd.to_daily_products(base["ridge"].reindex(preds.index), peak)["baseload"]

    threshold = trd.rolling_threshold(model_daily, real_daily, window=30, k=1.0)

    def _run(proxy: pd.Series, label: str) -> pd.DataFrame:
        premium = trd.rolling_premium(real_daily, proxy, window=60)
        res = trd.backtest_signal(
            model_daily, real_daily, proxy, threshold, premium=premium, cost=0.5
        )
        s = trd.summarize_backtest(res)
        print(
            f"[trade] [{label}] trades {s['n_trades']:,} | hit rate {s['hit_rate']:.1%} | "
            f"avg P&L/trade {s['avg_pnl_per_trade']:+.2f} | "
            f"info ratio {s['info_ratio_per_trade']:+.2f}"
        )
        res.to_parquet(out_dir / f"trade_backtest_{label}.parquet")
        return res

    print("[trade] DIAGNOSTIC - naive backward-looking anchor (trailing realized baseload):")
    print("[trade]   an unrealistically weak 'forward'; a good forecaster beats it trivially.")
    _run(trd.forward_proxy(real_daily, window=7), "anchor")

    print("[trade] REALISTIC - forward-looking consensus (market prices like the Ridge model):")
    print("[trade]   measures only LightGBM's incremental, nonlinear edge over public info.")
    res_consensus = _run(ridge_daily, "consensus")

    # Invalidation rule: only act on high-confidence days. Confidence = the
    # model's own predicted interval is narrow relative to its recent typical
    # width. Point-in-time: predicted width is known at the gate; the cutoff is
    # a trailing, shifted median, so no realised outcome enters the decision.
    width = trd.to_daily_products(preds["q95"] - preds["q05"], peak)["baseload"]
    width_cut = width.shift(1).rolling(60, min_periods=60).median()
    high_conf = width < width_cut
    res_cf = trd.apply_confidence_filter(res_consensus, high_conf)
    s = trd.summarize_backtest(res_cf)
    print("[trade] INVALIDATION - consensus, high-confidence (narrow-interval) days only:")
    print(
        f"[trade] [consensus+confidence] trades {s['n_trades']:,} | hit rate {s['hit_rate']:.1%} | "
        f"avg P&L/trade {s['avg_pnl_per_trade']:+.2f} | "
        f"info ratio {s['info_ratio_per_trade']:+.2f}"
    )
    res_cf.to_parquet(out_dir / "trade_backtest_consensus_confident.parquet")
    print("[trade] note: a real forward curve would embed public forecasts; true alpha "
          "likely sits at or below the consensus result. This is a mechanism demonstration.")


def _stage_llm(cfg: dict) -> None:
    out_dir = Path(cfg["data"]["processed_dir"])
    sample_path = llm_mod.ROOT / "samples" / "outage_news.txt"
    texts = [ln for ln in sample_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"[llm] read {len(texts)} outage/news items; provider={cfg['llm']['provider']} "
          f"model={cfg['llm']['model']}")

    result = llm_mod.extract_from_config(texts, cfg)

    if not result.events:
        print("[llm] no events extracted - no API key set or provider returned none.")
        print("[llm] set GEMINI_API_KEY in .env to enable live extraction (logged to logs/llm/).")
    for e in result.events:
        cap = f"{e.capacity_mw:.0f} MW" if e.capacity_mw else "n/a"
        print(f"[llm]   {e.direction.value:7s} | {e.fuel_type:12s} | {cap:>8s} | {e.asset}")

    rows = [e.model_dump(mode="json") for e in result.events]
    df = pd.DataFrame(rows)
    if not df.empty:
        out = out_dir / "llm_events.parquet"
        df.to_parquet(out)
        print(f"[llm] saved {len(df)} structured events to {out}")


def _stage_submission(cfg: dict) -> None:
    out_dir = Path(cfg["data"]["processed_dir"])
    mp = out_dir / "preds_model.parquet"
    if not mp.exists():
        raise SystemExit("[submission] need preds_model - run --stage model first.")
    m = pd.read_parquet(mp).sort_index()
    sub = pd.DataFrame(
        {
            "id": m.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "y_pred": m["q50"].round(2).to_numpy(),
        }
    )
    cp = out_dir / "preds_conformal.parquet"
    if cp.exists():  # prefer the calibrated 90% interval
        c = pd.read_parquet(cp).reindex(m.index)
        sub["y_pred_lower_90"] = c["q_lo"].round(2).to_numpy()
        sub["y_pred_upper_90"] = c["q_hi"].round(2).to_numpy()
    else:
        sub["y_pred_lower_90"] = m["q05"].round(2).to_numpy()
        sub["y_pred_upper_90"] = m["q95"].round(2).to_numpy()
    dest = out_dir.parents[1] / "submission.csv"
    sub.to_csv(dest, index=False)
    print(f"[submission] wrote {len(sub):,} hourly forecasts (id, y_pred + 90% interval) to {dest}")


def _stage_figures(cfg: dict) -> None:
    out_dir = Path(cfg["data"]["processed_dir"])
    fig_dir = out_dir.parents[1] / "reports" / "figures"
    print(f"[figures] generating figures into {fig_dir} ...")
    made = plots.make_all_figures(out_dir, fig_dir)
    print(f"[figures] wrote {len(made)} figures.")


def _stage_discover() -> None:
    print("[discover] probing candidate forecast-load filter ids ...")
    print(ingest.discover().to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Power fair-value pipeline")
    parser.add_argument(
        "--stage",
        choices=[
            "ingest", "qa", "features", "baselines", "model",
            "conformal", "ablation", "breakdown", "shap", "trade", "llm",
            "submission", "figures", "discover", "all",
        ],
        default="all",
    )
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    if args.stage == "discover":
        _stage_discover()
        return

    cfg = load_config(args.config)
    if args.stage in ("ingest", "all"):
        _stage_ingest(cfg)
    if args.stage in ("qa", "all"):
        _stage_qa(cfg)
    if args.stage in ("features", "all"):
        _stage_features(cfg)
    if args.stage in ("baselines", "all"):
        _stage_baselines(cfg)
    if args.stage in ("model", "all"):
        _stage_model(cfg)
    if args.stage in ("conformal", "all"):
        _stage_conformal(cfg)
    if args.stage in ("ablation", "all"):
        _stage_ablation(cfg)
    if args.stage in ("breakdown", "all"):
        _stage_breakdown(cfg)
    if args.stage in ("shap", "all"):
        _stage_shap(cfg)
    if args.stage in ("trade", "all"):
        _stage_trade(cfg)
    if args.stage in ("llm", "all"):
        _stage_llm(cfg)
    if args.stage in ("submission", "all"):
        _stage_submission(cfg)
    if args.stage in ("figures", "all"):
        _stage_figures(cfg)


if __name__ == "__main__":
    main()
