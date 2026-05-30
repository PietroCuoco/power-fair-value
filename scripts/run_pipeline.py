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
from power_fv import ingest, qa
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
    qa.write_reports(report, out_dir)
    print(f"[qa] clean dataset + qa_report written to {out_dir}")


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


def _stage_discover() -> None:
    print("[discover] probing candidate forecast-load filter ids ...")
    print(ingest.discover().to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Power fair-value pipeline")
    parser.add_argument(
        "--stage",
        choices=[
            "ingest", "qa", "features", "baselines", "model",
            "conformal", "ablation", "breakdown", "discover", "all",
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


if __name__ == "__main__":
    main()
