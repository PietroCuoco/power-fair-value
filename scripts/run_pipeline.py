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

from power_fv import features as feat
from power_fv import ingest, qa
from power_fv.config import load_config


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


def _stage_discover() -> None:
    print("[discover] probing candidate forecast-load filter ids ...")
    print(ingest.discover().to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Power fair-value pipeline")
    parser.add_argument(
        "--stage",
        choices=["ingest", "qa", "features", "discover", "all"],
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


if __name__ == "__main__":
    main()
