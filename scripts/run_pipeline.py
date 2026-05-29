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


def _stage_discover() -> None:
    print("[discover] probing candidate forecast-load filter ids ...")
    print(ingest.discover().to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Power fair-value pipeline")
    parser.add_argument(
        "--stage",
        choices=["ingest", "qa", "discover", "all"],
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


if __name__ == "__main__":
    main()
