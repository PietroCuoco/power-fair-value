"""Figure generation for the report and README (curated set of five).

Each figure makes one distinct, defensible scientific claim, with honest axes and
labelled units (EUR/MWh, MW). Figures read the result parquets written by the
pipeline and are skipped gracefully if an input is absent.

  01 merit-order            - prices are convex in residual load (nonlinearity)
  02 model comparison       - LightGBM beats both baselines out-of-sample
  03 coverage calibration   - CQR restores ~nominal 90% interval coverage
  04 ablation               - skill comes from fundamentals, not autocorrelation
  05 trading significance   - the trading edge is real but modest (permutation)

Run via ``python scripts/run_pipeline.py --stage figures``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C_NAIVE = "#9aa0a6"
C_RIDGE = "#4c78a8"
C_LGBM = "#c0392b"
C_ACC = "#2c7d59"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


def _save(fig: plt.Figure, fig_dir: Path, name: str) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _abs_err(actual: pd.Series, pred: pd.Series) -> pd.Series:
    return (actual - pred).abs()


# --- 01. merit-order curve --------------------------------------------------

def fig_merit_order(proc: Path, fig_dir: Path) -> Path:
    X = pd.read_parquet(proc / "features_X.parquet")
    y = pd.read_parquet(proc / "target_y.parquet")["price_da"]
    rl = X["residual_load_fc"].reindex(y.index)
    d = pd.DataFrame({"rl": rl, "price": y}).dropna()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    hb = ax.hexbin(d["rl"], d["price"], gridsize=45, cmap="Blues", bins="log", mincnt=1)
    bins = pd.qcut(d["rl"], 30, duplicates="drop")
    med = d.groupby(bins, observed=True).agg(rl=("rl", "median"), price=("price", "median"))
    ax.plot(med["rl"], med["price"], color=C_LGBM, lw=2, label="binned median (merit order)")
    ax.set_xlabel("residual load forecast (MW)")
    ax.set_ylabel("day-ahead price (EUR/MWh)")
    ax.set_title("Merit-order structure: price rises convexly with residual load")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(hb, ax=ax, label="log10(count)")
    return _save(fig, fig_dir, "01_merit_order")


# --- 02. model comparison ---------------------------------------------------

def fig_model_comparison(proc: Path, fig_dir: Path) -> Path:
    b = pd.read_parquet(proc / "preds_baselines.parquet")
    m = pd.read_parquet(proc / "preds_model.parquet")
    a = b["actual"]
    rows = {
        "Seasonal-naive": (_abs_err(a, b["seasonal_naive"]).mean(),
                           np.sqrt(((a - b["seasonal_naive"]) ** 2).mean())),
        "Ridge": (_abs_err(a, b["ridge"]).mean(), np.sqrt(((a - b["ridge"]) ** 2).mean())),
        "LightGBM": (_abs_err(m["actual"], m["q50"]).mean(),
                     np.sqrt(((m["actual"] - m["q50"]) ** 2).mean())),
    }
    labels = list(rows)
    maes = [rows[k][0] for k in labels]
    rmses = [rows[k][1] for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - 0.2, maes, 0.4, label="MAE", color=C_RIDGE)
    ax.bar(x + 0.2, rmses, 0.4, label="RMSE", color=C_NAIVE)
    for xi, mae in zip(x, maes, strict=True):
        ax.text(xi - 0.2, mae + 0.5, f"{mae:.1f}", ha="center", fontsize=8)
    base = maes[0]
    for xi, mae in zip(x[1:], maes[1:], strict=True):
        ax.text(xi - 0.2, mae / 2, f"{1 - mae / base:+.0%}\nvs naive", ha="center",
                fontsize=7, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Forecast accuracy: LightGBM vs baselines (out-of-sample)")
    ax.legend(fontsize=8)
    return _save(fig, fig_dir, "02_model_comparison")


# --- 03. interval coverage calibration --------------------------------------

def fig_coverage(proc: Path, fig_dir: Path) -> Path:
    m = pd.read_parquet(proc / "preds_model.parquet")
    cov_raw = float(((m["actual"] >= m["q05"]) & (m["actual"] <= m["q95"])).mean())
    w_raw = float((m["q95"] - m["q05"]).mean())
    c = pd.read_parquet(proc / "preds_conformal.parquet")
    cov_cqr = float(((c["actual"] >= c["q_lo"]) & (c["actual"] <= c["q_hi"])).mean())
    w_cqr = float((c["q_hi"] - c["q_lo"]).mean())

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].bar(["raw QR", "CQR"], [cov_raw * 100, cov_cqr * 100], color=[C_NAIVE, C_ACC])
    axes[0].axhline(90, color="k", ls="--", lw=1, label="nominal 90%")
    for i, v in enumerate([cov_raw, cov_cqr]):
        axes[0].text(i, v * 100 + 1, f"{v:.1%}", ha="center", fontsize=9)
    axes[0].set_ylabel("empirical coverage (%)")
    axes[0].set_title("Coverage of the 90% interval")
    axes[0].legend(fontsize=8)
    axes[1].bar(["raw QR", "CQR"], [w_raw, w_cqr], color=[C_NAIVE, C_ACC])
    for i, v in enumerate([w_raw, w_cqr]):
        axes[1].text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=9)
    axes[1].set_ylabel("mean interval width (EUR/MWh)")
    axes[1].set_title("Cost of calibration: wider but honest")
    fig.suptitle("Conformal calibration restores coverage to ~nominal", y=1.02)
    return _save(fig, fig_dir, "03_coverage_calibration")


# --- 04. ablation -----------------------------------------------------------

def fig_ablation(proc: Path, fig_dir: Path) -> Path:
    s = pd.read_parquet(proc / "ablation.parquet").iloc[:, 0]
    order = ["full", "no_residual_load_fc", "no_forecasts"]
    s = s.reindex([k for k in order if k in s.index]).dropna()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(range(len(s)), s.values, color=[C_ACC, C_RIDGE, C_LGBM][: len(s)])
    full = s.get("full", s.iloc[0])
    for i, (name, v) in enumerate(s.items()):
        delta = "" if name == "full" else f"\n(+{v - full:.1f})"
        ax.text(i, v + 0.3, f"{v:.1f}{delta}", ha="center", fontsize=8)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.index, rotation=10)
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Feature ablation: skill comes from the fundamentals forecasts")
    return _save(fig, fig_dir, "04_ablation")


# --- 05. SIMULATION: permutation null for the trading hit rate --------------

def fig_trading_significance(proc: Path, fig_dir: Path, n_iter: int = 20000) -> Path:
    con = pd.read_parquet(proc / "trade_backtest_consensus.parquet")
    traded = con[con["position"] != 0]
    settle = traded["settle"].to_numpy()
    obs = float((traded["position"].to_numpy() * settle > 0).mean())
    n = len(settle)
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=(n_iter, n))
    null_hits = ((signs * settle) > 0).mean(axis=1)
    p_val = float((null_hits >= obs).mean())
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.hist(null_hits * 100, bins=40, color=C_NAIVE, alpha=0.8, label="random-direction null")
    ax.axvline(obs * 100, color=C_LGBM, lw=2, label=f"observed {obs:.1%}")
    ax.axvline(50, color="k", ls="--", lw=1)
    ax.set_xlabel("hit rate (%)")
    ax.set_ylabel("frequency")
    ax.set_title(f"Is the edge real? Permutation test, p = {p_val:.4f} ({n} trades)")
    ax.legend(fontsize=8)
    return _save(fig, fig_dir, "05_trading_significance")


_FIGURES = [
    fig_merit_order,
    fig_model_comparison,
    fig_coverage,
    fig_ablation,
    fig_trading_significance,
]


def make_all_figures(proc_dir: str | Path, fig_dir: str | Path) -> list[Path]:
    """Generate every figure whose inputs exist; skip the rest with a note."""
    proc, fig_dir = Path(proc_dir), Path(fig_dir)
    made: list[Path] = []
    for fn in _FIGURES:
        try:
            made.append(fn(proc, fig_dir))
            print(f"[figures]   {made[-1].name}")
        except FileNotFoundError as exc:
            print(f"[figures]   skipped {fn.__name__}: missing input ({exc})")
        except Exception as exc:  # keep going; report the rest
            print(f"[figures]   FAILED {fn.__name__}: {type(exc).__name__}: {exc}")
    return made
