"""Figure generation for the report and README.

Every figure answers a specific analytical question, uses honest axes (bars from
zero, labelled units in EUR/MWh or MW), and where it strengthens the argument it
quantifies uncertainty by simulation rather than asserting it. Figures read the
result parquets written by the pipeline; each is skipped gracefully if its
inputs are absent, so a partial pipeline still produces what it can.

Run via ``python scripts/run_pipeline.py --stage figures``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Consistent palette across figures.
C_NAIVE = "#9aa0a6"
C_RIDGE = "#4c78a8"
C_LGBM = "#c0392b"
C_BAND = "#c0392b"
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


# --- 1. price overview with regimes ----------------------------------------

def fig_price_overview(proc: Path, fig_dir: Path) -> Path:
    df = pd.read_parquet(proc / "dataset_clean.parquet")
    price = df["price_da"]
    spike = price.quantile(0.95)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(price.index, price.values, lw=0.4, color="#34495e", label="Day-ahead price")
    neg = price[price < 0]
    hi = price[price >= spike]
    ax.scatter(neg.index, neg.values, s=4, color="#2980b9", label="negative", zorder=3)
    ax.scatter(hi.index, hi.values, s=4, color="#e67e22", label="spike (>=95th pct)", zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("German day-ahead price: full sample, with negative and spike regimes")
    ax.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.9)
    return _save(fig, fig_dir, "01_price_overview")


# --- 2. merit-order curve ---------------------------------------------------

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
    return _save(fig, fig_dir, "02_merit_order")


# --- 3. forecast vs actual for the most volatile week -----------------------

def fig_forecast_week(proc: Path, fig_dir: Path) -> Path:
    p = pd.read_parquet(proc / "preds_model.parquet").sort_index()
    peak_ts = p["actual"].idxmax()
    half = pd.Timedelta(days=3, hours=12)
    w = p.loc[peak_ts - half : peak_ts + half]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(w.index, w["q05"], w["q95"], color=C_BAND, alpha=0.18,
                    label="90% interval (q05-q95)")
    ax.plot(w.index, w["actual"], color="#222", lw=1.3, label="actual")
    ax.plot(w.index, w["q50"], color=C_LGBM, lw=1.3, ls="--", label="forecast (median)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"Forecast vs actual, most volatile week ({w.index[0]:%Y-%m-%d})")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    return _save(fig, fig_dir, "03_forecast_week")


# --- 4. model comparison ----------------------------------------------------

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
    return _save(fig, fig_dir, "04_model_comparison")


# --- 5. cumulative skill (visualises the DM result + its stability) ---------

def fig_cumulative_skill(proc: Path, fig_dir: Path) -> Path:
    b = pd.read_parquet(proc / "preds_baselines.parquet")
    m = pd.read_parquet(proc / "preds_model.parquet").reindex(b.index)
    a = b["actual"]
    cum_naive = (_abs_err(a, b["seasonal_naive"]) - _abs_err(a, m["q50"])).cumsum()
    cum_ridge = (_abs_err(a, b["ridge"]) - _abs_err(a, m["q50"])).cumsum()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(cum_naive.index, cum_naive.values, color=C_NAIVE, label="vs seasonal-naive")
    ax.plot(cum_ridge.index, cum_ridge.values, color=C_RIDGE, label="vs Ridge")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("cumulative |error| saved (EUR/MWh)")
    ax.set_title("Cumulative accuracy gain of LightGBM (steady rise = stable, real edge)")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, fig_dir, "05_cumulative_skill")


# --- 6. error distribution --------------------------------------------------

def fig_error_distribution(proc: Path, fig_dir: Path) -> Path:
    b = pd.read_parquet(proc / "preds_baselines.parquet")
    m = pd.read_parquet(proc / "preds_model.parquet").reindex(b.index)
    a = b["actual"]
    e_lgbm = (a - m["q50"]).dropna()
    e_ridge = (a - b["ridge"]).dropna()
    lim = np.nanpercentile(np.abs(e_ridge), 99)
    bins = np.linspace(-lim, lim, 80)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.hist(e_ridge, bins=bins, color=C_RIDGE, alpha=0.5,
            label=f"Ridge (sd {e_ridge.std():.1f})")
    ax.hist(e_lgbm, bins=bins, color=C_LGBM, alpha=0.5,
            label=f"LightGBM (sd {e_lgbm.std():.1f})")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(e_lgbm.mean(), color=C_LGBM, ls="--", lw=1,
               label=f"LGBM mean {e_lgbm.mean():+.2f}")
    ax.set_xlabel("forecast error = actual - forecast (EUR/MWh)")
    ax.set_ylabel("count")
    ax.set_title("Error distribution: LightGBM is tighter and near-unbiased")
    ax.legend(fontsize=8)
    return _save(fig, fig_dir, "06_error_distribution")


# --- 7. MAE by hour ---------------------------------------------------------

def fig_mae_by_hour(proc: Path, fig_dir: Path) -> Path:
    s = pd.read_parquet(proc / "mae_by_hour.parquet").iloc[:, 0]
    worst = set(s.sort_values(ascending=False).head(3).index)
    colors = [C_LGBM if h in worst else C_RIDGE for h in s.index]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(s.index, s.values, color=colors)
    ax.set_xlabel("hour of day (local)")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Where the model struggles by hour (red = worst, evening peak)")
    ax.set_xticks(range(0, 24, 2))
    return _save(fig, fig_dir, "07_mae_by_hour")


# --- 8. MAE by regime -------------------------------------------------------

def fig_mae_by_regime(proc: Path, fig_dir: Path) -> Path:
    t = pd.read_parquet(proc / "mae_by_regime.parquet")
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    bars = ax.bar(t.index, t["mae"], color=[C_ACC, C_RIDGE, C_LGBM][: len(t)])
    for bar, (_, row) in zip(bars, t.iterrows(), strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, row["mae"] + 0.5,
                f"{row['mae']:.1f}\nn={int(row['n']):,}", ha="center", fontsize=8)
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Accuracy by price regime (spikes are intrinsically hard)")
    return _save(fig, fig_dir, "08_mae_by_regime")


# --- 9. interval coverage calibration ---------------------------------------

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
    return _save(fig, fig_dir, "09_coverage_calibration")


# --- 10. SHAP importance ----------------------------------------------------

def fig_shap(proc: Path, fig_dir: Path) -> Path:
    s = pd.read_parquet(proc / "shap_importance.parquet").iloc[:, 0].sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.barh(s.index, s.values, color=C_RIDGE)
    ax.set_xlabel("mean |SHAP| (EUR/MWh contribution)")
    ax.set_title("What the model relies on (SHAP global importance)")
    return _save(fig, fig_dir, "10_shap_importance")


# --- 11. ablation -----------------------------------------------------------

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
    return _save(fig, fig_dir, "11_ablation")


# --- 12. feature correlation (explains SHAP-vs-ablation divergence) ----------

def fig_feature_correlation(proc: Path, fig_dir: Path) -> Path:
    X = pd.read_parquet(proc / "features_X.parquet")
    keys = [
        "residual_load_fc", "fc_load_total", "fc_gen_wind_pv", "fc_gen_total",
        "fc_gen_pv", "price_lag_24h", "price_lag_168h", "price_roll7_mean",
    ]
    keys = [k for k in keys if k in X.columns]
    corr = X[keys].corr()
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=8)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title("Feature collinearity: why SHAP and ablation legitimately differ")
    fig.colorbar(im, ax=ax, fraction=0.046, label="correlation")
    return _save(fig, fig_dir, "12_feature_correlation")


# --- 13. trading equity curves ----------------------------------------------

def fig_trading_equity(proc: Path, fig_dir: Path) -> Path:
    con = pd.read_parquet(proc / "trade_backtest_consensus.parquet").sort_index()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(con.index, con["pnl"].cumsum(), color=C_RIDGE, label="consensus signal")
    cf_path = proc / "trade_backtest_consensus_confident.parquet"
    if cf_path.exists():
        cf = pd.read_parquet(cf_path).sort_index()
        ax.plot(cf.index, cf["pnl"].cumsum(), color=C_ACC,
                label="consensus + confidence filter")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("cumulative P&L (EUR/MWh, notional)")
    ax.set_title("Trading signal equity curve (mechanism demo, not deployable alpha)")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, fig_dir, "13_trading_equity")


# --- 14. SIMULATION: permutation null for the hit rate ----------------------

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
    ax.hist(null_hits * 100, bins=40, color=C_NAIVE, alpha=0.8,
            label="random-direction null")
    ax.axvline(obs * 100, color=C_LGBM, lw=2, label=f"observed {obs:.1%}")
    ax.axvline(50, color="k", ls="--", lw=1)
    ax.set_xlabel("hit rate (%)")
    ax.set_ylabel("frequency")
    ax.set_title(f"Is the edge real? Permutation test, p = {p_val:.4f} ({n} trades)")
    ax.legend(fontsize=8)
    return _save(fig, fig_dir, "14_trading_significance")


# --- 15. SIMULATION: transaction-cost sensitivity ---------------------------

def fig_cost_sensitivity(proc: Path, fig_dir: Path) -> Path:
    con = pd.read_parquet(proc / "trade_backtest_consensus.parquet")
    pos = con["position"].to_numpy()
    settle = con["settle"].to_numpy()
    traded = pos != 0
    n_tr = int(traded.sum())
    gross = pos * settle
    costs = np.linspace(0, 5, 51)
    avg_per_trade = [(gross[traded].sum() - c * n_tr) / n_tr for c in costs]
    breakeven = gross[traded].sum() / n_tr
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(costs, avg_per_trade, color=C_RIDGE, lw=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0.5, color=C_ACC, ls="--", lw=1, label="assumed cost (0.50)")
    ax.axvline(breakeven, color=C_LGBM, ls="--", lw=1, label=f"break-even ({breakeven:.2f})")
    ax.set_xlabel("transaction cost (EUR/MWh)")
    ax.set_ylabel("avg P&L per trade (EUR/MWh)")
    ax.set_title("Cost sensitivity: the edge is not a low-cost artifact")
    ax.legend(fontsize=8)
    return _save(fig, fig_dir, "15_cost_sensitivity")


_FIGURES = [
    fig_price_overview, fig_merit_order, fig_forecast_week, fig_model_comparison,
    fig_cumulative_skill, fig_error_distribution, fig_mae_by_hour, fig_mae_by_regime,
    fig_coverage, fig_shap, fig_ablation, fig_feature_correlation,
    fig_trading_equity, fig_trading_significance, fig_cost_sensitivity,
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
