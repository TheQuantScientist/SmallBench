"""
SmallBench Plot — Visualization & Statistics for Stock Forecasting Results

Generates:
  1. Per-symbol time series: Actual vs Predicted (overview + zoom)
  2. Cross-model comparison: bar charts of MAE/RMSE/MAPE per model
  3. Per-sector heatmap: MAPE across symbols and lookbacks
  4. Summary CSV: aggregated metrics for all experiments

Usage:
  python plot.py                          # plot all results
  python plot.py --model qwen2.5:3b      # specific model
  python plot.py --sector tech            # specific sector
  python plot.py --summary-only           # only generate summary CSV (no plots)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"

SECTORS = {
    "tech": ["AAPL", "ADBE", "AVGO", "CRM", "CSCO", "GOOGL", "INTC", "MSFT", "NVDA", "ORCL"],
    "energy": ["COP", "CVX", "EOG", "EPD", "ET", "KMI", "SLB", "VLO", "WMB", "XOM"],
    "finance": ["AXP", "BAC", "BLK", "C", "GS", "JPM", "MA", "MS", "V", "WFC"],
    "healthcare": ["ABBV", "AMGN", "BMY", "DHR", "JNJ", "LLY", "MRK", "PFE", "TMO", "UNH"],
}

ALL_SYMBOLS = {sym: sector for sector, syms in SECTORS.items() for sym in syms}


# ════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════

def collect_all_metrics() -> pd.DataFrame:
    rows = []
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for mf in sorted(model_dir.glob("*_metrics.json")):
            try:
                with open(mf, encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue

            symbol = d.get("symbol", mf.stem.split("_")[0])
            sector = d.get("sector", ALL_SYMBOLS.get(symbol, "unknown"))
            lookback = d.get("lookback", 0)
            horizon = d.get("horizon", 30)
            valid = d.get("valid_windows", d.get("n_valid_windows", 0))
            fails = d.get("parse_failures", 0)
            success_rate = d.get("success_rate", "")

            for step in d.get("metrics_per_step", d.get("results_per_step", [])):
                rows.append({
                    "model": model_name,
                    "sector": sector,
                    "symbol": symbol,
                    "lookback": lookback,
                    "horizon": horizon,
                    "step": step.get("step", step.get("horizon_step", 0)),
                    "mae": step.get("mae", np.nan),
                    "rmse": step.get("rmse", np.nan),
                    "mape": step.get("mape", np.nan),
                    "n_samples": step.get("n", step.get("n_samples", 0)),
                    "valid_windows": valid,
                    "parse_failures": fails,
                    "success_rate": success_rate,
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_predictions(model_dir: Path, symbol: str, lookback: int, horizon: int = 30):
    key = f"lb{lookback}_fh{horizon}"
    file_path = model_dir / f"{symbol}_{key}_predictions.json"
    if not file_path.exists():
        return None
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return [row for row in data if not row.get("parse_failed", False)]
    except Exception:
        return None


def get_timeseries_df(data: list, step: int = 1) -> pd.DataFrame:
    dates, actuals, preds = [], [], []
    for row in data:
        a = row.get("actual", [])
        p = row.get("predicted", [])
        if len(a) >= step and len(p) >= step:
            dates.append(row["date"])
            actuals.append(a[step - 1])
            preds.append(p[step - 1])
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "Actual": actuals, "Predicted": preds})
    return df.sort_values("Date").reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
#  PLOT 1: TIME SERIES — Actual vs Predicted
# ════════════════════════════════════════════════════════════════

def plot_timeseries(model_dir: Path, model_name: str, symbol: str, lookback: int, out_dir: Path):
    data = load_predictions(model_dir, symbol, lookback)
    if not data:
        return

    df = get_timeseries_df(data, step=1)
    if df.empty:
        return

    _, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [2, 1]})

    # Overview
    ax1 = axes[0]
    ax1.plot(df["Date"], df["Actual"], label="Actual", color="#1f77b4", linewidth=2)
    ax1.plot(df["Date"], df["Predicted"], label=f"Predicted ({model_name})",
             color="#d62728", linestyle="--", linewidth=1.5, alpha=0.85)
    ax1.set_title(f"{symbol} — {model_name} | Lookback={lookback}", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Closing Price (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Zoom first month
    df_zoom = df[df["Date"].dt.month == df["Date"].iloc[0].month]
    if len(df_zoom) < 3:
        df_zoom = df.head(20)

    ax2 = axes[1]
    ax2.plot(df_zoom["Date"], df_zoom["Actual"], label="Actual",
             color="#1f77b4", linewidth=2, marker="o", markersize=5)
    ax2.plot(df_zoom["Date"], df_zoom["Predicted"], label="Predicted",
             color="#d62728", linestyle="--", linewidth=1.5, marker="x", markersize=7)
    ax2.set_title("Zoomed View (First Month)", fontsize=12)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

    plt.tight_layout()
    filename = out_dir / f"{symbol}_lb{lookback}_timeseries.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════════
#  PLOT 2: CROSS-MODEL COMPARISON — Bar Charts
# ════════════════════════════════════════════════════════════════

def plot_model_comparison(df_all: pd.DataFrame, out_dir: Path):
    if df_all.empty:
        return

    for step in [1, 7, 14, 30]:
        df_step = df_all[df_all["step"] == step]
        if df_step.empty:
            continue

        avg = df_step.groupby("model")[["mae", "rmse", "mape"]].mean().reset_index()
        avg = avg.sort_values("mape")

        _, axes = plt.subplots(1, 3, figsize=(20, 6))

        for ax, metric, color, title in zip(
            axes,
            ["mae", "rmse", "mape"],
            ["#2196F3", "#FF9800", "#4CAF50"],
            ["MAE", "RMSE", "MAPE (%)"],
        ):
            bars = ax.barh(avg["model"], avg[metric], color=color, alpha=0.85)
            ax.set_xlabel(title)
            ax.set_title(f"{title} @ Step +{step}d")
            ax.grid(axis="x", alpha=0.3)
            for bar, val in zip(bars, avg[metric]):
                if not np.isnan(val):
                    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                            f"{val:.2f}", va="center", fontsize=9)

        plt.suptitle(f"Cross-Model Comparison — Step +{step}d (avg across all symbols)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / f"comparison_step{step}.png", dpi=200, bbox_inches="tight")
        plt.close()


# ════════════════════════════════════════════════════════════════
#  PLOT 3: SECTOR HEATMAP — MAPE per Symbol × Lookback
# ════════════════════════════════════════════════════════════════

def plot_sector_heatmap(df_all: pd.DataFrame, out_dir: Path):
    if df_all.empty:
        return

    for model in df_all["model"].unique():
        df_m = df_all[(df_all["model"] == model) & (df_all["step"] == 1)]
        if df_m.empty:
            continue

        for sector, symbols in SECTORS.items():
            df_sec = df_m[df_m["sector"] == sector]
            if df_sec.empty:
                continue

            pivot = df_sec.pivot_table(
                index="symbol", columns="lookback", values="mape", aggfunc="mean"
            )
            if pivot.empty:
                continue

            _, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.6)))
            im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"lb={c}" for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                                fontsize=9, color="black" if val < 30 else "white")

            plt.colorbar(im, ax=ax, label="MAPE (%)")
            ax.set_title(f"MAPE@+1d — {model} | Sector: {sector}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(out_dir / f"heatmap_{model}_{sector}.png", dpi=200, bbox_inches="tight")
            plt.close()


# ════════════════════════════════════════════════════════════════
#  PLOT 4: SUCCESS RATE — Parse reliability per model
# ════════════════════════════════════════════════════════════════

def plot_success_rates(df_all: pd.DataFrame, out_dir: Path):
    if df_all.empty:
        return

    df_uniq = df_all.drop_duplicates(subset=["model", "symbol", "lookback"])
    if df_uniq.empty:
        return

    agg = df_uniq.groupby("model").agg(
        total_valid=("valid_windows", "sum"),
        total_fail=("parse_failures", "sum"),
    ).reset_index()
    agg["total"] = agg["total_valid"] + agg["total_fail"]
    agg["success_pct"] = (agg["total_valid"] / agg["total"] * 100).round(1)
    agg = agg.sort_values("success_pct", ascending=True)

    _, ax = plt.subplots(figsize=(10, max(4, len(agg) * 0.7)))
    colors = ["#4CAF50" if x > 90 else "#FF9800" if x > 70 else "#F44336" for x in agg["success_pct"]]
    bars = ax.barh(agg["model"], agg["success_pct"], color=colors, alpha=0.85)

    for bar, pct, valid, total in zip(bars, agg["success_pct"], agg["total_valid"], agg["total"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}% ({valid}/{total})", va="center", fontsize=9)

    ax.set_xlabel("Parse Success Rate (%)")
    ax.set_title("Model Reliability — Parse Success Rate", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "success_rates.png", dpi=200, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════════
#  SUMMARY CSV
# ════════════════════════════════════════════════════════════════

def generate_summary_csv(df_all: pd.DataFrame, out_dir: Path):
    if df_all.empty:
        print("No metrics data found.")
        return

    summary = (
        df_all.groupby(["model", "sector", "step"])
        .agg(
            avg_mae=("mae", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_mape=("mape", "mean"),
            total_samples=("n_samples", "sum"),
        )
        .reset_index()
        .round(4)
    )

    csv_path = out_dir / "summary_metrics.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary saved → {csv_path}")
    print(f"\nTop-10 best configs by MAPE:")
    top = summary.sort_values("avg_mape").head(10)
    print(top.to_string(index=False))

    overall = (
        df_all.groupby("model")
        .agg(avg_mae=("mae", "mean"), avg_mape=("mape", "mean"))
        .reset_index()
        .sort_values("avg_mape")
        .round(4)
    )
    print(f"\nOverall model ranking:")
    print(overall.to_string(index=False))


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SmallBench — Plot Results")
    parser.add_argument("--model", type=str, default=None, help="Filter by model name")
    parser.add_argument("--sector", type=str, default=None, help="Filter by sector")
    parser.add_argument("--summary-only", action="store_true", help="Only generate summary CSV")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SmallBench — Results Visualization")
    print("=" * 60)

    df_all = collect_all_metrics()
    if df_all.empty:
        print("\nNo results found in", RESULTS_DIR)
        print("Run main.py first to generate predictions.")
        return

    if args.model:
        matching = [m for m in df_all["model"].unique() if args.model in m]
        if matching:
            df_all = df_all[df_all["model"].isin(matching)]
        else:
            print(f"No results for model matching '{args.model}'")
            return

    if args.sector:
        df_all = df_all[df_all["sector"] == args.sector]

    print(f"\nFound metrics for {df_all['model'].nunique()} models, "
          f"{df_all['symbol'].nunique()} symbols, "
          f"{len(df_all)} data points")

    generate_summary_csv(df_all, PLOTS_DIR)

    if args.summary_only:
        return

    print("\nGenerating cross-model comparison plots...")
    plot_model_comparison(df_all, PLOTS_DIR)

    print("Generating sector heatmaps...")
    plot_sector_heatmap(df_all, PLOTS_DIR)

    print("Generating success rate chart...")
    plot_success_rates(df_all, PLOTS_DIR)

    print("\nGenerating per-symbol time series plots...")
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        model_plot_dir = PLOTS_DIR / model_name
        model_plot_dir.mkdir(parents=True, exist_ok=True)

        for sector, symbols in SECTORS.items():
            if args.sector and sector != args.sector:
                continue
            for symbol in symbols:
                for lb in [1, 7, 14, 21, 30]:
                    plot_timeseries(model_dir, model_name, symbol, lb, model_plot_dir)

    print(f"\nAll plots saved → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
