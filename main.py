"""
SmallBench Pipeline — Unified Stock Price Forecasting with Small Language Models

Combines best practices from all branches:
  - Technical indicators (RSI, MA, ATR, Volatility)  ← dev-nhan
  - Advanced prompt with template + reference price   ← dev-Binh
  - Few-shot examples                                 ← dev-nhan
  - Smart retry mechanism                             ← dev-nhan
  - Compact JSON input                                ← dev-nhan
  - Robust multi-format CSV loader                    ← dev-Binh + dev-han
  - Failure tracking & detailed logging               ← dev-nhan
  - Async concurrent processing                       ← all branches

Usage:
  python main.py                                    # all models, all sectors
  python main.py --model qwen2.5:3b                 # specific model
  python main.py --sector tech                      # specific sector
  python main.py --symbol AAPL                      # specific symbol
  python main.py --model gemma3:4b --sector finance --lookback 14
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import ollama
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

AVAILABLE_MODELS = [
    "qwen2.5:3b",
    "gemma3:4b",
    "gemma2:2b",
    "llama3.2:3b",
    "phi3:3.8b",
    "gemma4:e2b",
]

SECTORS = {
    "tech":       {"source": "dev-Binh",  "symbols": ["AAPL", "ADBE", "AVGO", "CRM", "CSCO", "GOOGL", "INTC", "MSFT", "NVDA", "ORCL"]},
    "energy":     {"source": "dev-nghia", "symbols": ["COP", "CVX", "EOG", "EPD", "ET", "KMI", "SLB", "VLO", "WMB", "XOM"]},
    "finance":    {"source": "dev-nhan",  "symbols": ["AXP", "BAC", "BLK", "C", "GS", "JPM", "MA", "MS", "V", "WFC"]},
    "healthcare": {"source": "dev-han",   "symbols": ["ABBV", "AMGN", "BMY", "DHR", "JNJ", "LLY", "MRK", "PFE", "TMO", "UNH"]},
}

TEST_START_DATE = "2026-01-01"
LOOKBACKS = [1, 7, 14, 21, 30]
FORECAST_HORIZON = 30
EVAL_STEPS = [1, 7, 14, 21, 30]

TEMPERATURE = 0.1
TOP_P = 0.90
MAX_CONCURRENT = 3
MAX_RETRIES = 3

# ════════════════════════════════════════════════════════════════
#  PROMPT ENGINEERING  (best of dev-Binh + dev-nhan)
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """You are a stock forecasting expert.
Your task is to predict the next {horizon} daily Closing prices.
Input: historical OHLCV + technical indicators (MA5, MA10, MA20, RSI, Volatility, ATR) for the past {lookback} days.
Before forecasting, understand the underlying trend, momentum, volatility, and volume changes carefully.

CRITICAL OUTPUT RULES (MUST follow strictly):
1. Output EXACTLY {horizon} numbers, no more, no less.
2. Separate numbers with SEMICOLON (;) only.
3. Each number: EXACTLY 4 decimal places (e.g., {ref_example}).
4. NO text, NO symbols, NO brackets, NO explanations, NO newlines.

Must follow this template: {template}
Example output: {example}

Count output to ensure enough {horizon} numbers before responding. No more, No less.

{fewshot}

VIOLATIONS WILL CAUSE ERRORS. Output ONLY the numbers."""

FEWSHOT_EXAMPLE = """Real example:
Input: {{"symbol":"BAC","lb":7,"data":[{{"c":45.12,"o":45.00,"h":45.50,"l":44.80,"v":50000000,"ma5":45.10,"rsi":52.3}},{{"c":45.30,"o":45.12,"h":45.60,"l":45.00,"v":48000000,"ma5":45.15,"rsi":54.1}},{{"c":45.15,"o":45.30,"h":45.40,"l":44.90,"v":52000000,"ma5":45.18,"rsi":51.8}},{{"c":45.40,"o":45.15,"h":45.70,"l":45.10,"v":47000000,"ma5":45.22,"rsi":55.2}},{{"c":45.25,"o":45.40,"h":45.50,"l":45.00,"v":51000000,"ma5":45.24,"rsi":53.5}},{{"c":45.50,"o":45.25,"h":45.80,"l":45.20,"v":49000000,"ma5":45.32,"rsi":56.8}},{{"c":45.35,"o":45.50,"h":45.60,"l":45.10,"v":53000000,"ma5":45.33,"rsi":54.9}}]}}
Output: 45.4200;45.5800;45.3100;45.6700;45.4500;45.7200;45.5300;45.6100;45.4800;45.7500;45.5600;45.6900;45.4100;45.7800;45.6200;45.5100;45.8000;45.6500;45.7300;45.5800;45.8200;45.6700;45.7100;45.5900;45.8400;45.7000;45.7600;45.6300;45.8100;45.7200"""

RETRY_MESSAGES = [
    "ERROR: Your output was not valid. DO NOT output Python dicts, JSON objects, or code. "
    "Output ONLY numbers separated by semicolons. Example: 45.2300;45.4100;45.1800;...",
    "CRITICAL: You MUST output EXACTLY {horizon} numbers. Your previous output had wrong "
    "format or wrong count. Output ONLY: number;number;number;... ({horizon} numbers total). "
    "NOTHING else. Each number must have 4 decimal places.",
]

# ════════════════════════════════════════════════════════════════
#  DATA LOADING  (unified loader for all CSV formats)
# ════════════════════════════════════════════════════════════════

def load_stock_csv(path: Path) -> pd.DataFrame:
    """
    Smart CSV loader that handles all formats from every branch:
      - Yahoo Finance format (Price,Close,... with 3 header rows)
      - Standard CSV with lowercase columns (open, high, low, close, volume)
      - Standard CSV with mixed-case columns (Open, High, Low, Close, Volume)
    Returns DataFrame with standardized lowercase columns: Date, open, high, low, close, volume
    """
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
    except Exception:
        return pd.DataFrame()

    try:
        if first_line.startswith("Price,"):
            df = pd.read_csv(path, skiprows=3, header=None)
            df.columns = ["Date", "close", "high", "low", "open", "volume"]
        else:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
    except Exception:
        return pd.DataFrame()

    col_map = {"date": "Date"}
    for col in df.columns:
        cl = col.lower()
        if cl in ("open", "high", "low", "close", "volume"):
            col_map[col] = cl
    df = df.rename(columns=col_map)

    required = ["Date", "open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "close"]).sort_values("Date").reset_index(drop=True)
    return df[required]


def load_symbol_data(sector: str, symbol: str) -> pd.DataFrame:
    """
    Load data for a symbol. For healthcare (dev-han), merges history + ground_truth.
    """
    sector_dir = DATA_DIR / sector
    full_path = sector_dir / f"{symbol}_1d_full.csv"
    hist_path = sector_dir / f"{symbol}_input_history.csv"

    frames = []
    if hist_path.exists():
        df_hist = load_stock_csv(hist_path)
        if not df_hist.empty:
            frames.append(df_hist)

    if full_path.exists():
        df_full = load_stock_csv(full_path)
        if not df_full.empty:
            frames.append(df_full)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    return df


# ════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS  (from dev-nhan main_improved.py)
# ════════════════════════════════════════════════════════════════

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_features(df: pd.DataFrame, lookback: int) -> List[Dict[str, Any]]:
    recent = df.tail(lookback).copy()
    result = []

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    rsi = compute_rsi(close, 14)
    volatility = close.rolling(5).std()
    atr = (high - low).rolling(5).mean()

    for _, row in recent.iterrows():
        idx = row.name
        entry = {
            "c": round(float(close.iloc[idx]), 4),
            "o": round(float(row["open"]), 4),
            "h": round(float(row["high"]), 4),
            "l": round(float(row["low"]), 4),
            "v": int(row["volume"]),
        }
        if not np.isnan(ma5.iloc[idx]):
            entry["ma5"] = round(float(ma5.iloc[idx]), 4)
        if not np.isnan(ma10.iloc[idx]):
            entry["ma10"] = round(float(ma10.iloc[idx]), 4)
        if not np.isnan(ma20.iloc[idx]):
            entry["ma20"] = round(float(ma20.iloc[idx]), 4)
        if not np.isnan(rsi.iloc[idx]):
            entry["rsi"] = round(float(rsi.iloc[idx]), 2)
        if not np.isnan(volatility.iloc[idx]):
            entry["vol"] = round(float(volatility.iloc[idx]), 4)
        if not np.isnan(atr.iloc[idx]):
            entry["atr"] = round(float(atr.iloc[idx]), 4)

        result.append(entry)

    return result


def prepare_input_json(df: pd.DataFrame, lookback: int, symbol: str) -> str:
    features = compute_features(df, lookback)
    payload = {"symbol": symbol, "lb": lookback, "data": features}
    return json.dumps(payload, separators=(",", ":"))


# ════════════════════════════════════════════════════════════════
#  PARSING  (combined regex from all branches)
# ════════════════════════════════════════════════════════════════

def parse_prediction(text: str, horizon: int) -> Optional[List[float]]:
    if not text:
        return None

    text = text.strip()

    # Remove stock symbol prefixes (e.g. "BAC;", "AXP;")
    text = re.sub(r"^[A-Z]{2,5};", "", text)
    # Remove non-numeric prefixes/suffixes
    text = re.sub(r"^[^0-9.;\-]+", "", text)
    text = re.sub(r"[^0-9.;\-]+$", "", text)
    # Fix phi bug: spaces inside numbers "357.130981 445312"
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)
    # Normalize separators
    text = text.replace(",", ";")
    text = re.sub(r"\s+", ";", text)

    parts = [p.strip() for p in text.split(";") if p.strip()]

    if len(parts) < horizon:
        parts = re.findall(r"-?\d+\.\d{1,6}", text)
    if len(parts) < horizon:
        parts = re.findall(r"-?\d+\.\d+", text)

    if len(parts) < horizon:
        return None

    try:
        return [round(float(s.replace(",", ".")), 4) for s in parts[:horizon]]
    except (ValueError, TypeError):
        return None


# ════════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = (
        np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        if np.all(y_true != 0)
        else np.nan
    )
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


# ════════════════════════════════════════════════════════════════
#  ASYNC PREDICTION WORKER
# ════════════════════════════════════════════════════════════════

async def predict_one(
    idx: int,
    df: pd.DataFrame,
    semaphore: asyncio.Semaphore,
    client: ollama.AsyncClient,
    model_name: str,
    lookback: int,
    horizon: int,
    symbol: str,
) -> Optional[Dict[str, Any]]:
    async with semaphore:
        start = time.perf_counter()

        window = df.iloc[idx - lookback : idx]
        actual = df["close"].iloc[idx : idx + horizon].values
        json_input = prepare_input_json(window, lookback, symbol)
        ref_price = float(window["close"].iloc[-1])

        template = ";".join(["number"] * horizon)
        example = ";".join(
            [f"{ref_price * (1 + i * 0.002):.4f}" for i in range(horizon)]
        )
        ref_example = f"{ref_price:.4f}"

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            horizon=horizon,
            lookback=lookback,
            template=template,
            example=example,
            ref_example=ref_example,
            fewshot=FEWSHOT_EXAMPLE,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{symbol} data:\n\n{json_input}\n\nNext {horizon} closing prices:",
            },
        ]

        raw = None
        preds = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat(
                    model=model_name,
                    messages=messages,
                    options={
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "num_predict": 1500,
                    },
                )
                raw = response["message"]["content"].strip()
            except Exception as e:
                date_str = df["Date"].iloc[idx].strftime("%Y-%m-%d")
                if attempt == MAX_RETRIES - 1:
                    print(f"    [{date_str}] LLM error after {MAX_RETRIES} attempts: {e}")
                    return None
                await asyncio.sleep(1)
                continue

            preds = parse_prediction(raw, horizon)
            if preds is not None and len(preds) == horizon:
                break

            if attempt < len(RETRY_MESSAGES):
                retry_msg = RETRY_MESSAGES[attempt].format(horizon=horizon)
                messages.append({"role": "user", "content": retry_msg})

        date_str = df["Date"].iloc[idx].strftime("%Y-%m-%d")
        duration = time.perf_counter() - start

        if preds is None or len(preds) != horizon:
            snippet = (raw[:120] + "...") if raw and len(raw) > 120 else raw
            print(f"    [{date_str}] Parse failed ({duration:.2f}s) — raw: {snippet}")
            return {
                "date": date_str,
                "raw_output": raw,
                "parse_failed": True,
                "symbol": symbol,
                "lookback": lookback,
            }

        print(f"    [{date_str}] OK ({duration:.2f}s)")
        return {
            "date": date_str,
            "actual": [round(float(x), 4) for x in actual],
            "predicted": preds,
            "raw_output": raw,
        }


# ════════════════════════════════════════════════════════════════
#  PROCESS ONE SYMBOL
# ════════════════════════════════════════════════════════════════

async def process_symbol(
    sector: str,
    symbol: str,
    model_name: str,
    lookbacks: List[int],
    model_results_dir: Path,
):
    df = load_symbol_data(sector, symbol)
    if df.empty:
        print(f"  [SKIP] {symbol} — no data found in data/{sector}/")
        return

    print(f"\n{'═' * 80}")
    print(f"  {symbol} ({sector}) | Model: {model_name}")
    print(f"  Range: {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()} | Rows: {len(df)}")
    print(f"{'═' * 80}")

    test_mask = df["Date"] >= pd.to_datetime(TEST_START_DATE)
    if not test_mask.any():
        print(f"  [SKIP] No data after {TEST_START_DATE}")
        return
    test_start_idx = test_mask.idxmax()

    if test_start_idx < max(lookbacks):
        print(f"  [SKIP] Not enough historical data before test start")
        return

    client = ollama.AsyncClient()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    for lookback in lookbacks:
        horizon = FORECAST_HORIZON

        out_key = f"{symbol}_lb{lookback}_fh{horizon}"
        pred_file = model_results_dir / f"{out_key}_predictions.json"
        metrics_file = model_results_dir / f"{out_key}_metrics.json"

        if pred_file.exists() and metrics_file.exists():
            print(f"\n  [SKIP] {symbol} lb={lookback} — already completed")
            continue

        if len(df) - test_start_idx < horizon:
            print(f"\n  [SKIP] lb={lookback} — not enough test data")
            continue

        task_indices = list(range(test_start_idx, len(df) - horizon + 1))
        total_expected = len(task_indices)
        print(f"\n  → lb={lookback:2d} | fh={horizon} | {total_expected} windows")

        tasks = [
            predict_one(i, df, semaphore, client, model_name, lookback, horizon, symbol)
            for i in task_indices
        ]

        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        failures = []
        for res in results_raw:
            if isinstance(res, Exception) or res is None:
                failures.append({"error": str(res) if isinstance(res, Exception) else "None"})
                continue
            if res.get("parse_failed"):
                failures.append(res)
                continue
            results.append(res)

        n_valid = len(results)
        n_fail = len(failures)
        print(f"    Valid: {n_valid} | Failures: {n_fail}")

        if not results:
            print("    No valid predictions — skipping save")
            continue

        results = sorted(results, key=lambda x: x["date"])
        df_res = pd.DataFrame(results)

        metrics_list = []
        for step in EVAL_STEPS:
            if step > horizon:
                continue
            y_true = (
                df_res["actual"]
                .apply(lambda x, s=step: x[s - 1] if len(x) >= s else np.nan)
                .dropna()
                .values
            )
            y_pred = (
                df_res["predicted"]
                .apply(lambda x, s=step: x[s - 1] if len(x) >= s else np.nan)
                .dropna()
                .values
            )
            if len(y_true) == 0:
                continue
            m = compute_metrics(y_true, y_pred)
            metrics_list.append(
                {"step": step, "mae": m["mae"], "rmse": m["rmse"], "mape": m["mape"], "n": len(y_true)}
            )
            print(f"      +{step:2d}d  MAE:{m['mae']:10.4f}  RMSE:{m['rmse']:10.4f}  MAPE:{m['mape']:7.2f}%")

        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if failures:
            fail_file = model_results_dir / f"{out_key}_failures.json"
            with open(fail_file, "w", encoding="utf-8") as f:
                json.dump({"total": len(failures), "failures": failures}, f, indent=2, ensure_ascii=False)

        success_rate = (n_valid / total_expected) * 100 if total_expected > 0 else 0
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "symbol": symbol,
                    "sector": sector,
                    "model": model_name,
                    "lookback": lookback,
                    "horizon": horizon,
                    "test_start": TEST_START_DATE,
                    "total_windows": total_expected,
                    "valid_windows": n_valid,
                    "parse_failures": n_fail,
                    "success_rate": f"{success_rate:.2f}%",
                    "metrics_per_step": metrics_list,
                },
                f,
                indent=2,
            )

        print(f"    Saved → {pred_file.name}, {metrics_file.name}")
        print(f"    Success rate: {n_valid}/{total_expected} ({success_rate:.1f}%)")


# ════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="SmallBench — Stock Forecasting with SLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model to run. Options: {', '.join(AVAILABLE_MODELS)}. Default: all models.",
    )
    parser.add_argument(
        "--sector",
        type=str,
        default=None,
        choices=list(SECTORS.keys()),
        help="Sector to process. Default: all sectors.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Specific stock symbol to process (e.g. AAPL).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        nargs="+",
        default=None,
        help=f"Lookback windows. Default: {LOOKBACKS}",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Max concurrent async requests. Default: {MAX_CONCURRENT}",
    )
    return parser.parse_args()


async def main_async():
    args = parse_args()

    global MAX_CONCURRENT
    MAX_CONCURRENT = args.max_concurrent
    lookbacks = args.lookback or LOOKBACKS

    models = [args.model] if args.model else AVAILABLE_MODELS
    sectors = [args.sector] if args.sector else list(SECTORS.keys())

    print("=" * 80)
    print("  SmallBench Pipeline — Stock Price Forecasting with SLMs")
    print("=" * 80)
    print(f"  Models:    {', '.join(models)}")
    print(f"  Sectors:   {', '.join(sectors)}")
    print(f"  Lookbacks: {lookbacks}")
    print(f"  Horizon:   {FORECAST_HORIZON}")
    print(f"  Test from: {TEST_START_DATE}")
    print(f"  Concurrency: {MAX_CONCURRENT}")
    if args.symbol:
        print(f"  Symbol filter: {args.symbol}")
    print("=" * 80)

    for model_name in models:
        model_dir_name = model_name.replace(":", "_").replace("/", "_")
        model_results_dir = RESULTS_DIR / model_dir_name
        model_results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'▓' * 80}")
        print(f"  MODEL: {model_name}")
        print(f"  Results → {model_results_dir}")
        print(f"{'▓' * 80}")

        for sector in sectors:
            sector_info = SECTORS[sector]
            symbols = sector_info["symbols"]

            if args.symbol:
                if args.symbol.upper() in symbols:
                    symbols = [args.symbol.upper()]
                else:
                    continue

            print(f"\n  ── Sector: {sector} ({len(symbols)} symbols) ──")

            for symbol in symbols:
                await process_symbol(
                    sector, symbol, model_name, lookbacks, model_results_dir
                )

    print(f"\n{'=' * 80}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main_async())
