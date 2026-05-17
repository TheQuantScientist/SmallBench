import asyncio
import json
import time
import re
import ollama
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────

DATA_DIR    = Path("data")
RESULTS_DIR = Path("result_improved_v1")

MODEL_NAME = "gemma3:4b"  # Change model here: qwen2.5:3b, phi4-mini:3.8b, llama3.2:3b

TEST_START_DATE = "2026-01-01"

LOOKBACKS         = [1, 7, 14, 21, 30]
FORECAST_HORIZONS = [30]

EVAL_STEPS        = [1, 7, 14, 21, 30]

TEMPERATURE = 0.1
TOP_P       = 0.90

MAX_CONCURRENT = 3

# ────────────────────────────────────────────────
#  IMPROVEMENT 1: Technical Indicators as Features
#  Instead of raw OHLCV, compute MA, RSI, volatility
#  → shorter input, model understands patterns better
# ────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_features(df: pd.DataFrame, lookback: int) -> List[Dict[str, Any]]:
    """
    Compute technical indicators for the last `lookback` days.
    Returns a list of dicts (compact format).
    """
    recent = df.tail(lookback).copy()
    result = []

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Pre-compute indicators on full series to avoid NaN in rolling windows
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    rsi = compute_rsi(close, 14)
    volatility = close.rolling(5).std()
    volume_ma = volume.rolling(5).mean()
    atr = (high - low).rolling(5).mean()

    for _, row in recent.iterrows():
        idx = row.name
        entry = {
            "c": round(float(close.iloc[idx]), 2),
            "o": round(float(row["open"]), 2),
            "h": round(float(row["high"]), 2),
            "l": round(float(row["low"]), 2),
            "v": int(row["volume"]),
        }
        # Add technical indicators (skip NaN)
        if not np.isnan(ma5.iloc[idx]):
            entry["ma5"] = round(float(ma5.iloc[idx]), 2)
        if not np.isnan(ma10.iloc[idx]):
            entry["ma10"] = round(float(ma10.iloc[idx]), 2)
        if not np.isnan(ma20.iloc[idx]):
            entry["ma20"] = round(float(ma20.iloc[idx]), 2)
        if not np.isnan(rsi.iloc[idx]):
            entry["rsi"] = round(float(rsi.iloc[idx]), 2)
        if not np.isnan(volatility.iloc[idx]):
            entry["vol"] = round(float(volatility.iloc[idx]), 4)
        if not np.isnan(volume_ma.iloc[idx]):
            entry["vol_ma"] = int(volume_ma.iloc[idx])
        if not np.isnan(atr.iloc[idx]):
            entry["atr"] = round(float(atr.iloc[idx]), 2)

        result.append(entry)

    return result


def prepare_input_json(df: pd.DataFrame, lookback: int, symbol: str) -> str:
    """
    IMPROVEMENT 3: Compact input format with technical indicators.
    Much shorter than raw OHLCV JSON → less token count → less overload on small models.
    """
    features = compute_features(df, lookback)

    payload = {
        "symbol": symbol,
        "lb": lookback,
        "data": features
    }

    return json.dumps(payload, separators=(",", ":"))


# ────────────────────────────────────────────────
#  IMPROVEMENT 2: Few-shot Examples in Prompt
#  Show model real input → output examples
# ────────────────────────────────────────────────

FEWSHOT_EXAMPLE = """Example:
Input: {"symbol":"BAC","lb":7,"data":[{"c":45.12,"o":45.00,"h":45.50,"l":44.80,"v":50000000,"ma5":45.10,"rsi":52.3},{"c":45.30,"o":45.12,"h":45.60,"l":45.00,"v":48000000,"ma5":45.15,"rsi":54.1},{"c":45.15,"o":45.30,"h":45.40,"l":44.90,"v":52000000,"ma5":45.18,"rsi":51.8},{"c":45.40,"o":45.15,"h":45.70,"l":45.10,"v":47000000,"ma5":45.22,"rsi":55.2},{"c":45.25,"o":45.40,"h":45.50,"l":45.00,"v":51000000,"ma5":45.24,"rsi":53.5},{"c":45.50,"o":45.25,"h":45.80,"l":45.20,"v":49000000,"ma5":45.32,"rsi":56.8},{"c":45.35,"o":45.50,"h":45.60,"l":45.10,"v":53000000,"ma5":45.33,"rsi":54.9}]}
Output: 45.4200;45.5800;45.3100;45.6700;45.4500;45.7200;45.5300;45.6100;45.4800;45.7500;45.5600;45.6900;45.4100;45.7800;45.6200;45.5100;45.8000;45.6500;45.7300;45.5800;45.8200;45.6700;45.7100;45.5900;45.8400;45.7000;45.7600;45.6300;45.8100;45.7200"""

SYSTEM_PROMPT_TEMPLATE = """You are a stock forecasting expert.
Your task is to predict the next {horizon} daily Closing prices.
Input: historical OHLCV + technical indicators (MA5, MA10, MA20, RSI, Volatility, ATR) for the past {lookback} days.

CRITICAL OUTPUT RULES (MUST follow strictly):
1. Output EXACTLY {horizon} numbers, no more, no less.
2. Separate with SEMICOLON (;) only.
3. Each number: EXACTLY 4 decimal places (e.g., 45.2301).
4. NO text, NO symbols, NO brackets, NO explanations, NO newlines.

{fewshot}

VIOLATIONS WILL CAUSE ERRORS. Output ONLY the numbers."""


# ────────────────────────────────────────────────
#  IMPROVEMENT 4: Better Retry Mechanism
#  Retry messages address the SPECIFIC error, not just "use 2 decimals"
# ────────────────────────────────────────────────

RETRY_MESSAGES = [
    "ERROR: Your output was not valid. DO NOT output Python dicts, JSON objects, or code. Output ONLY numbers separated by semicolons. Example: 45.2300;45.4100;45.1800;...",
    f"CRITICAL: You MUST output EXACTLY {30} numbers. Your previous output had wrong format or wrong count. Output ONLY: number;number;number;... ({30} numbers total). NOTHING else. Each number must have 4 decimal places.",
]


# ────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────

def parse_prediction(text: str, horizon: int) -> Optional[List[float]]:
    if not text:
        return None

    text = text.strip()

    # Remove common prefixes: stock symbols, text before numbers
    text = re.sub(r'^[A-Z]{2,5};', '', text)
    text = re.sub(r'^[^0-9.;\-]+', '', text)
    text = re.sub(r'[^0-9.;\-]+$', '', text)

    # Fix spaces in numbers: "357.130981 445312" → "357.130981445312"
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)

    # Normalize separators
    text = text.replace(',', ';')
    text = re.sub(r'\s+', ';', text)

    parts = [p.strip() for p in text.split(';') if p.strip()]

    # Fallback: regex find all decimal numbers
    if len(parts) < horizon:
        parts = re.findall(r'-?\d+\.\d{1,6}', text)
    if len(parts) < horizon:
        parts = re.findall(r'-?\d+\.\d+', text)

    if len(parts) < horizon:
        return None

    try:
        preds = []
        for s in parts[:horizon]:
            s = s.replace(',', '.')
            val = float(s)
            preds.append(round(val, 4))
        return preds
    except (ValueError, TypeError):
        return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan}

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan

    return {"mae": mae, "rmse": rmse, "mape": mape}


# ────────────────────────────────────────────────
#  ASYNC PREDICTION WORKER
# ────────────────────────────────────────────────

async def predict_one(
    idx: int,
    df: pd.DataFrame,
    semaphore: asyncio.Semaphore,
    client: ollama.AsyncClient,
    lookback: int,
    horizon: int,
    symbol: str
) -> Optional[Dict[str, Any]]:
    async with semaphore:
        start = time.perf_counter()

        window = df.iloc[idx - lookback : idx]
        actual = df["close"].iloc[idx : idx + horizon].values
        json_input = prepare_input_json(window, lookback, symbol)

        fewshot = FEWSHOT_EXAMPLE
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            horizon=horizon,
            lookback=lookback,
            fewshot=fewshot
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"{symbol} data:\n\n{json_input}\n\nNext {horizon} closing prices:"}
        ]

        raw = None
        preds = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = await client.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    options={"temperature": TEMPERATURE, "top_p": TOP_P, "num_predict": 1500}
                )
                raw = response["message"]["content"].strip()
            except Exception as e:
                date_str = df["Date"].iloc[idx].strftime("%Y-%m-%d")
                if attempt == max_retries - 1:
                    print(f"[{date_str}] LLM error after {max_retries} attempts: {e}")
                    return None
                await asyncio.sleep(1)
                continue

            preds = parse_prediction(raw, horizon)

            if preds is not None and len(preds) == horizon:
                break

            # IMPROVEMENT 4: Retry with specific error messages
            if attempt < len(RETRY_MESSAGES):
                retry_msg = RETRY_MESSAGES[attempt].replace("{horizon}", str(horizon))
                messages.append({"role": "user", "content": retry_msg})
                print(f"  Attempt {attempt+1} failed, retrying with specific instruction...")

        date_str = df["Date"].iloc[idx].strftime("%Y-%m-%d")
        duration = time.perf_counter() - start

        if preds is None or len(preds) != horizon:
            print(f"[{date_str}] Parse failed after {max_retries} attempts ({duration:.2f}s) — raw: {raw[:120]}...")
            return {
                "date": date_str,
                "raw_output": raw,
                "parse_failed": True,
                "lookback": lookback,
                "horizon": horizon,
                "symbol": symbol
            }

        actual_rounded = [round(float(x), 4) for x in actual]

        print(f"[{date_str}] ({duration:.2f}s)")
        return {
            "date": date_str,
            "actual": actual_rounded,
            "predicted": preds,
            "raw_output": raw,
        }


# ────────────────────────────────────────────────
#  PROCESS ONE SYMBOL
# ────────────────────────────────────────────────

async def process_symbol(data_path: Path, symbol: str):
    print(f"\n{'=' * 90}")
    print(f"PROCESSING {symbol} — {data_path.name}")
    print(f"{'=' * 90}\n")

    try:
        df = pd.read_csv(data_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    print(f"Range: {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")
    print(f"Rows: {len(df)}\n")

    test_mask = df["Date"] >= pd.to_datetime(TEST_START_DATE)
    if not test_mask.any():
        print("No data after test start date.")
        return
    test_start_idx = test_mask.idxmax()

    max_lb = max(LOOKBACKS)
    if test_start_idx < max_lb:
        print("Not enough historical data before test start.")
        return

    client = ollama.AsyncClient()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    for lookback in LOOKBACKS:
        for horizon in FORECAST_HORIZONS:
            print(f"  → lookback={lookback:2d} | horizon={horizon:2d}")

            if len(df) - test_start_idx < lookback + horizon:
                print("    Not enough data for this combo — skipping")
                continue

            tasks = []
            for i in range(test_start_idx, len(df) - horizon + 1):
                tasks.append(predict_one(i, df, semaphore, client, lookback, horizon, symbol))

            results_raw = await asyncio.gather(*tasks, return_exceptions=True)

            results = []
            failures = []
            parse_fail_count = 0

            for res in results_raw:
                if isinstance(res, Exception) or res is None:
                    parse_fail_count += 1
                    failures.append({
                        "date": df["Date"].iloc[test_start_idx + len(failures)].strftime("%Y-%m-%d") if test_start_idx + len(failures) < len(df) else "unknown",
                        "error": str(res) if isinstance(res, Exception) else "None result",
                        "lookback": lookback,
                        "horizon": horizon,
                        "symbol": symbol
                    })
                    continue
                if res.get("parse_failed"):
                    parse_fail_count += 1
                    failures.append(res)
                    continue
                results.append(res)

            if not results:
                print("    No valid predictions.")
                continue

            df_res = pd.DataFrame(results)
            n_valid = len(df_res)
            print(f"    Valid: {n_valid} | Failures: {parse_fail_count}")

            # ── Compute metrics at selected steps ──
            metrics_list = []
            for step in EVAL_STEPS:
                if step > horizon:
                    continue
                y_true = df_res["actual"].apply(lambda x: x[step-1] if len(x) >= step else np.nan).dropna().values
                y_pred = df_res["predicted"].apply(lambda x: x[step-1] if len(x) >= step else np.nan).dropna().values

                if len(y_true) == 0:
                    continue

                m = compute_metrics(y_true, y_pred)
                metrics_list.append({
                    "horizon_step": step,
                    "mae":  m["mae"],
                    "rmse": m["rmse"],
                    "mape": m["mape"],
                    "n_samples": len(y_true)
                })
                print(f"      +{step:2d}d  MAE:{m['mae']:8.4f}  RMSE:{m['rmse']:8.4f}  MAPE:{m['mape']:6.2f}%")

            # ── Save predictions ──
            key = f"lb{lookback}_fh{horizon}"
            pred_file = RESULTS_DIR / f"{symbol}_{key}_predictions.json"
            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # ── Save parse failures log ──
            if failures:
                fail_file = RESULTS_DIR / f"{symbol}_{key}_failures.json"
                with open(fail_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "symbol": symbol,
                        "lookback": lookback,
                        "horizon": horizon,
                        "total_failures": len(failures),
                        "failures": failures
                    }, f, indent=2, ensure_ascii=False)
                print(f"    Saved failures → {fail_file.name}")

            # ── Save metrics ──
            metrics_file = RESULTS_DIR / f"{symbol}_{key}_metrics.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump({
                    "symbol": symbol,
                    "lookback": lookback,
                    "horizon": horizon,
                    "test_start": TEST_START_DATE,
                    "n_valid_windows": n_valid,
                    "parse_failures": parse_fail_count,
                    "date_range": [df_res['date'].min(), df_res['date'].max()] if n_valid > 0 else [],
                    "metrics_per_step": metrics_list
                }, f, indent=2)

            print(f"    Saved → {pred_file.name}")
            print(f"    Saved → {metrics_file.name}\n")


# ────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────

async def main_async():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data_files = sorted(DATA_DIR.glob("*_1d_full.csv"))

    if not data_files:
        print("No *_1d_full.csv files found in", DATA_DIR)
        return

    print(f"Found {len(data_files)} symbols to process:")
    for f in data_files:
        print(f"  • {f.stem.split('_')[0]}")
    print()

    # Process all symbols (remove [:1] after testing)
    for data_path in data_files[:1]:  # Test with first symbol only
        symbol = data_path.stem.split('_')[0]
        await process_symbol(data_path, symbol)


if __name__ == "__main__":
    asyncio.run(main_async())
