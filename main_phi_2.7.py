import asyncio
import json
import time
import ollama
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
from datetime import datetime
import re
from typing import List, Optional, Dict, Any

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────

DATA_DIR    = Path("C:/HocCode/lab/data")
RESULTS_DIR = Path("C:/HocCode/lab/result_phi_2.7b_1")

MODEL_NAME = "phi:2.7b" # <-- thay ten model o day


TEST_START_DATE = "2026-01-01"

LOOKBACKS         = [1, 7, 14, 21, 30]
FORECAST_HORIZONS = [30]               # ← only the longest one

EVAL_STEPS        = [1, 7, 14, 21, 30] # horizons you want to report metrics for

TEMPERATURE = 0.1
TOP_P       = 0.90

MAX_CONCURRENT = 3


# ────────────────────────────────────────────────
#  SYSTEM PROMPT TEMPLATE
# ────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are a stock forecasting expert.
Your task is to predict the next {horizon} daily Closing prices.
Input data includes historical Open, High, Low, Volume of the past {lookback} days. Use only the provided data.

CRITICAL OUTPUT RULES (You MUST follow strictly):
1. Output EXACTLY {horizon} numbers, no more, no less.
2. Separate numbers with SEMICOLON (;), no other separators.
3. Each number must have EXACTLY 4 decimal places (e.g., 142.3502, not 142.35 or 142.4).
4. ABSOLUTELY NO text, words, stock symbols (e.g., BAC, AXP), explanations, brackets, newlines, or extra characters.
5. DO NOT output Python dicts, JSON arrays, lists, or any code blocks. ONLY semicolon-separated numbers.

TEMPLATE: {template}
EXAMPLE: {example}

VIOLATIONS WILL CAUSE ERRORS. Output ONLY the numbers as specified.
"""


# ────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────

def prepare_input_json(df: pd.DataFrame, lookback: int, symbol: str) -> str:
    recent = df.tail(lookback).copy()
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
    
    data_list = recent[["Date", "open", "high", "low", "close", "volume"]].to_dict("records")
    
    payload = {
        "symbol": symbol,
        "timeframe": "1d",
        "lookback_days": lookback,
        "data": data_list
    }
    
    return json.dumps(payload, separators=(",", ":"), indent=None)


def extract_numbers_from_text(text: str, horizon: int) -> Optional[List[float]]:
    """Try to find horizon numbers from any text format including dicts, JSON, etc."""
    # Try semicolon-separated numbers first
    parts = [p.strip() for p in text.split(';') if p.strip()]
    
    # Try regex for decimal numbers
    if len(parts) < horizon:
        parts = re.findall(r'-?\d+\.\d{1,6}', text)
    if len(parts) < horizon:
        parts = re.findall(r'-?\d+\.\d+', text)
    if len(parts) < horizon:
        parts = re.findall(r'-?\d+', text)

    if len(parts) < horizon:
        return None

    try:
        preds = []
        for s in parts[:horizon]:
            s = s.replace(',', '.')
            val = float(s)
            preds.append(round(val, 3))
        return preds
    except (ValueError, TypeError):
        return None


def strip_code_fences(text: str) -> str:
    """Remove Python/JSON code fences like ```python ... ``` or ``` ... ```."""
    text = re.sub(r'```(?:python|json)?\s*', '', text)
    text = re.sub(r'\s*```', '', text)
    return text


def parse_prediction(text: str, horizon: int) -> Optional[List[float]]:
    if not text:
        return None

    text = text.strip()
    
    # Remove code fences first
    text = strip_code_fences(text)
    
    # Remove common prefixes like "BAC;", "AXP;", stock symbols followed by semicolon
    text = re.sub(r'^[A-Z]{2,5};', '', text)
    text = re.sub(r'^[^0-9.;\-]+', '', text)
    text = re.sub(r'[^0-9.;\-]+$', '', text)

    # Fix phi's bug: spaces in the middle of numbers like "357.130981 445312"
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    
    # Normalize: replace commas with semicolons (phi often uses comma as separator)
    text = text.replace(',', ';')
    
    # Normalize: replace multiple spaces with semicolon
    text = re.sub(r'\s+', ';', text)

    # Try standard semicolon split
    parts = [p.strip() for p in text.split(';') if p.strip()]
    
    if len(parts) >= horizon:
        try:
            preds = []
            for s in parts[:horizon]:
                s = s.replace(',', '.')
                val = float(s)
                preds.append(round(val, 3))
            return preds
        except (ValueError, TypeError):
            pass

    # Fallback: try extracting numbers from any text format
    return extract_numbers_from_text(text, horizon)


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

        template = ";".join(["number"] * horizon)
        example   = ";".join([f"{142.350 + i*0.5:.3f}" for i in range(horizon)])
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            horizon=horizon,
            lookback=lookback,
            template=template,
            example=example
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"{symbol} daily data (JSON):\n\n{json_input}\n\nNext {horizon} closing prices:"}
        ]
        
        raw = None
        preds = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = await client.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    options={"temperature": TEMPERATURE, "top_p": TOP_P, "num_predict": 2048}
                )
                raw = response["message"]["content"].strip()
            except Exception as e:
                date_str = df["Date"].iloc[idx].strftime("%Y-%m-%d")
                if attempt == max_retries - 1:
                    print(f"[{date_str}] LLM error after {max_retries} attempts: {e}")
                    return None
                continue
            
            preds = parse_prediction(raw, horizon)
            
            if preds is not None and len(preds) == horizon:
                break
            
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt+1} failed, retrying with shorter format...")
                if attempt == 0:
                    # First retry: tell model to stop outputting dicts/JSON
                    messages[1]["content"] = (
                        f"{symbol} daily data (JSON):\n\n{json_input}\n\n"
                        f"Next {horizon} closing prices — ONLY numbers with semicolons:\n"
                        f"DO NOT output Python dicts, JSON arrays, lists, or code blocks.\n"
                        f"Example of CORRECT output: 142.35;142.85;143.35;143.85"
                    )
                else:
                    # Second retry: even shorter, most explicit format
                    messages[1]["content"] = (
                        f"{symbol} daily data (JSON):\n\n{json_input}\n\n"
                        f"Output exactly {horizon} numbers, semicolon-separated, 2 decimals each:\n"
                        f"Example: 142.35;142.85;143.35"
                    )
        
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

        actual_rounded = [round(float(x), 3) for x in actual]

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
    print(f"\n{'═' * 90}")
    print(f"PROCESSING {symbol} — {data_path.name}")
    print(f"{'═' * 90}\n")

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
        for horizon in FORECAST_HORIZONS:   # now usually just one value: 28
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

            # ── Save predictions (full list of values) ──
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

    # TEST MODE: Only process first symbol to avoid freezing
    # Remove [:1] to process all symbols after testing successfully
    for data_path in data_files:
        symbol = data_path.stem.split('_')[0]
        await process_symbol(data_path, symbol)


if __name__ == "__main__":
    asyncio.run(main_async())