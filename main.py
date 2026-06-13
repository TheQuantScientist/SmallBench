import os
import asyncio
import json
import argparse
import time
import re
import aiohttp
from dotenv import load_dotenv

load_dotenv()
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

from typing import List, Optional, Dict, Any

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────
#
# =====================================================================
# HƯỚNG DẪN CHẠY SCRIPT TỪ COMMAND LINE (CLI)
# =====================================================================
# Script hỗ trợ cấu hình động qua tham số dòng lệnh (CLI). 
# Nếu không truyền tham số, script sẽ sử dụng các cấu hình mặc định.
#
# CÁC THAM SỐ:
# --sector     : Chỉ định thư mục ngành cần chạy (nằm trong `data/`).
#                Ví dụ: tech, finance. Nếu bỏ trống sẽ chạy toàn bộ.
#
# --lookbacks  : Mảng các ngày nhìn lại, cách nhau bởi khoảng trắng.
#                Ví dụ: --lookbacks 7 14 21 30
#
# VÍ DỤ CỤ THỂ:
# python main.py --sector tech --lookbacks 1 7
# =====================================================================

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")

MODEL_ALIASES = {
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma2-2b": "google/gemma-2-2b-it",
    "hermes3": "NousResearch/Hermes-3-Llama-3.2-3B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "phi2": "microsoft/phi-2",
}

HF_MODEL_ID = MODEL_ALIASES["hermes3"]
API_URL = os.getenv("API_URL", "https://looks-exp-route-jesus.trycloudflare.com/api/v1/generate")
API_KEY = os.getenv("API_KEY", "")

TEST_START_DATE = "2026-01-01"
TEST_END_DATE   = "2026-04-17"

LOOKBACKS         = [1, 7, 14, 21, 30]
FORECAST_HORIZONS = [30]

EVAL_STEPS        = [1, 7, 14, 21, 30]

TEMPERATURE = 0.1
TOP_P       = 0.90

MAX_CONCURRENT = 8

# ────────────────────────────────────────────────
#  SYSTEM PROMPT TEMPLATE
# ────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are a stock forecasting expert
Your task is to predict the next {horizon} daily Closing prices.
Input data includes only historical Closing prices of the past {lookback} days. Use only the provided data.
Before forecasting, understand the underlying trend and momentum carefully to make realistic predictions.
Predictions must be as close as possible to real-world closing prices.
Output exactly {horizon} numbers separated by semicolon with 3 decimal places.

Must follow this exact closing price template: {template}
Example: {example}

No text, no words, no explanations, no brackets, no newlines.
"""

# ────────────────────────────────────────────────
#  HELPERS (Synthesized Improvements)
# ────────────────────────────────────────────────

# IMPROVEMENT 1: Better Retry Mechanism (from dev-nhan)
RETRY_MESSAGES = [
    "ERROR: Your output was not valid. DO NOT output Python dicts, JSON objects, or code. Output ONLY numbers separated by semicolons.",
    "CRITICAL: You MUST output EXACTLY {horizon} numbers. Your previous output had wrong format or wrong count. Output ONLY: number;number;... NOTHING else.",
]

def round_price_data(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """IMPROVEMENT 2: Round prices to avoid precision noise (from dev-Binh & dev-han)"""
    df = df.copy()
    cols_to_round = ['Open', 'High', 'Low', 'Close', 'Volume', 'open', 'high', 'low', 'close', 'volume']
    existing_cols = [col for col in cols_to_round if col in df.columns]
    df[existing_cols] = df[existing_cols].round(decimals)
    return df


def prepare_input_compact(df: pd.DataFrame, lookback: int, symbol: str) -> str:
    """Compact format with only close prices to save context window tokens."""
    close_col = "Close" if "Close" in df.columns else "close"
    closes = df[close_col].tail(lookback).round(3)
    if closes.empty:
        return ""
    data_lines = "\n".join(f"{v:.3f}" for v in closes)
    return f"Symbol: {symbol}\nLookback: {lookback}\nData:\nc\n{data_lines}"

def parse_prediction(text: str, horizon: int) -> Optional[List[float]]:
    """IMPROVEMENT 5: Robust Regex-based parsing logic (from dev-han and dev-nhan)"""
    if not text:
        return None

    text = text.strip()
    text = re.sub(r'^[A-Z]{2,5};', '', text)
    text = re.sub(r'^[^0-9.;\-]+', '', text)
    text = re.sub(r'[^0-9.;\-]+$', '', text)
    
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    text = text.replace(',', ';')
    text = re.sub(r'\s+', ';', text)

    parts = [p.strip() for p in text.split(';') if p.strip()]

    if len(parts) < horizon:
        parts = re.findall(r'-?\d+\.\d+', text)

    if len(parts) < horizon:
        return None

    try:
        preds = []
        for s in parts[:horizon]:
            val = float(s.replace(',', '.'))
            preds.append(round(val, 3))
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
    session: aiohttp.ClientSession,
    lookback: int,
    horizon: int,
    symbol: str
) -> Optional[Dict[str, Any]]:
    async with semaphore:
        start = time.perf_counter()

        window = df.iloc[idx - lookback : idx]
        
        close_col = "Close" if "Close" in df.columns else "close"
        actual = df[close_col].iloc[idx : idx + horizon].values
        data_input = prepare_input_compact(window, lookback, symbol)

        template = ";".join(["number"] * horizon)
        
        # Few-shot example with close-price-only input
        example_preds = ";".join([f"{142.350 + i*0.5:.3f}" for i in range(horizon)])
        example_str = (
            f"Input:\n"
            f"Symbol: {symbol}\n"
            f"Lookback: {lookback}\n"
            f"Data:\n"
            f"c\n"
            f"142.350\n"
            f"142.800\n"
            f"...\n"
            f"Output: {example_preds}"
        )

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            horizon=horizon,
            lookback=lookback,
            template=template,
            example=example_str
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"{data_input}\n\nNext {horizon} closing prices:"}
        ]

        raw = None
        preds = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # API Backend requires `model_name` and `prompt` format (e.g. FastAPI / vLLM subset)
                prompt_str = "\n\n".join([f"{m['role'].upper()}:\n{m['content']}" for m in messages])
                payload = {
                    "model_name": HF_MODEL_ID,
                    "prompt": prompt_str,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_tokens": 1500
                }
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }
                async with session.post(API_URL, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if "choices" in result:
                        raw = result["choices"][0]["message"]["content"].strip()
                    elif "message" in result:
                        raw = result["message"]["content"].strip()
                    elif "response" in result:
                        raw = result["response"].strip()
                    elif "generated_text" in result:
                        raw = result["generated_text"].strip()
                    else:
                        # Fallback heuristic for raw array of objects (like some HuggingFace TGI endpoints)
                        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                            raw = result[0]["generated_text"].strip()
                        else:
                            raw = str(result)

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

            if attempt < len(RETRY_MESSAGES):
                retry_msg = RETRY_MESSAGES[attempt].replace("{horizon}", str(horizon))
                messages.append({"role": "user", "content": retry_msg})
                print(f"  Attempt {attempt+1} failed, retrying with specific instruction...")

        date_str = df["Date"].iloc[idx].strftime("%Y-%m-%d")
        duration = time.perf_counter() - start

        if preds is None or len(preds) != horizon:
            raw_preview = (raw[:120] + "...") if raw else "(no response)"
            print(f"[{date_str}] Parse failed after {max_retries} attempts ({duration:.2f}s) — raw: {raw_preview}")
            return {
                "date": date_str,
                "raw_output": raw,
                "parse_failed": True
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

async def process_symbol(data_path: Path, symbol: str, out_dir: Path):
    print(f"\n{'═' * 90}")
    print(f"PROCESSING {symbol} — {data_path.name}")
    print(f"{'═' * 90}\n")

    try:
        df = pd.read_csv(data_path)
        df = round_price_data(df, decimals=4)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values("Date").reset_index(drop=True)
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

    end_mask = df["Date"] <= pd.to_datetime(TEST_END_DATE)
    if not end_mask.any():
        print("No data before test end date.")
        return
    test_end_idx = end_mask[end_mask].index[-1]

    max_lb = max(LOOKBACKS)
    if test_start_idx < max_lb:
        print("Not enough historical data before test start.")
        return

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        for lookback in LOOKBACKS:
            for horizon in FORECAST_HORIZONS:
                print(f"  → lookback={lookback:2d} | horizon={horizon:2d}")

                if test_start_idx > test_end_idx:
                    print("    Invalid test range — skipping")
                    continue

                tasks = []
                # Predict for EVERY single day in the test range, even if the future actuals are shorter than horizon
                for i in range(test_start_idx, test_end_idx + 1):
                    tasks.append(predict_one(i, df, semaphore, session, lookback, horizon, symbol))

                results_raw = await asyncio.gather(*tasks, return_exceptions=True)

                results = []
                parse_fail_count = 0

                for res in results_raw:
                    if isinstance(res, Exception) or res is None:
                        parse_fail_count += 1
                        continue
                    if res.get("parse_failed"):
                        parse_fail_count += 1
                        continue
                    results.append(res)

                if not results:
                    print("    No valid predictions.")
                    continue

                df_res = pd.DataFrame(results)
                n_valid = len(df_res)
                print(f"    Valid: {n_valid} | Failures: {parse_fail_count}")

                metrics_list = []
                for step in EVAL_STEPS:
                    if step > horizon:
                        continue
                    valid_mask = df_res["actual"].apply(lambda x: len(x) >= step)
                    y_true = df_res.loc[valid_mask, "actual"].apply(lambda x: x[step-1]).values
                    y_pred = df_res.loc[valid_mask, "predicted"].apply(lambda x: x[step-1]).values
                    
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

                key = f"lb{lookback}_fh{horizon}"
                pred_file = out_dir / f"{symbol}_{key}_predictions.json"
                with open(pred_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                metrics_file = out_dir / f"{symbol}_{key}_metrics.json"
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "symbol": symbol,
                        "lookback": lookback,
                        "horizon": horizon,
                        "test_start": TEST_START_DATE,
                        "n_valid_windows": n_valid,
                        "parse_failures": parse_fail_count,
                        "date_range": [df_res['date'].min(), df_res['date'].max()],
                        "metrics_per_step": metrics_list
                    }, f, indent=2)

                print(f"    Saved → {pred_file.name}")
                print(f"    Saved → {metrics_file.name}\n")

# ────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────

async def main_async(sector: Optional[str], model_alias: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / model_alias
    out_dir.mkdir(parents=True, exist_ok=True)

    if sector:
        search_dir = DATA_DIR / sector
        data_files = sorted(search_dir.glob("*_1d_full.csv"))
    else:
        data_files = sorted(DATA_DIR.rglob("*_1d_full.csv"))

    if not data_files:
        print(f"No *_1d_full.csv files found for sector={sector if sector else 'all'} in {DATA_DIR}")
        return

    print(f"Found {len(data_files)} symbols to process:")
    for f in data_files:
        print(f"  • {f.stem.split('_')[0]}")
    print()

    for data_path in data_files:
        symbol = data_path.stem.split('_')[0]
        await process_symbol(data_path, symbol, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLM Forecasting")
    parser.add_argument("--lookbacks", type=int, nargs="+", default=LOOKBACKS, help="List of lookbacks to run (e.g. --lookbacks 7 14 21)")
    parser.add_argument("--sector", type=str, default=None, help="Specific sector folder to run (e.g. tech, finance, energy, healthcare). If None, runs all.")

    args = parser.parse_args()
    
    # Override globals
    LOOKBACKS = args.lookbacks
    
    # Determine the model alias for the results directory name
    model_alias = next((k for k, v in MODEL_ALIASES.items() if v == HF_MODEL_ID), HF_MODEL_ID)
    model_alias = model_alias.replace("/", "_")
    
    print(f"--- Config ---")
    print(f"Model    : {HF_MODEL_ID}")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Sector   : {args.sector if args.sector else 'All'}")
    print(f"--------------\n")

    asyncio.run(main_async(args.sector, model_alias))
