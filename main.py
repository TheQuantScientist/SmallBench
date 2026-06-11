"""
SmallBench Pipeline — Unified Stock Price Forecasting with Small Language Models

Backend: HTTP API (vLLM / DCP AI Core Engine) — không dùng Ollama local.

Usage:
  python main.py
      → 5 models × 40 stocks × 5 lookbacks (full benchmark)

  python main.py --model qwen2.5:3b
  python main.py --sector tech
  python main.py --symbol AAPL
  python main.py --model gemma3:4b --sector finance --lookback 14

Models (alias → HuggingFace):
  gemma3:4b, gemma2:2b, qwen2.5:3b, hermes3:3b, phi2:2.7b
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

_env_path = PROJECT_ROOT / ".env"
if _env_path.is_file():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _val = _line.split("=", 1)
            os.environ.setdefault(_key.strip(), _val.strip())

API_URL = os.getenv(
    "API_URL",
    "https://looks-exp-route-jesus.trycloudflare.com/api/v1/generate",
)
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError(f"Thiếu API_KEY trong {_env_path}")

# Alias CLI → HuggingFace model id (API backend)
MODEL_REGISTRY: Dict[str, str] = {
    "gemma3:4b":    "google/gemma-3-4b-it",
    "gemma2:2b":    "google/gemma-2-2b-it",
    "qwen2.5:3b":   "Qwen/Qwen2.5-3B-Instruct",
    "hermes3:3b":   "NousResearch/Hermes-3-Llama-3.2-3B",
    "phi2:2.7b":    "microsoft/phi-2",
}

AVAILABLE_MODELS = list(MODEL_REGISTRY.keys())

SECTORS = {
    "tech":       {"source": "dev-Binh",  "symbols": ["AAPL", "ADBE", "AVGO", "CRM", "CSCO", "GOOGL", "INTC", "MSFT", "NVDA", "ORCL"]},
    "energy":     {"source": "dev-nghia", "symbols": ["COP", "CVX", "EOG", "EPD", "ET", "KMI", "SLB", "VLO", "WMB", "XOM"]},
    "finance":    {"source": "dev-nhan",  "symbols": ["AXP", "BAC", "BLK", "C", "GS", "JPM", "MA", "MS", "V", "WFC"]},
    "healthcare": {"source": "dev-han",   "symbols": ["ABBV", "AMGN", "BMY", "DHR", "JNJ", "LLY", "MRK", "PFE", "TMO", "UNH"]},
}

TOTAL_STOCKS = sum(len(info["symbols"]) for info in SECTORS.values())

# ── Đề bài ──
# Mỗi ngày trong [TEST_START_DATE, TEST_END_DATE]:
#   với mỗi lookback ∈ LOOKBACKS → dùng `lookback` ngày trước làm input → dự đoán FORECAST_HORIZON ngày tiếp theo
TEST_START_DATE = "2026-01-01"
TEST_END_DATE = "2026-04-17"
LOOKBACKS = [1, 7, 14, 21, 30]
FORECAST_HORIZON = 30
EVAL_STEPS = [1, 7, 14, 21, 30]  # đánh giá sai số tại các mốc +1,+7,+14,+21,+30 ngày

TEMPERATURE = 0.1
TOP_P = 0.90
MAX_TOKENS = 512
MAX_CONCURRENT = 3
MAX_RETRIES = 3
REQUEST_TIMEOUT = 300.0

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
    """Compute indicators on full history up to prediction point (positional index)."""
    df = df.reset_index(drop=True)
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

    start = max(0, len(df) - lookback)
    for pos in range(start, len(df)):
        row = df.iloc[pos]
        entry = {
            "c": round(float(close.iloc[pos]), 4),
            "o": round(float(row["open"]), 4),
            "h": round(float(row["high"]), 4),
            "l": round(float(row["low"]), 4),
            "v": int(row["volume"]),
        }
        if not np.isnan(ma5.iloc[pos]):
            entry["ma5"] = round(float(ma5.iloc[pos]), 4)
        if not np.isnan(ma10.iloc[pos]):
            entry["ma10"] = round(float(ma10.iloc[pos]), 4)
        if not np.isnan(ma20.iloc[pos]):
            entry["ma20"] = round(float(ma20.iloc[pos]), 4)
        if not np.isnan(rsi.iloc[pos]):
            entry["rsi"] = round(float(rsi.iloc[pos]), 2)
        if not np.isnan(volatility.iloc[pos]):
            entry["vol"] = round(float(volatility.iloc[pos]), 4)
        if not np.isnan(atr.iloc[pos]):
            entry["atr"] = round(float(atr.iloc[pos]), 4)

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


def get_task_indices(
    df: pd.DataFrame,
    lookback: int,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> List[int]:
    """
    Đề bài: mỗi ngày D trong [test_start, test_end], với lookback L ∈ LOOKBACKS:
      - Input  = L ngày giá trước D (lấy từ toàn bộ lịch sử df.iloc[:i])
      - Output = dự đoán giá 30 ngày tiếp theo (FORECAST_HORIZON)
    Điều kiện: i >= lookback (đủ L ngày trước ngày dự đoán trong data).
    """
    indices = []
    for i in range(lookback, len(df)):
        d = df["Date"].iloc[i]
        if d < test_start:
            continue
        if d > test_end:
            break
        indices.append(i)
    return indices


# ════════════════════════════════════════════════════════════════
#  API CLIENT  (HTTP — dev-Binh / dev-nghia)
# ════════════════════════════════════════════════════════════════

def resolve_hf_model(model_alias: str) -> str:
    if model_alias in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_alias]
    if "/" in model_alias:
        return model_alias
    raise ValueError(
        f"Model không hợp lệ: {model_alias}. "
        f"Chọn một trong: {', '.join(AVAILABLE_MODELS)}"
    )


async def call_generate_api(
    client: httpx.AsyncClient,
    hf_model_id: str,
    prompt: str,
) -> str:
    resp = await client.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model_name": hf_model_id,
            "prompt": prompt,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "top_p": TOP_P,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"API error: {data}")
    return data["response"].strip()


# ════════════════════════════════════════════════════════════════
#  ASYNC PREDICTION WORKER
# ════════════════════════════════════════════════════════════════

async def predict_one(
    idx: int,
    df: pd.DataFrame,
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    hf_model_id: str,
    lookback: int,
    horizon: int,
    symbol: str,
) -> Optional[Dict[str, Any]]:
    async with semaphore:
        start = time.perf_counter()

        history = df.iloc[:idx]
        if len(history) < lookback:
            return None
        actual = df["close"].iloc[idx : min(idx + horizon, len(df))].values
        json_input = prepare_input_json(history, lookback, symbol)
        ref_price = float(history["close"].iloc[-1])

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

        user_content = (
            f"{symbol} data:\n\n{json_input}\n\nNext {horizon} closing prices:"
        )
        prompt = f"{system_prompt}\n\n{user_content}"

        raw = None
        preds = None

        for attempt in range(MAX_RETRIES):
            try:
                raw = await call_generate_api(client, hf_model_id, prompt)
            except Exception as e:
                date_str = df["Date"].iloc[idx].strftime("%Y-%m-%d")
                if attempt == MAX_RETRIES - 1:
                    print(f"    [{date_str}] API error after {MAX_RETRIES} attempts: {e}")
                    return None
                await asyncio.sleep(1)
                continue

            preds = parse_prediction(raw, horizon)
            if preds is not None and len(preds) == horizon:
                break

            if attempt < len(RETRY_MESSAGES):
                retry_msg = RETRY_MESSAGES[attempt].format(horizon=horizon)
                prompt = f"{prompt}\n\n{retry_msg}"

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

    test_start = pd.to_datetime(TEST_START_DATE)
    test_end = pd.to_datetime(TEST_END_DATE)

    df = df.sort_values("Date").reset_index(drop=True)

    # Cần data sau TEST_END để có actual so sánh (horizon=30 ngày)
    if df["Date"].iloc[-1] < test_start:
        print(f"  [SKIP] No data from {TEST_START_DATE}")
        return

    pred_mask = (df["Date"] >= test_start) & (df["Date"] <= test_end)
    if not pred_mask.any():
        print(f"  [SKIP] No prediction days in {TEST_START_DATE} → {TEST_END_DATE}")
        return

    print(f"\n{'═' * 80}")
    print(f"  {symbol} ({sector}) | Model: {model_name}")
    print(f"  Data:    {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()} ({len(df)} rows)")
    print(f"  Predict: {TEST_START_DATE} → {TEST_END_DATE} | LOOKBACKS={lookbacks} | horizon={FORECAST_HORIZON}")
    print(f"{'═' * 80}")

    hf_model_id = resolve_hf_model(model_name)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # Mỗi giá trị trong LOOKBACKS = 1 lần chạy riêng (lb=1,7,14,21,30)
        for lookback in lookbacks:
            horizon = FORECAST_HORIZON

            out_key = f"{symbol}_lb{lookback}_fh{horizon}"
            pred_file = model_results_dir / f"{out_key}_predictions.json"
            metrics_file = model_results_dir / f"{out_key}_metrics.json"

            if pred_file.exists() and metrics_file.exists():
                print(f"\n  [SKIP] {symbol} lb={lookback} — already completed")
                continue

            task_indices = get_task_indices(df, lookback, test_start, test_end)
            total_expected = len(task_indices)

            if not task_indices:
                print(f"\n  [SKIP] lb={lookback} — không đủ {lookback} ngày lịch sử trước ngày {TEST_START_DATE}")
                continue

            pred_start = df["Date"].iloc[task_indices[0]].strftime("%Y-%m-%d")
            pred_end = df["Date"].iloc[task_indices[-1]].strftime("%Y-%m-%d")
            print(f"\n  → lb={lookback:2d} | predict {horizon}d ahead | {total_expected} ngày ({pred_start} → {pred_end})")

            tasks = [
                predict_one(i, df, semaphore, client, hf_model_id, lookback, horizon, symbol)
                for i in task_indices
            ]

            results_raw = await asyncio.gather(*tasks, return_exceptions=True)

            results = []
            failures = []
            for res in results_raw:
                if isinstance(res, Exception):
                    failures.append({"error": str(res)})
                    if len(failures) <= 3:
                        print(f"    [ERROR] {res}")
                    continue
                if res is None:
                    failures.append({"error": "API returned None"})
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
                        "hf_model_id": hf_model_id,
                        "lookback": lookback,
                        "horizon": horizon,
                        "test_start": TEST_START_DATE,
                        "test_end": TEST_END_DATE,
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

    if args.model:
        try:
            resolve_hf_model(args.model)
        except ValueError as e:
            print(e)
            return
        models = [args.model]
    else:
        models = AVAILABLE_MODELS
    sectors = [args.sector] if args.sector else list(SECTORS.keys())

    if args.symbol:
        sym = args.symbol.upper()
        n_stocks = 1 if any(sym in SECTORS[s]["symbols"] for s in sectors) else 0
    else:
        n_stocks = sum(len(SECTORS[s]["symbols"]) for s in sectors)

    n_configs = len(models) * n_stocks * len(lookbacks)

    print("=" * 80)
    print("  SmallBench Pipeline — Stock Price Forecasting with SLMs")
    print("=" * 80)
    print(f"  API_URL:     {API_URL}")
    print(f"  Models:      {len(models)} — {', '.join(models)}")
    print(f"  Sectors:     {', '.join(sectors)}")
    print(f"  Stocks:      {n_stocks}")
    print(f"  Lookbacks:   {lookbacks}")
    print(f"  Horizon:     {FORECAST_HORIZON}")
    print(f"  Test range:  {TEST_START_DATE} → {TEST_END_DATE}")
    print(f"  Concurrency: {MAX_CONCURRENT}")
    print(f"  Total configs: {n_configs}  ({len(models)} models × {n_stocks} stocks × {len(lookbacks)} lookbacks)")
    if args.symbol:
        print(f"  Symbol filter: {args.symbol.upper()}")
    print("=" * 80)
    print("\n  Model mapping:")
    for alias, hf_id in MODEL_REGISTRY.items():
        mark = "✓" if alias in models else " "
        print(f"    [{mark}] {alias:16s} → {hf_id}")
    print("=" * 80)

    for model_name in models:
        model_dir_name = model_name.replace(":", "_").replace("/", "_")
        model_results_dir = RESULTS_DIR / model_dir_name
        model_results_dir.mkdir(parents=True, exist_ok=True)

        hf_id = resolve_hf_model(model_name)
        print(f"\n{'▓' * 80}")
        print(f"  MODEL: {model_name}  →  {hf_id}")
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
