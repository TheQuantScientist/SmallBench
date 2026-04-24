import asyncio
import json
import time
import ollama
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Optional, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────
BASE_DATA_DIR = Path(r"D:\forecast slm proj\data")
HISTORY_DIR   = BASE_DATA_DIR / "history"
TRUTH_DIR     = BASE_DATA_DIR / "ground_truth"
RESULTS_DIR   = Path(r"D:\forecast slm proj\results")


MODEL_NAME = "gemma3:4b"
TEST_START_DATE   = "2026-01-01"
LOOKBACKS         = [14, 21, 30]  # Bạn có thể thêm các mốc cần chạy vào đây
FORECAST_HORIZONS = [30]
EVAL_STEPS        = [1, 7, 14, 21, 30] # Các mốc để tính bảng thống kê


TEMPERATURE = 0.1
TOP_P       = 0.90
MAX_CONCURRENT = 4


SYSTEM_PROMPT_TEMPLATE = """You are a stock forecasting expert
Your task is to predict the next {horizon} daily Closing prices.
Input data includes historical Open, High, Low, Volume of the past {lookback} days.
Output exactly {horizon} numbers separated by semicolon with 3 decimal places.
Must follow this exact template: {template}
No text, no explanations.
"""


# ────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────
def load_stock_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, skiprows=3, header=None)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna().sort_values('Date').reset_index(drop=True)
    except: return pd.DataFrame()


def prepare_input_json(df: pd.DataFrame, lookback: int, symbol: str) -> str:
    recent = df.tail(lookback).copy()
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
    data_list = recent[["Date", "Open", "High", "Low", "Close", "Volume"]].to_dict("records")
    return json.dumps({"symbol": symbol, "data": data_list}, separators=(",", ":"))


def parse_prediction(text: str, horizon: int) -> Optional[List[float]]:
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(',', '.'))
    if len(numbers) < horizon: return None
    return [round(float(n), 3) for n in numbers[:horizon]]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0: return {"mae": 0, "rmse": 0, "mape": 0}
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else 0
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ────────────────────────────────────────────────
#  WORKER
# ────────────────────────────────────────────────
async def predict_one(idx: int, df: pd.DataFrame, semaphore: asyncio.Semaphore,
                     client: ollama.AsyncClient, lookback: int, horizon: int, symbol: str):
    async with semaphore:
        start_time = time.perf_counter()
        window = df.iloc[idx - lookback : idx]
        actual = df["Close"].iloc[idx : idx + horizon].values
        json_input = prepare_input_json(window, lookback, symbol)
        template = ";".join(["number"] * horizon)
       
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(horizon=horizon, lookback=lookback, template=template)
        try:
            response = await client.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Data:\n{json_input}\nPredict next {horizon}:"}
                ],
                options={"temperature": TEMPERATURE, "top_p": TOP_P}
            )
            preds = parse_prediction(response["message"]["content"], horizon)
            curr_date = df["Date"].iloc[idx].strftime("%Y-%m-%d")
            if preds:
                # Trả về kết quả kèm theo cả giá thực tế để tính toán sau này
                return {"date": curr_date, "actual": actual.tolist(), "predicted": preds}
        except: return None


async def process_symbol(symbol: str):
    hist_path, truth_path = HISTORY_DIR / f"{symbol}_input_history.csv", TRUTH_DIR / f"{symbol}_1d_full.csv"
    if not (hist_path.exists() and truth_path.exists()): return


    df = pd.concat([load_stock_data(hist_path), load_stock_data(truth_path)]).drop_duplicates('Date').sort_values("Date").reset_index(drop=True)
    start_idx = df[df["Date"] >= pd.to_datetime(TEST_START_DATE)].index[0]
    client, semaphore = ollama.AsyncClient(), asyncio.Semaphore(MAX_CONCURRENT)


    for lb in LOOKBACKS:
        out_file = RESULTS_DIR / f"{symbol}_lb{lb}_test.json"
        metrics_file = RESULTS_DIR / f"{symbol}_lb{lb}_metrics.json"


        # Nếu đã có cả file dự báo và file metrics thì bỏ qua
        if out_file.exists() and metrics_file.exists():
            print(f"⏭️  Bỏ qua {symbol} Lookback={lb} (Đã có đủ kết quả).")
            continue


        print(f"\n🚀 Đang chạy: {symbol} | Lookback={lb}")
        tasks = [predict_one(i, df, semaphore, client, lb, 30, symbol) for i in range(start_idx, len(df) - 30 + 1)]
        results = [r for r in await asyncio.gather(*tasks) if r]
       
        if results:
            # 1. Lưu file JSON dự báo
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
           
            # 2. LOGIC EVAL_STEPS (Kế thừa từ thầy)
            df_res = pd.DataFrame(results)
            metrics_summary = []
            print(f"📊 Kết quả Metric cho {symbol} (lb={lb}):")


            for step in EVAL_STEPS:
                # Lấy giá trị tại mốc 'step' (ví dụ ngày thứ 7 sau dự báo)
                y_true = df_res["actual"].apply(lambda x: x[step-1] if len(x) >= step else np.nan).dropna().values
                y_pred = df_res["predicted"].apply(lambda x: x[step-1] if len(x) >= step else np.nan).dropna().values
               
                if len(y_true) > 0:
                    m = compute_metrics(y_true, y_pred)
                    metrics_summary.append({
                        "step": step,
                        "mae": round(m["mae"], 4),
                        "rmse": round(m["rmse"], 4),
                        "mape": round(m["mape"], 4)
                    })
                    print(f"   Step +{step:2d}d | MAE: {m['mae']:8.3f} | RMSE: {m['rmse']:8.3f} | MAPE: {m['mape']:6.2f}%")


            # 3. Lưu file Metrics JSON
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump({
                    "symbol": symbol,
                    "lookback": lb,
                    "results_per_step": metrics_summary
                }, f, indent=2)
           
            print(f"✅ Đã lưu dự báo và bảng sai số cho {symbol}")


# ────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────
async def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_symbols = sorted([f.name.split('_')[0] for f in HISTORY_DIR.glob("*_input_history.csv")])
   
    print(f"Tìm thấy tổng cộng {len(all_symbols)} mã cổ phiếu.")
    for sym in all_symbols:
        await process_symbol(sym)
   
    print("\n🏁 TẤT CẢ CÁC MÃ ĐÃ ĐƯỢC XỬ LÝ XONG!")


if __name__ == "__main__":
    asyncio.run(main())

