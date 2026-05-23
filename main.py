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
MODEL_NAME = "gemma2:2b"
BASE_DATA_DIR = Path(r"D:\forecast slm proj\data")
HISTORY_DIR   = BASE_DATA_DIR / "history"
TRUTH_DIR     = BASE_DATA_DIR / "ground_truth"
BASE_RESULTS_DIR = Path(r"D:\forecast slm proj\results")
RESULTS_DIR = BASE_RESULTS_DIR / MODEL_NAME.replace(":", "_")

TEST_START_DATE   = "2026-01-01"
LOOKBACKS         = [1, 14, 21, 30]  
FORECAST_HORIZONS = [30]
EVAL_STEPS        = [1, 7, 14, 21, 30]

TEMPERATURE = 0.1
TOP_P       = 0.90
MAX_CONCURRENT = 12 # Giảm xuống 6 để tránh tràn RAM và swap memory khi chạy đa luồng CPU

SYSTEM_PROMPT_TEMPLATE = """Task: Predict next {horizon} daily closing prices.
Input: {lookback} days history.
STRICT RULES:
- Output EXACTLY {horizon} numbers.
- Separate numbers ONLY by semicolon (;).
- NO text, NO explanations.
Prediction:"""

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
            
        df = df.dropna().sort_values('Date').reset_index(drop=True)
        # Ép dữ liệu bảng gốc làm tròn chuẩn 4 chữ số thập phân
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(4)
        return df
    except: return pd.DataFrame()

def prepare_input_json(df: pd.DataFrame, lookback: int, symbol: str) -> str:
    recent = df.tail(lookback).copy()
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
    data_list = recent[["Date", "Open", "High", "Low", "Close", "Volume"]].to_dict("records")
    return json.dumps({"symbol": symbol, "data": data_list}, separators=(",", ":"))

def parse_prediction(text: str, horizon: int) -> List[Any]:
    # Sử dụng Regex thông minh: Tách chuỗi theo dấu chấm phẩy (;) của Horizon trước
    raw_segments = text.replace(',', '.').split(';')
    
    parsed_horizon_list = []
    
    for i in range(horizon):
        if i < len(raw_segments):
            segment = raw_segments[i].strip()
            # Tìm tất cả các cụm số trong phân đoạn này
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", segment)
            
            if numbers:
                val = float(numbers[0])
                # BỘ LỌC TOÁN HỌC: Nếu dính số Volume (>1000) hoặc số năm (2026) -> Lưu vết lỗi số
                if val > 1000 or val == 2026:
                    parsed_horizon_list.append(f"Fail_Num({numbers[0]})")
                else:
                    parsed_horizon_list.append(round(val, 4))
            else:
                # Nếu phân đoạn này HOÀN TOÀN LÀ CHỮ (AI nói leo, viết giải thích)
                # repr() hoặc rút ngắn chuỗi chữ đó để hiện lên màn hình
                clean_text = segment.replace('\n', ' ').strip()
                if len(clean_text) > 15:
                    clean_text = clean_text[:12] + "..."
                parsed_horizon_list.append(f"Fail_Text({clean_text if clean_text else 'Empty'})")
        else:
            # Nếu AI lười, in thiếu không đủ 30 phân đoạn phân tách bằng dấu ;
            parsed_horizon_list.append("Fail_Text(Missing)")
            
    return parsed_horizon_list

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0: return {"mae": 0, "rmse": 0, "mape": 0}
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else 0
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}

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
       
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(horizon=horizon, lookback=lookback)
        try:
            response = await client.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Data:\n{json_input}\nPredict next {horizon}:"}
                ],
                options={"temperature": TEMPERATURE, "top_p": TOP_P}
            )
            content = response["message"]["content"]
            preds = parse_prediction(content, horizon)
            curr_date = df["Date"].iloc[idx].strftime("%Y-%m-%d")
            duration = time.perf_counter() - start_time
            actual_list = [round(float(a), 4) for a in actual.tolist()]

            # Bóc tách danh sách chi tiết các mốc lỗi và nguyên nhân cụ thể của nó
            failed_details = []
            for i, v in enumerate(preds):
                if isinstance(v, str) and v.startswith("Fail_"):
                    # Trích xuất nội dung lỗi thực tế của AI tại mốc đó
                    err_content = v.replace("Fail_Text(", "").replace("Fail_Num(", "").replace(")", "")
                    failed_details.append(f"+{i+1}d ({err_content})")

            fail_count = len(failed_details)

            if fail_count == 0:
                # TRƯỜNG HỢP 1: Hoàn hảo không lỗi mốc nào
                viz = f"{preds[0]:.4f}; {preds[1]:.4f}; {preds[2]:.4f} ... {preds[-2]:.4f}; {preds[-1]:.4f}"
                print(f"   [OK] Ngày {curr_date} | Xử lý: {duration:.2f}s | Horizon: {viz}")
                return {"date": curr_date, "actual": actual_list, "predicted": preds, "duration": duration, "status": "Success"}
            
            elif fail_count < horizon:
                # TRƯỜNG HỢP 2: Lỗi một vài mốc -> IN RÕ AI ĐÃ VIẾT CHỮ GÌ HOẶC SỐ SAI NÀO
                viz_elements = [f"{v:.4f}" if not isinstance(v, str) else "❌" for v in preds]
                viz = f"{viz_elements[0]}; {viz_elements[1]}; {viz_elements[2]} ... {viz_elements[-2]}; {viz_elements[-1]}"
                
                print(f"\n   [⚠️ PARTIAL FAIL] Ngày {curr_date} | Lỗi {fail_count}/{horizon} ngày forecast")
                print(f"      👉 Chi tiết lỗi tại mốc: {', '.join(failed_details)}")
                print(f"      👉 Chuỗi nhận diện tổng quan: {viz}\n")
                
                # Làm sạch mảng số để lưu file JSON (mốc lỗi lưu thành null để dễ lập trình tính toán toán học)
                clean_preds = [v if not isinstance(v, str) else None for v in preds]
                return {"date": curr_date, "actual": actual_list, "predicted": clean_preds, "duration": duration, "status": "Partial Fail"}
            
            else:
                # TRƯỜNG HỢP 3: Lỗi trắng toàn bộ chuỗi 30 ngày
                print(f"   [❌ FULL PARSE FAIL] Ngày {curr_date} | Lỗi toàn bộ chuỗi 30 ngày.")
                print(f"      👉 AI viết sai hoàn toàn: {repr(content)}")
                return {"date": curr_date, "actual": actual_list, "predicted": None, "raw_ai_response": content, "status": "Parse Fail"}
                
        except Exception as e: 
            curr_date = df["Date"].iloc[idx].strftime("%Y-%m-%d")
            print(f"   [⚠️ ERROR] Ngày {curr_date} | Lỗi hệ thống: {str(e)}")
            return {"date": curr_date, "actual": [round(float(a), 4) for a in actual.tolist()], "predicted": None, "raw_ai_response": str(e), "status": "Error"}

async def process_symbol(symbol: str):
    hist_path, truth_path = HISTORY_DIR / f"{symbol}_input_history.csv", TRUTH_DIR / f"{symbol}_1d_full.csv"
    if not (hist_path.exists() and truth_path.exists()): return

    df = pd.concat([load_stock_data(hist_path), load_stock_data(truth_path)]).drop_duplicates('Date').sort_values("Date").reset_index(drop=True)
    start_idx = df[df["Date"] >= pd.to_datetime(TEST_START_DATE)].index[0]
    client, semaphore = ollama.AsyncClient(), asyncio.Semaphore(MAX_CONCURRENT)

    for lb in LOOKBACKS:
        out_file = RESULTS_DIR / f"{symbol}_lb{lb}_test.json"
        metrics_file = RESULTS_DIR / f"{symbol}_lb{lb}_metrics.json"

        if out_file.exists() and metrics_file.exists():
            print(f"⏭️  Bỏ qua {symbol} Lookback={lb} (Đã có đủ kết quả).")
            continue

        task_indices = range(start_idx, len(df) - 30 + 1)
        total_expected = len(task_indices)
        print(f"\n🚀 Đang chạy: {symbol} | Lookback={lb} | Tổng cộng: {total_expected} ngày cần xử lý")
        
        tasks = [predict_one(i, df, semaphore, client, lb, 30, symbol) for i in task_indices]
        
        # as_completed giúp in log thời gian tiến độ từng ngày ngay lập tức ra màn hình khi xong
        results = []
        for task in asyncio.as_completed(tasks):
            res = await task
            if res: results.append(res)
       
        if results:
            results = sorted(results, key=lambda x: x['date'])
            # 1. Lưu file JSON dự báo FULL (Chứa cả mảng số đúng và vết chữ lỗi của AI)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
           
            # 2. CHỈ lọc những ngày "Success" để tính toán Metrics ngầm
            success_results = [r for r in results if r["status"] == "Success"]
            metrics_summary = []

            if success_results:
                df_res = pd.DataFrame(success_results)
                for step in EVAL_STEPS:
                    y_true = df_res["actual"].apply(lambda x: x[step-1] if len(x) >= step else np.nan).dropna().values
                    y_pred = df_res["predicted"].apply(lambda x: x[step-1] if len(x) >= step else np.nan).dropna().values
                   
                    if len(y_true) > 0:
                        m = compute_metrics(y_true, y_pred)
                        metrics_summary.append({
                            "step": step,
                            "mae": m["mae"],
                            "rmse": m["rmse"],
                            "mape": m["mape"]
                        })
            
            # Không print bảng MAE ra màn hình nữa, in trực tiếp dòng chốt tỷ lệ thành công khoa học
            success_count = len(success_results)
            fail_count = total_expected - success_count
            success_rate = (success_count / total_expected) * 100

            # 3. Lưu file Metrics JSON
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump({
                    "symbol": symbol,
                    "lookback": lb,
                    "total_samples_attempted": total_expected,
                    "success_samples": success_count,
                    "fail_samples": fail_count,
                    "success_rate": f"{success_rate:.2f}%",
                    "results_per_step": metrics_summary
                }, f, indent=2)
           
            print(f"✅ Đã lưu kết quả ngầm cho {symbol} | Tỷ lệ Parse chuẩn: {success_count}/{total_expected} ({success_rate:.2f}%)")

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