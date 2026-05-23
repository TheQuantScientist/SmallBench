import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ────────────────────────────────────────────────
#  CẤU HÌNH ĐƯỜNG DẪN
# ────────────────────────────────────────────────
# Trỏ đường dẫn này đến thư mục chứa các file JSON của bạn
RESULTS_DIR = Path(r"D:\forecast slm proj\results\gemma2_2b") 
OUTPUT_METRICS_DIR = RESULTS_DIR / "summary_metrics"
OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_STEPS = [1, 7, 14, 21, 30]

def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}

def process_json_files():
    # Tìm tất cả các file JSON chứa kết quả dự báo (thường có chữ 'predictions')
    json_files = list(RESULTS_DIR.glob("*predictions.json"))
    
    if not json_files:
        print(f"❌ Không tìm thấy file JSON nào tại {RESULTS_DIR}")
        return

    for file_path in json_files:
        print(f"📊 Đang xử lý: {file_path.name}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Chuyển dữ liệu sang DataFrame để dễ xử lý
        df = pd.DataFrame(data)
        
        # Nếu file bị lỗi không có cột actual/predicted thì bỏ qua
        if "actual" not in df.columns or "predicted" not in df.columns:
            continue

        metrics_summary = []
        
        # Tính toán sai số cho từng mốc (Step)
        for step in EVAL_STEPS:
            # Lấy giá trị tại mốc ngày thứ 'step' (index là step-1)
            y_true = df["actual"].apply(lambda x: x[step-1] if len(x) >= step else None).dropna()
            y_pred = df["predicted"].apply(lambda x: x[step-1] if len(x) >= step else None).dropna()
            
            # Đảm bảo 2 tập dữ liệu khớp nhau về độ dài
            common_idx = y_true.index.intersection(y_pred.index)
            y_true = y_true.loc[common_idx].values
            y_pred = y_pred.loc[common_idx].values

            if len(y_true) > 0:
                m = compute_metrics(y_true, y_pred)
                metrics_summary.append({
                    "step": step,
                    "mae": round(float(m["mae"]), 4),
                    "rmse": round(float(m["rmse"]), 4),
                    "mape": round(float(m["mape"]), 4),
                    "n_samples": len(y_true)
                })
        
        # Xuất ra file JSON mới chứa kết quả sai số
        symbol = file_path.stem.replace("_predictions", "")
        output_file = OUTPUT_METRICS_DIR / f"{symbol}_metrics_summary.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "file_source": file_path.name,
                "metrics": metrics_summary
            }, f, indent=2)
            
        print(f"✅ Đã lưu kết quả vào: {output_file.name}")
        for res in metrics_summary:
            print(f"   Step +{res['step']:2d}d | MAE: {res['mae']:8.3f} | MAPE: {res['mape']:6.2f}%")

if __name__ == "__main__":
    process_json_files()