import json
import pandas as pd
from pathlib import Path

# ────────────────────────────────────────────────
#  CẤU HÌNH ĐƯỜNG DẪN
# ────────────────────────────────────────────────
# Trỏ đến thư mục chứa các file _metrics.json của Gemma 3 4B
METRICS_DIR = Path(r"D:\forecast slm proj\results\gemma2_2b") 

# File Excel/CSV đầu ra để bạn nộp cho thầy
OUTPUT_PATH = METRICS_DIR / "Summary_Gemma2_2B_LB1.csv"

def generate_summary_table():
    summary_data = []
    
    # Tìm tất cả các file metrics của lookback 30
    metrics_files = list(METRICS_DIR.glob("*_lb1_metrics.json"))
    
    if not metrics_files:
        print(f"❌ Không tìm thấy file metrics nào tại {METRICS_DIR}")
        return

    for file_path in metrics_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            symbol = data.get("symbol", file_path.stem.split('_')[0])
            lookback = data.get("lookback", 30)
            
            # Duyệt qua các mốc step (+1, +7, +14, +21, +30)
            for res in data.get("results_per_step", []):
                summary_data.append({
                    "Stock": symbol,
                    "Lookback": lookback,
                    "Horizon_Step": f"+{res['step']}d",
                    "MAE": res["mae"],
                    "RMSE": res["rmse"],
                    "MAPE (%)": res["mape"]
                })
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc file {file_path.name}: {e}")

    # Tạo bảng bằng Pandas
    df_summary = pd.DataFrame(summary_data)

    if df_summary.empty:
        print("❌ Bảng dữ liệu trống, hãy kiểm tra lại cấu trúc file JSON.")
        return

    # Sắp xếp lại cho đẹp: theo Tên Stock và theo Step
    df_summary = df_summary.sort_values(by=["Stock", "Horizon_Step"])

    # Xuất ra file CSV (có thể mở bằng Excel)
    df_summary.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    
    print(f"✅ Đã lập bảng tổng hợp thành công!")
    print(f"📍 File lưu tại: {OUTPUT_PATH}")
    print("\nXem trước bảng:")
    print(df_summary.head(10))

if __name__ == "__main__":
    generate_summary_table()