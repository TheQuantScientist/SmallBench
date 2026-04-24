import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# 1. CẤU HÌNH ĐƯỜNG DẪN (Bạn hãy chỉnh lại nếu thư mục của bạn khác)
RESULTS_DIR = Path(r"D:\forecast slm proj\results")
FILE_NAME = "ABBV_lb14_test.json"  # Tên file bạn muốn vẽ
FILE_PATH = RESULTS_DIR / FILE_NAME

def draw_plots():
    # 2. ĐỌC DỮ LIỆU TỪ FILE JSON
    if not FILE_PATH.exists():
        print(f"❌ Không tìm thấy file: {FILE_PATH}")
        return

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. TRÍCH XUẤT DỮ LIỆU (Lấy dự báo 1 bước - 1-step ahead)
    records = []
    for entry in data:
        records.append({
            "Date": pd.to_datetime(entry["date"]),
            "Actual": entry["actual"][0],     # Giá thực tế ngày đầu tiên
            "Predicted": entry["predicted"][0] # Giá AI đoán cho ngày đó
        })

    df = pd.DataFrame(records).sort_values("Date")

    # --- BIỂU ĐỒ 1: TỔNG THỂ ---
    plt.figure(figsize=(15, 7))
    plt.plot(df["Date"], df["Actual"], label="Thực tế (Actual)", color="#1f77b4", linewidth=2)
    plt.plot(df["Date"], df["Predicted"], label="Dự báo (Gemma 3)", color="#d62728", linestyle="--", alpha=0.8)
    
    plt.title(f"So sánh giá thực tế và dự báo - {FILE_NAME}", fontsize=14, fontweight='bold')
    plt.xlabel("Thời gian")
    plt.ylabel("Giá đóng cửa (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(RESULTS_DIR / f"{FILE_NAME.replace('.json', '')}_overview.png")
    print(f"✅ Đã lưu biểu đồ tổng thể tại: {RESULTS_DIR}")

    # --- BIỂU ĐỒ 2: ZOOM CẬN CẢNH (Tháng 1/2026) ---
    zoom_df = df[(df["Date"] >= "2026-01-01") & (df["Date"] <= "2026-01-31")]
    
    if not zoom_df.empty:
        plt.figure(figsize=(15, 8))
        plt.plot(zoom_df["Date"], zoom_df["Actual"], label="Thực tế (Actual)", 
                 color="#1f77b4", marker='o', markersize=6, linewidth=2)
        plt.plot(zoom_df["Date"], zoom_df["Predicted"], label="Dự báo (Gemma 3)", 
                 color="#d62728", marker='x', markersize=8, linestyle='--', linewidth=2)

        # Định dạng hiển thị ngày trên trục X
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())

        plt.title(f"Cận cảnh dự báo tháng 01/2026 - {FILE_NAME}", fontsize=14, fontweight='bold')
        plt.xlabel("Ngày/Tháng")
        plt.ylabel("Giá đóng cửa (USD)")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        plt.savefig(RESULTS_DIR / f"{FILE_NAME.replace('.json', '')}_zoom.png")
        print(f"✅ Đã lưu biểu đồ zoom tại: {RESULTS_DIR}")
    
    plt.show() # Hiển thị lên màn hình

if __name__ == "__main__":
    draw_plots()