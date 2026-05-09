import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────
# Thay đổi thư mục nếu bạn đang chạy model khác (ví dụ: gemma-3)
RESULTS_DIR = Path("../result/hermes3_3b")
PLOT_DIR = Path("../result/hermes3_3b/plot")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "ORCL", "ADBE", "CRM", "CSCO", "INTC", "AVGO"]
MODEL_NAME = "hermes3:3b" # Tên model hiển thị trên chú thích (Legend)

def load_predictions(symbol: str, lookback: int, horizon: int):
    key = f"lb{lookback}_fh{horizon}"
    file_path = RESULTS_DIR / f"{symbol}_{key}_predictions.json"

    if not file_path.exists():
        print(f"⚠️ Không tìm thấy file: {file_path.name}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [row for row in data if not row.get("parse_failed", False)]

def get_timeseries_df(data, step=1):
    """Trích xuất dữ liệu của bước dự đoán t+step thành DataFrame."""
    dates, actuals, preds = [], [], []
    for row in data:
        if len(row["actual"]) >= step and len(row["predicted"]) >= step:
            dates.append(row["date"])
            actuals.append(row["actual"][step - 1])
            preds.append(row["predicted"][step - 1])

    df = pd.DataFrame({"Date": pd.to_datetime(dates), "Actual": actuals, "Predicted": preds})
    return df.sort_values("Date")

# ────────────────────────────────────────────────
#  1. BIỂU ĐỒ TỔNG QUAN (ĐƯỜNG LIỀN & NÉT ĐỨT)
# ────────────────────────────────────────────────
def plot_full_overview(df: pd.DataFrame, symbol: str, lookback: int):
    plt.figure(figsize=(15, 7))
    plt.plot(df["Date"], df["Actual"], label='Thực tế (Actual)', color='#1f77b4', linewidth=2)
    plt.plot(df["Date"], df["Predicted"], label=f'Dự báo ({MODEL_NAME})', color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8)

    plt.title(f"So sánh giá thực tế và dự báo - {symbol}_lb{lookback}_test.json", fontsize=14, fontweight='bold')
    plt.xlabel("Thời gian")
    plt.ylabel("Giá đóng cửa (USD)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='-', alpha=0.3)

    # Định dạng trục x hiển thị tháng-năm cho dễ nhìn
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.tight_layout()

    filename = PLOT_DIR / f"{symbol}_lb{lookback}_overview.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ────────────────────────────────────────────────
#  2. BIỂU ĐỒ CẬN CẢNH (CÓ MARKER & TRỤC X LÀ NGÀY/THÁNG)
# ────────────────────────────────────────────────
def plot_zoomed_view(df: pd.DataFrame, symbol: str, lookback: int, zoom_year=2026, zoom_month=1):
    # Lọc dữ liệu theo tháng và năm cần zoom
    df_zoom = df[(df['Date'].dt.year == zoom_year) & (df['Date'].dt.month == zoom_month)]

    if df_zoom.empty:
        print(f"   -> ⚠️ Không có dữ liệu cho tháng {zoom_month}/{zoom_year} để vẽ cận cảnh.")
        return

    plt.figure(figsize=(15, 7))

    # Thêm marker 'o' cho thực tế và 'x' cho dự báo như trong ảnh
    plt.plot(df_zoom["Date"], df_zoom["Actual"], label='Thực tế (Actual)',
             color='#1f77b4', linewidth=2, marker='o', markersize=6)
    plt.plot(df_zoom["Date"], df_zoom["Predicted"], label=f'Dự báo ({MODEL_NAME})',
             color='#d62728', linestyle='--', linewidth=1.5, marker='x', markersize=8)

    plt.title(f"Cận cảnh dự báo tháng {zoom_month:02d}/{zoom_year} - {symbol}_lb{lookback}_test.json", fontsize=14, fontweight='bold')
    plt.xlabel("Ngày/Tháng")
    plt.ylabel("Giá đóng cửa (USD)")
    plt.legend(loc='upper right')

    # Lưới đứt nét nhạt giống ảnh thứ 2
    plt.grid(True, linestyle='--', alpha=0.5)

    # Định dạng trục x chuẩn Ngày/Tháng (Ví dụ: 01/01, 02/01)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.xaxis.set_major_locator(mdates.DayLocator()) # Hiển thị từng ngày một
    plt.xticks(rotation=0)
    plt.tight_layout()

    filename = PLOT_DIR / f"{symbol}_lb{lookback}_zoomed_{zoom_month:02d}_{zoom_year}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ────────────────────────────────────────────────
#  VÒNG LẶP CHÍNH
# ────────────────────────────────────────────────
if __name__ == "__main__":
    HORIZON = 30
    LOOKBACKS = [1,14,21,30] # Bạn có thể thêm [1, 14, 21, 30] nếu cần

    print(f"Bắt đầu xuất biểu đồ cho {len(SYMBOLS)} mã cổ phiếu...\n")

    for symbol in SYMBOLS:
        for lb in LOOKBACKS:
            data = load_predictions(symbol, lb, HORIZON)
            if not data:
                continue

            # Lấy dữ liệu dự báo bước 1 (t+1) để nối thành chuỗi liên tục
            df_plot = get_timeseries_df(data, step=1)

            if df_plot.empty:
                continue

            print(f"📊 Đang vẽ [{symbol}] - Lookback {lb}...")

            # 1. Vẽ tổng quan
            plot_full_overview(df_plot, symbol, lb)

            # 2. Vẽ cận cảnh (Ví dụ: tháng 1 năm 2026)
            plot_zoomed_view(df_plot, symbol, lb, zoom_year=2026, zoom_month=1)

    print(f"\n✅ Hoàn tất! Ảnh đã được lưu tại: {PLOT_DIR}")