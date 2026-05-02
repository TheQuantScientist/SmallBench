import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────
RESULTS_DIR = Path("../result/qwen2.5_3b")

# Thư mục lưu ảnh biểu đồ
PLOT_DIR = Path("../result/qwen2.5_3b/plot")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_predictions(symbol: str, lookback: int, horizon: int):
    key = f"lb{lookback}_fh{horizon}"
    file_path = RESULTS_DIR / f"{symbol}_{key}_predictions.json"

    if not file_path.exists():
        print(f"  ⚠️ Không tìm thấy file dữ liệu: {file_path.name}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_data = [row for row in data if not row.get("parse_failed", False)]
    return valid_data

# ────────────────────────────────────────────────
#  1. VẼ VÀ LƯU BIỂU ĐỒ LIÊN TỤC
# ────────────────────────────────────────────────
def plot_continuous_step(symbol: str, lookback: int, horizon: int, step: int = 1):
    data = load_predictions(symbol, lookback, horizon)
    if not data: return

    dates, actuals, preds = [], [], []

    for row in data:
        if len(row["actual"]) >= step and len(row["predicted"]) >= step:
            dates.append(row["date"])
            actuals.append(row["actual"][step - 1])
            preds.append(row["predicted"][step - 1])

    df = pd.DataFrame({"Date": pd.to_datetime(dates), "Actual": actuals, "Predicted": preds})
    df = df.sort_values("Date")

    plt.figure(figsize=(15, 5))
    plt.plot(df["Date"], df["Actual"], label='Thực tế', color='#1f77b4', linewidth=2)
    plt.plot(df["Date"], df["Predicted"], label=f'Dự đoán (t+{step})', color='#ff7f0e', linestyle='--', linewidth=1.5)

    plt.title(f"[{symbol}] Thực tế vs Dự đoán (Bước t+{step}) | LOOKBACK = {lookback}", fontsize=14, fontweight='bold')
    plt.xlabel("Ngày")
    plt.ylabel("Giá đóng cửa (Close)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # LƯU FILE ẢNH VÀ ĐÓNG BIỂU ĐỒ
    filename = PLOT_DIR / f"{symbol}_lb{lookback}_step{step}_continuous.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    💾 LƯU THÀNH CÔNG: {filename.name}")

    plt.close() # RẤT QUAN TRỌNG: Giải phóng RAM, không hiển thị ra màn hình

# ────────────────────────────────────────────────
#  2. VẼ VÀ LƯU BIỂU ĐỒ QUỸ ĐẠO
# ────────────────────────────────────────────────
def plot_forecast_trajectory(symbol: str, lookback: int, horizon: int, target_date: str = None):
    data = load_predictions(symbol, lookback, horizon)
    if not data: return

    record = None
    if target_date:
        for row in data:
            if row["date"] == target_date:
                record = row
                break
        if not record: return
    else:
        record = data[len(data) // 2]

    start_date = record["date"]
    actual_seq = record["actual"]
    pred_seq = record["predicted"]

    steps = np.arange(1, len(actual_seq) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(steps, actual_seq, label='Thực tế', color='green', marker='o', markersize=4)
    plt.plot(steps, pred_seq, label='Dự đoán (Model sinh ra)', color='red', marker='x', linestyle='--', markersize=4)

    plt.title(f"[{symbol}] Quỹ đạo dự đoán {horizon} ngày | Bắt đầu: {start_date} | LOOKBACK = {lookback}", fontsize=12)
    plt.xlabel("Số ngày tới (Horizon steps)")
    plt.ylabel("Giá đóng cửa (Close)")
    plt.xticks(steps)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # LƯU FILE ẢNH VÀ ĐÓNG BIỂU ĐỒ
    safe_date = start_date.replace("-", "")
    filename = PLOT_DIR / f"{symbol}_lb{lookback}_trajectory_{safe_date}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    💾 LƯU THÀNH CÔNG: {filename.name}")

    plt.close() # RẤT QUAN TRỌNG: Giải phóng RAM, không hiển thị ra màn hình

# ────────────────────────────────────────────────
#  MAIN EXECUTION BATCH
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # Danh sách các mã cổ phiếu bạn muốn chạy
    SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "ORCL", "ADBE", "CRM", "CSCO", "INTC", "AVGO"]

    HORIZON = 30
    LOOKBACKS = [1,7, 14, 21, 30]
    # Lấy cùng 1 ngày để dễ so sánh quỹ đạo. Nếu set = None, code sẽ chọn ngẫu nhiên
    TARGET_DATE = None

    print(f"Bắt đầu xuất ảnh hàng loạt vào: {PLOT_DIR.name}...\n")

    for sym in SYMBOLS:
        print(f"\n{'━' * 70}")
        print(f" BẮT ĐẦU VẼ CHO MÃ: {sym} ")
        print(f"{'━' * 70}")

        for lb in LOOKBACKS:
            print(f"  → Đang xử lý Lookback = {lb} ...")

            # Vẽ bước t+1
            plot_continuous_step(sym, lb, HORIZON, step=1)

            # Vẽ quỹ đạo 30 ngày
            plot_forecast_trajectory(sym, lb, HORIZON, target_date=TARGET_DATE)

    print(f"\n✅ ĐÃ HOÀN THÀNH TOÀN BỘ! Vui lòng kiểm tra thư mục: {PLOT_DIR}")