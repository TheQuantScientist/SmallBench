import yfinance as yf
import os

# 1. Danh sách 10 mã ngành Y tế (Healthcare) bạn đảm nhận
healthcare_stocks = ["PFE", "JNJ", "UNH", "ABBV", "MRK", "LLY", "TMO", "DHR", "BMY", "AMGN"]

# 2. Cấu hình thời gian đúng theo Task (01/01/2021 đến 17/04/2026)
START_DATE = "2021-01-01"
END_DATE = "2026-04-17"

# Tạo thư mục lưu trữ nếu chưa có
if not os.path.exists('data'):
    os.makedirs('data')

print("Đang bắt đầu tải dữ liệu...")

for symbol in healthcare_stocks:
    try:
        # Tải dữ liệu từ Yahoo Finance
        df = yf.download(symbol, start=START_DATE, end=END_DATE)
        
        if not df.empty:
            # Lưu file theo định dạng mà code của thầy bạn yêu cầu: TênMã_1d_full.csv
            file_name = f"data/{symbol}_1d_full.csv"
            df.to_csv(file_name)
            print(f"Thành công: Đã lưu {file_name}")
        else:
            print(f"Cảnh báo: Không có dữ liệu cho mã {symbol}")
            
    except Exception as e:
        print(f"Lỗi khi tải mã {symbol}: {e}")

print("--- HOÀN THÀNH ---")