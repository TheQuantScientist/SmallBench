import yfinance as yf
import pandas as pd
from pathlib import Path

DataDir = Path("C:/HocCode/lab")
DataDir.mkdir(parents=True, exist_ok=True)

FinanceStocks = ["JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP", "BLK"]

for TickerName in FinanceStocks:
    StockData = yf.download(TickerName, start="2021-01-01", end="2026-04-17") 
    StockData.reset_index(inplace=True)
    
    # gom các cột MultiIndex nếu có thành 1 index bình thường để dễ xử lý
    if isinstance(StockData.columns, pd.MultiIndex):
        StockData.columns = StockData.columns.get_level_values(0)
    
    StockData.columns = StockData.columns.str.lower()
    StockData.rename(columns={'date': 'Date'}, inplace=True)
    
    StockData.to_csv(DataDir / f"{TickerName}_1d_full.csv", index=False)
    print(f"Đã tải {TickerName}")