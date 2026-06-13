from __future__ import annotations

"""
Tải dữ liệu giá cổ phiếu (OHLCV) → data/<sector>/<SYMBOL>_1d_full.csv

  python crawl_data.py
  python crawl_data.py --sector energy
  python crawl_data.py --symbol XOM
"""

from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

# ── Cấu hình ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
START = "2025-05-30"
END = "2026-05-30"   # yfinance: ngày kết thúc không tính → lấy đủ tới 17/04/2026

SECTORS = {
    "tech":       ["AAPL", "ADBE", "AVGO", "CRM", "CSCO", "GOOGL", "INTC", "MSFT", "NVDA", "ORCL"],
    "energy":     ["COP", "CVX", "EOG", "EPD", "ET", "KMI", "SLB", "VLO", "WMB", "XOM"],
    "finance":    ["AXP", "BAC", "BLK", "C", "GS", "JPM", "MA", "MS", "V", "WFC"],
    "healthcare": ["ABBV", "AMGN", "BMY", "DHR", "JNJ", "LLY", "MRK", "PFE", "TMO", "UNH"],
}

HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_company_name(symbol: str) -> str:
    """BeautifulSoup: lấy tên công ty từ Yahoo (chỉ để hiển thị)."""
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}"
        html = requests.get(url, headers=HEADERS, timeout=10).text
        h1 = BeautifulSoup(html, "html.parser").find("h1")
        return h1.get_text(strip=True) if h1 else symbol
    except Exception:
        return symbol


def fetch_ohlcv(symbol: str) -> pd.DataFrame | None:
    """yfinance: tải Open, High, Low, Close, Volume."""
    raw = yf.download(symbol, start=START, end=END, progress=False, auto_adjust=False)
    if raw.empty:
        return None

    df = raw.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    df = df[["Date", "open", "high", "low", "close", "volume"]]
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].round(3)
    df["volume"] = df["volume"].astype(int)
    return df


def save(sector: str, symbol: str) -> bool:
    name = get_company_name(symbol)
    df = fetch_ohlcv(symbol)
    if df is None:
        print(f"  skip {symbol}")
        return False

    folder = DATA_DIR / sector
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{symbol}_1d_full.csv"
    df.to_csv(path, index=False, float_format="%.3f")
    print(f"  ok {symbol} ({name}) — {len(df)} dòng → {path}")
    return True


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sector", choices=SECTORS)
    p.add_argument("--symbol")
    args = p.parse_args()

    print(f"Tải dữ liệu {START} → 2026-04-17\n")

    ok = 0
    for sector, symbols in SECTORS.items():
        if args.sector and sector != args.sector:
            continue
        for sym in symbols:
            if args.symbol and sym != args.symbol.upper():
                continue
            print(f"[{sector}]")
            if save(sector, sym):
                ok += 1

    print(f"\nXong: {ok} file đã lưu.")


if __name__ == "__main__":
    main()
