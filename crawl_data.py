"""
Crawl OHLCV:
  - BeautifulSoup: kiểm tra mã + lấy tên công ty từ Yahoo Finance
  - yfinance     : tải lịch sử giá (OHLCV)

Output: data/<sector>/<SYMBOL>_1d_full.csv  (Date,open,high,low,close,volume)

  python crawl_data.py
  python crawl_data.py --sector energy
  python crawl_data.py --symbol XOM
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent / "data"
START = "2025-05-30"
END = "2026-04-17"
YFINANCE_END = "2026-04-18"  # end exclusive → lấy đủ tới 17/04/2026

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

SECTORS = {
    "tech": ["AAPL", "ADBE", "AVGO", "CRM", "CSCO", "GOOGL", "INTC", "MSFT", "NVDA", "ORCL"],
    "energy": ["COP", "CVX", "EOG", "EPD", "ET", "KMI", "SLB", "VLO", "WMB", "XOM"],
    "finance": ["AXP", "BAC", "BLK", "C", "GS", "JPM", "MA", "MS", "V", "WFC"],
    "healthcare": ["ABBV", "AMGN", "BMY", "DHR", "JNJ", "LLY", "MRK", "PFE", "TMO", "UNH"],
}


def scrape_quote(symbol: str) -> str | None:
    """BeautifulSoup: xác nhận mã tồn tại trên Yahoo, trả về tên công ty."""
    url = f"https://finance.yahoo.com/quote/{symbol}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)

        price = soup.find("fin-streamer", {"data-field": "regularMarketPrice"})
        if price:
            return symbol
    except requests.RequestException:
        return None
    return None


def download(symbol: str) -> pd.DataFrame | None:
    name = scrape_quote(symbol)
    if not name:
        return None

    df = yf.Ticker(symbol).history(start=START, end=YFINANCE_END, auto_adjust=False)
    if df.empty:
        return None

    df = df.reset_index()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })

    keep = ["Date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()

    end_dt = pd.to_datetime(END)
    df = df[(df["Date"] >= pd.to_datetime(START)) & (df["Date"] <= end_dt)]
    df = df.dropna(subset=["close"]).sort_values("Date").reset_index(drop=True)

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].round(3)
    df["volume"] = df["volume"].astype("int64")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sector", choices=SECTORS)
    p.add_argument("--symbol")
    args = p.parse_args()

    pairs = []
    for sector, symbols in SECTORS.items():
        if args.sector and sector != args.sector:
            continue
        for sym in symbols:
            if args.symbol and sym != args.symbol.upper():
                continue
            pairs.append((sector, sym))

    print(f"Crawl {len(pairs)} mã | {START} → {END}\n")

    for sector, sym in pairs:
        out = DATA_DIR / sector
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{sym}_1d_full.csv"

        print(f"[{sector}] {sym}...", end=" ", flush=True)
        time.sleep(0.5)  # tránh spam Yahoo khi dùng BeautifulSoup

        df = download(sym)
        if df is None:
            print("skip")
            continue

        df.to_csv(path, index=False, float_format="%.3f")
        print(f"ok ({len(df)} rows) → {path.name}")


if __name__ == "__main__":
    main()
