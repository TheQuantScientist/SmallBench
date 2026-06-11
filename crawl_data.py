"""
Crawl stock OHLCV data via yfinance.

Date range: TEST_START_DATE → TEST_END_DATE (khớp main.py)
  2026-01-01 → 2026-04-17

Output: data/{sector}/{SYMBOL}_1d_full.csv
Format: Date,open,high,low,close,volume

Usage:
  python crawl_data.py
  python crawl_data.py --sector tech
  python crawl_data.py --symbol AAPL
"""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

TEST_START_DATE = "2025-05-30"
TEST_END_DATE = "2026-04-17"
# yfinance end date is exclusive → +1 day để lấy đủ 17/04/2026
YFINANCE_END = "2026-04-18"

SECTORS = {
    "tech": ["AAPL", "ADBE", "AVGO", "CRM", "CSCO", "GOOGL", "INTC", "MSFT", "NVDA", "ORCL"],
    "energy": ["COP", "CVX", "EOG", "EPD", "ET", "KMI", "SLB", "VLO", "WMB", "XOM"],
    "finance": ["AXP", "BAC", "BLK", "C", "GS", "JPM", "MA", "MS", "V", "WFC"],
    "healthcare": ["ABBV", "AMGN", "BMY", "DHR", "JNJ", "LLY", "MRK", "PFE", "TMO", "UNH"],
}


def fetch_symbol(symbol: str, start: str, end: str) -> pd.DataFrame:
    stock = yf.Ticker(symbol)
    df = stock.history(start=start, end=end, auto_adjust=False)

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    cols = ["Date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]].copy()

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(TEST_END_DATE)
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
    df = df.dropna(subset=["Date", "close"]).sort_values("Date").reset_index(drop=True)

    return df


def crawl_sector(sector: str, symbols: list[str], start: str, end: str) -> dict[str, int]:
    out_dir = DATA_DIR / sector
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    for symbol in symbols:
        print(f"  [{sector}] {symbol}...", end=" ", flush=True)
        try:
            df = fetch_symbol(symbol, start, end)
            if df.empty:
                print("SKIP (no data)")
                saved[symbol] = 0
                continue

            path = out_dir / f"{symbol}_1d_full.csv"
            df.to_csv(path, index=False, float_format="%.4f")
            saved[symbol] = len(df)
            print(f"OK ({len(df)} rows) → {path.name}")

        except Exception as e:
            print(f"ERROR: {e}")
            saved[symbol] = 0

    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="Crawl stock data for SmallBench pipeline")
    parser.add_argument("--sector", choices=list(SECTORS.keys()), default=None)
    parser.add_argument("--symbol", default=None, help="e.g. AAPL")
    return parser.parse_args()


def main():
    args = parse_args()

    sectors = {args.sector: SECTORS[args.sector]} if args.sector else SECTORS

    print("=" * 60)
    print("  SmallBench — Crawl Data")
    print("=" * 60)
    print(f"  Range: {TEST_START_DATE} → {TEST_END_DATE}")
    print(f"  Output: {DATA_DIR}/{{sector}}/{{SYMBOL}}_1d_full.csv")
    print("=" * 60)

    total_ok = 0
    total_rows = 0

    for sector, symbols in sectors.items():
        if args.symbol:
            sym = args.symbol.upper()
            if sym not in symbols:
                continue
            symbols = [sym]

        print(f"\n── Sector: {sector} ({len(symbols)} symbols) ──")
        results = crawl_sector(sector, symbols, TEST_START_DATE, YFINANCE_END)

        for sym, n in results.items():
            if n > 0:
                total_ok += 1
                total_rows += n

    print(f"\n{'=' * 60}")
    print(f"  Done: {total_ok} files saved, {total_rows} total rows")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
