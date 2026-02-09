import requests
import polars as pl
import pandas as pd
from datetime import date, timedelta, datetime
import io
import zipfile
from pathlib import Path
import time
import random


class BhavcopyLoader:
    """
    Ingests historical NSE FO Bhavcopy data.
    Handles downloading, extraction, schema normalization, and partitioned storage.
    """

    # Base URL pattern: https://archives.nseindia.com/content/historical/DERIVATIVES/2023/JAN/fo01JAN2023bhav.csv.zip
    BASE_URL = "https://archives.nseindia.com/content/historical/DERIVATIVES"

    def __init__(self, root_path: str = "./data/history"):
        self.root_path = Path(root_path)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def download_and_process_date(self, trade_date: date) -> bool:
        """
        Downloads and processes bhavcopy for a specific date.
        Returns True if successful, False if failure (or holiday).
        """
        # Construct URL
        year = trade_date.strftime("%Y")
        mon = trade_date.strftime("%b").upper()  # JAN, FEB
        day_str = trade_date.strftime("%d")  # 01
        mon_str = list(mon)  # J, A, N

        # URL Format: .../2023/JAN/fo01JAN2023bhav.csv.zip
        filename = f"fo{day_str}{mon}{year}bhav.csv.zip"
        url = f"{self.BASE_URL}/{year}/{mon}/{filename}"

        try:
            print(f"[Bhavcopy] Fetching {url}...")
            response = self.session.get(url, timeout=15)

            if response.status_code == 404:
                print(
                    f"[Bhavcopy] 404 Not Found (Likely Holiday/Weekend): {trade_date}"
                )
                return False

            if response.status_code != 200:
                print(f"[Bhavcopy] Error {response.status_code}")
                return False

            # Process Zip
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Expect one csv inside
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    # Use Polars to read CSV
                    df = pl.read_csv(f.read(), ignore_errors=True)

            # Normalize
            df_cleaned = self._normalize_schema(df)

            # Save
            self._save_to_lake(df_cleaned, trade_date)
            return True

        except Exception as e:
            print(f"[Bhavcopy] Failed processing {trade_date}: {e}")
            return False

    def _normalize_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Maps historical column names to standard schema.
        Standard: [date, symbol, expiry, strike, type, open, high, low, close, volume, oi]
        """
        # Common variations in NSE history
        col_map = {
            "TIMESTAMP": "date",
            "Date": "date",
            "SYMBOL": "symbol",
            "Symbol": "symbol",
            "EXPIRY_DT": "expiry",
            "Expiry": "expiry",
            "STRIKE_PR": "strike",
            "Strike Price": "strike",
            "OPTION_TYP": "type",
            "Option Type": "type",
            "OPEN": "open",
            "Open": "open",
            "HIGH": "high",
            "High": "high",
            "LOW": "low",
            "Low": "low",
            "CLOSE": "close",
            "Close": "close",
            "CONTRACTS": "volume",  # In older files, CONTRACTS was volume derivative key
            "No. of Contracts": "volume",
            "OPEN_INT": "oi",
            "Open Interest": "oi",
        }

        # Rename available columns
        existing_cols = df.columns
        rename_dict = {k: v for k, v in col_map.items() if k in existing_cols}
        df = df.rename(rename_dict)

        # Filter for Options only (Option TypeXX is Future)
        # Type is strictly CE/PE. Futures are usually marked XX or handled separately.
        # We want Options.
        if "type" in df.columns:
            df = df.filter(pl.col("type").is_in(["CE", "PE"]))

        # Parse Dates
        # NSE Date format matches: 02-JAN-2023
        try:
            df = df.with_columns(
                [
                    pl.col("date").str.strptime(pl.Date, "%d-%b-%Y"),
                    pl.col("expiry").str.strptime(pl.Date, "%d-%b-%Y"),
                ]
            )
        except:
            # Try other format if failed? mostly DD-MMM-YYYY
            pass

        # Select and Cast standard columns
        required = [
            "date",
            "symbol",
            "expiry",
            "strike",
            "type",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "oi",
        ]

        # Ensure all exist (fill 0 if volume/oi missing in really old files?)
        # But broadly they exist.

        # Type cast
        df = df.select(
            [
                pl.col("date"),
                pl.col("symbol"),
                pl.col("expiry"),
                pl.col("strike").cast(pl.Float64),
                pl.col("type"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Int64),
                pl.col("oi").cast(pl.Int64),
            ]
        )

        return df

    def _save_to_lake(self, df: pl.DataFrame, date_val: date):
        """
        Saves partitioned by Year/Month mostly, or Symbol?
        For historical backtesting, partitioning by Symbol is best for query speed on one ticker.
        Partitioning by Date is best for daily replay.

        Design Doc said: `symbol=.../year=...`
        """
        # We will iterate unique symbols and append?
        # Writing millions of rows partitioned by symbol locally in one go is heavy.
        # Ideally we write by DATE partition first (easy ingestion),
        # and checking the Design Doc: "Write to data/history/symbol=.../year=..."

        # Let's write by Date for now as 'Raw Staging',
        # re-partitioning to Symbol is a separate ETL usually.
        # BUT, the user asked for "support fast querying". Symbol partition is faster for "Give me NIFTY history".

        # Optim: Write entire day as one parquet? Then a separate job organizes?
        # Let's try writing Partitioned by Symbol strictly for NIFTY/BANKNIFTY (Major indices).

        target_symbols = ["NIFTY", "BANKNIFTY"]
        df_filtered = df.filter(pl.col("symbol").is_in(target_symbols))

        if df_filtered.is_empty():
            return

        # Write
        # Root/symbol=NIFTY/year=2023/data.parquet
        year_str = str(date_val.year)

        for sym in target_symbols:
            curr = df_filtered.filter(pl.col("symbol") == sym)
            if curr.is_empty():
                continue

            out_path = self.root_path / f"symbol={sym}" / f"year={year_str}"
            out_path.mkdir(parents=True, exist_ok=True)

            # File name: date.parquet
            fname = f"{date_val.strftime('%Y-%m-%d')}.parquet"
            curr.write_parquet(out_path / fname)

        print(f"[Bhavcopy] Saved {len(df_filtered)} rows for {target_symbols}")

    def download_range(self, start_date: date, end_date: date):
        print(f"Starting Batch Download: {start_date} to {end_date}")
        curr = start_date
        while curr <= end_date:
            if curr.weekday() < 5:  # Mon-Fri
                self.download_and_process_date(curr)
                time.sleep(random.uniform(0.1, 0.5))  # Polite delay
            curr += timedelta(days=1)
