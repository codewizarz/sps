import requests
import polars as pl
import pandas as pd
from datetime import datetime
import time
import os
import random
from pathlib import Path
import json


class NSECollector:
    """
    Robust NSE Option Chain Collector.
    - Manages Session/Cookies.
    - Rotates User-Agents.
    - Saves snapshots to Parquet.
    """

    BASE_URL = "https://www.nseindia.com"
    API_URL = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    ]

    def __init__(self, output_dir: str = "./data/live_snapshots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.current_ua = ""
        self.refresh_session()

    def refresh_session(self):
        """Initializes session with fresh cookies and headers."""
        try:
            self.current_ua = random.choice(self.USER_AGENTS)
            self.session.headers.update(
                {
                    "User-Agent": self.current_ua,
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                }
            )

            # Hit homepage to get valid cookies
            print("[NSECollector] Refreshing Session (Homepage)...")
            response = self.session.get(self.BASE_URL, timeout=10)
            if response.status_code == 200:
                print("[NSECollector] Session Refreshed.")
            else:
                print(f"[NSECollector] Homepage failed: {response.status_code}")

        except Exception as e:
            print(f"[NSECollector] Session Refresh Error: {e}")

    def fetch_option_chain(self, symbol: str = "NIFTY") -> pd.DataFrame:
        """Fetches and parses the option chain."""
        try:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
            response = self.session.get(url, timeout=10)

            if response.status_code == 401:
                print("[NSECollector] 401 Unauthorized. Refreshing Session...")
                self.refresh_session()
                time.sleep(2)
                response = self.session.get(url, timeout=10)

            if response.status_code != 200:
                print(f"[NSECollector] API Failed: {response.status_code}")
                return pd.DataFrame()  # Empty

            data = response.json()
            return self._parse_json(data)

        except Exception as e:
            print(f"[NSECollector] Fetch Error: {e}")
            return pd.DataFrame()

    def _parse_json(self, json_data: dict) -> pd.DataFrame:
        """Parses raw NSE JSON into a flat DataFrame."""
        records = json_data.get("records", {})
        data = records.get("data", [])
        timestamp = records.get("timestamp")  # "10-Jan-2025 15:30:00"

        flat_rows = []
        parse_ts = datetime.now()  # Fallback

        try:
            if timestamp:
                parse_ts = datetime.strptime(timestamp, "%d-%b-%Y %H:%M:%S")
        except:
            pass

        for item in data:
            expiry = item.get("expiryDate")
            strike = item.get("strikePrice")

            # CE
            if "CE" in item:
                ce = item["CE"]
                flat_rows.append(
                    {
                        "timestamp": parse_ts,
                        "symbol": "NIFTY",
                        "expiry": expiry,
                        "strike": strike,
                        "type": "CE",
                        "bid": ce.get("bidprice", 0),
                        "ask": ce.get(
                            "askPrice", 0
                        ),  # Note: NSE uses askPrice casing sometimes
                        "last_price": ce.get("lastPrice", 0),
                        "iv": ce.get("impliedVolatility", 0),
                        "volume": ce.get("totalTradedVolume", 0),
                        "oi": ce.get("openInterest", 0),
                        "change_oi": ce.get("changeinOpenInterest", 0),
                    }
                )

            # PE
            if "PE" in item:
                pe = item["PE"]
                flat_rows.append(
                    {
                        "timestamp": parse_ts,
                        "symbol": "NIFTY",
                        "expiry": expiry,
                        "strike": strike,
                        "type": "PE",
                        "bid": pe.get("bidprice", 0),
                        "ask": pe.get("askPrice", 0),
                        "last_price": pe.get("lastPrice", 0),
                        "iv": pe.get("impliedVolatility", 0),
                        "volume": pe.get("totalTradedVolume", 0),
                        "oi": pe.get("openInterest", 0),
                        "change_oi": pe.get("changeinOpenInterest", 0),
                    }
                )

        return pd.DataFrame(flat_rows)

    def save_snapshot(self, df: pd.DataFrame):
        if df.empty:
            return

        # Structure: output_dir/YYYY-MM-DD/HH-MM-SS.parquet
        ts = df["timestamp"].iloc[0]
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H-%M-%S")

        day_dir = self.output_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        file_path = day_dir / f"{time_str}.parquet"

        # Use simple PyArrow write
        df.to_parquet(file_path, index=False)
        print(f"[NSECollector] Saved snapshot: {file_path} ({len(df)} rows)")

    def run_scheduler(self, interval=60):
        print(f"[NSECollector] Starting Poller (Interval: {interval}s)...")
        while True:
            try:
                start_time = time.time()
                df = self.fetch_option_chain("NIFTY")
                self.save_snapshot(df)

                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n[NSECollector] Stopping...")
                break
            except Exception as e:
                print(f"[NSECollector] Loop Error: {e}")
                time.sleep(10)
                self.refresh_session()


if __name__ == "__main__":
    # Test Run
    collector = NSECollector()
    # Mocking Requests if network is not available in Agent env?
    # Agent usually doesn't have open internet.
    # I will add a mock switch for testing.
    pass
