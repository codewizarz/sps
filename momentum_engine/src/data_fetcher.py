import os
import logging
from typing import List, Optional
import pandas as pd
import yfinance as yf

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IndianDataFetcher:
    """
    A unified data fetcher for Indian equity and index data.
    Designed following the 'tool-design' principles:
    - Clear inputs and expected outputs
    - Consolidation of related functionality into a single class interface
    - Robust error handling for agent recovery

    Data is stored to the filesystem based on 'filesystem-context' principles,
    enabling caching and avoiding rapid recalculations or excessive API calls.
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Args:
            data_dir: Directory where the historical data CSVs will be stored.
        """
        self.data_dir = data_dir
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_nifty_symbols(self) -> List[str]:
        """
        Returns a static reliable list of Nifty 50 stock symbols (with .NS appended for Yahoo Finance).

        Returns:
            A list of string tickers.
        """
        logger.info("Fetching Nifty 50 symbols (static list)...")
        # Static list for current Nifty 50 constituents (example set)
        nifty_50_base = [
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "ICICIBANK",
            "INFY",
            "ITC",
            "SBIN",
            "LARSEN",
            "BAJFINANCE",
            "KOTAKBANK",
            "BHARTIARTL",
            "AXISBANK",
            "ASIANPAINT",
            "HINDUNILVR",
            "SUNPHARMA",
            "MARUTI",
            "TITAN",
            "BAJAJFINSV",
            "TATASTEEL",
            "ULTRACEMCO",
            "M&M",
            "WIPRO",
            "NTPC",
            "HCLTECH",
            "POWERGRID",
            "ADANIENT",
            "ADANIPORTS",
            "TATAMOTORS",
            "TECHM",
            "ONGC",
            "HINDALCO",
            "GRASIM",
            "JSWSTEEL",
            "COALINDIA",
            "BAJAJ-AUTO",
            "CIPLA",
            "APOLLOHOSP",
            "TATACONSUM",
            "DRREDDY",
            "BRITANNIA",
            "DIVISLAB",
            "HEROMOTOCO",
            "EICHERMOT",
            "BPCL",
            "INDUSINDBK",
            "SHREECEM",
            "NESTLEIND",
            "LTIM",
            # Note: A few might have different yfinance tickers (e.g. LTIM.NS, M&M.NS -> MM.NS)
        ]
        return [f"{sym.replace('&', '')}.NS" for sym in nifty_50_base]

    def fetch_historical(
        self, symbol: str, start_date: str, end_date: Optional[str] = None
    ) -> Optional[str]:
        """
        Fetches daily OHLCV historical data using yfinance for a symbol, cleans it,
        and saves it to the filesystem as a CSV.

        Args:
            symbol: Ticker symbol (e.g., 'RELIANCE.NS')
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD' (optional)

        Returns:
            The file path where data was saved, or None if failed.
        """
        logger.info(
            f"Fetching historical data for {symbol} from {start_date} to {end_date or 'today'}"
        )

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Clean data - basic handling
            df = df.dropna(subset=["Close"])

            # Save to 'data/raw/{symbol}.csv'
            # Replace invalid filename characters just in case
            safe_symbol = symbol.replace("^", "").replace(".", "_")
            filepath = os.path.join(self.data_dir, f"{safe_symbol}.csv")

            df.to_csv(filepath)
            logger.info(f"Data for {symbol} saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    def fetch_nifty_index_data(
        self, start_date: str, end_date: Optional[str] = None
    ) -> Optional[str]:
        """
        Fetches ^NSEI data for benchmark comparisons.

        Args:
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD' (optional)

        Returns:
            The file path where index data was saved.
        """
        return self.fetch_historical("^NSEI", start_date, end_date)
