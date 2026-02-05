import pandas as pd
import polars as pl
from pathlib import Path
from typing import Optional, List, Union

from nautilus_trader.model.data import QuoteTick, Bar, BarType
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler, BarDataWrangler
from nautilus_trader.model.instruments import Instrument


class ParquetDataLoader:
    """
    Standardized loader for Parquet data.
    """

    def __init__(self, data_root: Path):
        self.data_root = data_root

    def load_quotes_dataframe(
        self,
        instrument_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load quotes as a Pandas DataFrame.
        Expected schema: [timestamp, bid, ask, bid_size, ask_size]
        """
        path = self._get_path(instrument_id, "quotes")
        if not path.exists():
            raise FileNotFoundError(f"No quote data found for {instrument_id}")

        # Use Polars for efficient filtering
        q = pl.scan_parquet(path)

        if start_date:
            q = q.filter(pl.col("timestamp") >= pl.lit(pd.Timestamp(start_date)))
        if end_date:
            q = q.filter(pl.col("timestamp") <= pl.lit(pd.Timestamp(end_date)))

        return q.collect().to_pandas().set_index("timestamp").sort_index()

    def load_nautilus_quotes(
        self,
        instrument: Instrument,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[QuoteTick]:
        """
        Load quotes as Nautilus QuoteTick objects.
        """
        df = self.load_quotes_dataframe(str(instrument.id), start_date, end_date)
        wrangler = QuoteTickDataWrangler(instrument)
        # Ensure columns match what wrangler expects or rename
        # Wrangler expects: index=timestamp, default cols: bid, ask, bid_size, ask_size
        return wrangler.process(df)

    def _get_path(self, instrument_id: str, data_type: str) -> Path:
        """
        Resolve file path. Structure: data_root / type / instrument.parquet
        """
        safe_id = instrument_id.replace("/", "_")
        return self.data_root / data_type / f"{safe_id}.parquet"
