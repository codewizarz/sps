import polars as pl
from pathlib import Path
from typing import Iterator, List
import pandas as pd

from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.core.datetime import dt_to_unix_nanos


class LazyOptionsOptionsLoader:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def load_ticks(
        self, instrument: Instrument, start_dt: str, end_dt: str
    ) -> Iterator[QuoteTick]:
        """
        Lazy generator yielding QuoteTicks for a single instrument or batch.
        """
        # Resolve path based on partitioning
        # Assuming structure: root/symbol/expiry/date/data.parquet
        # We need to filter for specific Instrument parameters

        start_ns = dt_to_unix_nanos(pd.Timestamp(start_dt))
        end_ns = dt_to_unix_nanos(pd.Timestamp(end_dt))

        # Parse Instrument ID to get meta: NIFTY-20240125-21000-CE.NSE
        parts = instrument.id.value.split(".")[0].split("-")
        symbol = parts[0]
        strike = float(parts[2])
        right_code = parts[3]
        right = (
            "CE" if right_code == "CE" else "PE"
        )  # Mock data uses CE/PE directly usually

        q = pl.scan_parquet(self.dataset_path / "**" / "*.parquet")

        # Filter Logic
        q = q.filter(
            (pl.col("timestamp") >= start_ns)
            & (pl.col("timestamp") <= end_ns)
            & (pl.col("strike") == strike)
            & (pl.col("right") == right)
            & (pl.col("symbol") == symbol)
        )

        # Sort is crucial for streaming
        q = q.sort("timestamp")

        # Yield Chunks
        for chunk in q.collect(streaming=True).iter_rows(named=True):
            yield self._row_to_tick(chunk, instrument)

    def _row_to_tick(self, row: dict, instrument: Instrument) -> QuoteTick:
        # Convert row to QuoteTick
        # Timestamp must be uint64 nanos
        ts = row["timestamp"]
        if isinstance(ts, pd.Timestamp):
            ts = dt_to_unix_nanos(ts)
        elif isinstance(ts, int):
            pass  # Assume nanos

        return QuoteTick(
            instrument_id=instrument.id,
            bid_price=instrument.make_price(row["bid"]),
            ask_price=instrument.make_price(row["ask"]),
            bid_size=instrument.make_qty(1),  # Default or mapped
            ask_size=instrument.make_qty(1),
            ts_event=ts,
            ts_init=ts,
        )
