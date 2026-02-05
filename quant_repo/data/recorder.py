import polars as pl
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import time


class MarketRecorder:
    """
    Captures high-frequency tick/quote data and buffers it to disk.
    Designed to build a proprietary long-term option history.
    """

    def __init__(self, data_dir: Path, buffer_size: int = 100):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw" / "options_tick"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.buffer: List[Dict] = []
        self.buffer_size = buffer_size

        # Partitioning: date=YYYY-MM-DD

    def on_tick(self, tick: Dict):
        """
        Ingest a single tick/quote.
        tick format: {
            "timestamp": datetime,
            "symbol": str,
            "bid": float,
            "ask": float,
            "last": float,
            "volume": int,
            "oi": int,
            "greeks": {delta, gamma, etc...} (Optional)
        }
        """
        self.buffer.append(tick)

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """
        Writes buffer to Parquet.
        Strategies:
        1. Group by Date/Symbol.
        2. Append to existing or write new chunk.
        """
        if not self.buffer:
            return

        df = pl.DataFrame(self.buffer)

        # Add partition columns if missing
        if "date" not in df.columns:
            # Assuming timestamp is datetime
            df = df.with_columns(
                pl.col("timestamp").dt.date().cast(pl.String).alias("date")
            )

        # Group by date to handle cross-day buffers easily (though usually live is same day)
        partitions = df.partition_by("date")

        for part in partitions:
            date_str = part["date"][0]
            # Output path: data/raw/options_tick/date=2025-01-01/
            part_path = self.raw_dir / f"date={date_str}"
            part_path.mkdir(parents=True, exist_ok=True)

            # Filename: timestamp_chunk.parquet
            # Use ns timestamp to avoid collisions
            fname = f"{time.time_ns()}.parquet"
            save_path = part_path / fname

            # Drop partition col from file content (optimization, since it's in folder name)
            part.drop("date").write_parquet(save_path)

        self.buffer = []
        # print(f"[Recorder] Flushed to {self.raw_dir}")

    def consolidate(self, date_str: str):
        """
        Maintenance job: Merge small chunks into one large file per day/symbol.
        Call this End-Of-Day.
        """
        part_path = self.raw_dir / f"date={date_str}"
        if not part_path.exists():
            return

        # Scan all parquet files
        try:
            df = pl.read_parquet(str(part_path / "*.parquet"))

            # Sort by time
            df = df.sort("timestamp")

            # Write Consolidated
            consolidated_path = part_path / "consolidated.parquet"
            df.write_parquet(consolidated_path)

            # Cleanup chunks (Optional: in prod careful with deletion)
            # For now, we just create the optimized file.

            return consolidated_path
        except Exception as e:
            print(f"Consolidation failed: {e}")
            return None
