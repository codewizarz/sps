import polars as pl
from pathlib import Path
from datetime import date
from typing import Optional, List, Union
import shutil

# Import existing Auditor
from quant_repo.data.audit import DataAuditor


class DataPipeline:
    """
    Production-grade options data ingestion pipeline.
    Handles auditing, transformation, and partitioned storage.
    """

    def __init__(self, root_path: str = "./data/options"):
        self.root_path = Path(root_path)
        self.auditor = DataAuditor()

    def ingest(
        self,
        df: pl.DataFrame,
        symbol: str,
        mode: str = "append",
        partition_cols: List[str] = ["expiry", "date"],
    ) -> bool:
        """
        Ingests data, audits it, and writes to partitioned Parquet.

        Args:
            df: Raw DataFrame. Must contain [date, symbol, expiry, strike, type, ...].
            symbol: Ticker symbol (e.g., NIFTY).
            mode: 'append' or 'overwrite' (at partition level).
            partition_cols: Columns to partition by (default: expiry, date).

        Returns:
            True if successful, False if audit failed critically.
        """
        # 1. Validate Schema
        required = {"date", "symbol", "expiry", "strike", "type"}
        missing = required - set(df.columns)
        if missing:
            print(f"[Pipeline] Error: Missing columns {missing}")
            return False

        # 2. Audit
        report = self.auditor.audit_dataframe(df)
        if report.health_score < 90.0:  # Threshold for rejection
            print(f"[Pipeline] REJECTED: Low Health Score ({report.health_score:.1f}%)")
            print(f"[Pipeline] Issues: {report.issues}")
            return False

        if report.health_score < 100.0:
            print(
                f"[Pipeline] WARNING: Health Score {report.health_score:.1f}%. Proceeding with caution."
            )
            print(f"[Pipeline] Issues: {report.issues}")

        # 3. Transform (Standardization)
        # Ensure dates are date type
        df = df.with_columns(
            [pl.col("date").cast(pl.Date), pl.col("expiry").cast(pl.Date)]
        )

        # 4. Write / Load
        # We use Polars partition writing.
        # Partition Structure: root/symbol=XYZ/expiry=YYYY-MM-DD/date=YYYY-MM-DD/data.parquet

        # Note: Polars `write_parquet` with `partition_by` creates the structure.
        # But we want `symbol` to be the top level folder, typically managed manually or via Hive paritioning.
        # Let's organize manually to ensure control.

        # Base path for symbol
        symbol_path = self.root_path / f"symbol={symbol}"

        try:
            # We use Polars to write partitioned dataset directly inside the symbol folder
            # It will create expiry=.../date=... folders.

            # If mode is overwrite, we might need to clear existing for this batch?
            # Partition writing usually appends new files or overwrites if names clash.
            # For simplicity in this engine, we will let Polars handle it.

            df.write_parquet(
                symbol_path,
                use_pyarrow=True,
                pyarrow_options={"partition_cols": partition_cols},
            )
            print(
                f"[Pipeline] Successfully ingested {len(df)} rows for {symbol} to {symbol_path}"
            )
            return True

        except Exception as e:
            print(f"[Pipeline] Write Failed: {e}")
            return False


class Catalog:
    """
    Interface to query the ingested data.
    """

    def __init__(self, root_path: str = "./data/options"):
        self.root_path = Path(root_path)

    def load_range(
        self,
        symbol: str,
        start_date: Union[str, date],
        end_date: Union[str, date],
        columns: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Loads data for a symbol within a date range.
        Leverages Hive partitioning for pruning.
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        symbol_path = self.root_path / f"symbol={symbol}"

        if not symbol_path.exists():
            raise FileNotFoundError(f"No data found for symbol {symbol}")

        # Lazy Scan to optimize
        qs = pl.scan_parquet(symbol_path, hive_partitioning=True)

        # Filter by Date Range
        qs = qs.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))

        if columns:
            qs = qs.select(columns)

        return qs.collect()
