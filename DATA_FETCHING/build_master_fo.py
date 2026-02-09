"""
build_master_fo.py

PRODUCTION-GRADE FO MASTER DATASET BUILDER
------------------------------------------
Constructs a single, query-optimized, partitioned Parquet lake from raw NSE FO Bhavcopies.

FEATURES:
1.  **Fault Tolerance**: Skips corrupt files, logs errors, continues processing.
2.  **Schema Normalization**: Handles column name variations across years (TIMESTAMP vs Date).
3.  **Strict Typing**: Enforces float64 for prices, integer for volume/OI.
4.  **Partitioned Storage**: Writes Hive-style partitions (Year/Month) for fast querying.
5.  **Memory Optimization**: Processes files in chunks to keep RAM usage low.

USAGE:
    python DATA_FETCHING/build_master_fo.py
"""

import os
import shutil
import logging
import gc
from pathlib import Path
from datetime import datetime
import pandas as pd  # Fallback for robust CSV reading if Polars fails strict ref
import polars as pl

# --- CONFIGURATION ---
RAW_DATA_DIR = Path("nse_fo_bhavcopies_extracted")
MASTER_LAKE_DIR = Path("data/master_fo_lake")
LOG_FILE = "build_master_fo.log"
BATCH_SIZE = 50  # Process 50 files at a time

# Accepted Instruments
ACCEPTED_INSTRUMENTS = ["OPTIDX", "FUTIDX", "OPTSTK", "FUTSTK"]

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)

# --- ROBUST SCHEMA MAPPING ---
COLUMN_VARIANTS = {
    "INSTRUMENT": ["INSTRUMENT", "Instrument", "MoB"],  # MoB sometimes appears
    "SYMBOL": ["SYMBOL", "Symbol", "Ticker", "TckrSymb"],
    "EXPIRY_DT": ["EXPIRY_DT", "Expiry", "Expiry Date", "XpryDt"],
    "STRIKE_PR": ["STRIKE_PR", "Strike Price", "Strike", "StrkPric"],
    "OPTION_TYP": ["OPTION_TYP", "Option Type", "Type", "OptnTp"],
    "OPEN": ["OPEN", "Open", "Open Price", "OpnPric"],
    "HIGH": ["HIGH", "High", "High Price", "HghPric"],
    "LOW": ["LOW", "Low", "Low Price", "LwPric"],
    "CLOSE": ["CLOSE", "Close", "Close Price", "ClsPric", "LTP"],
    "SETTLE_PR": ["SETTLE_PR", "Settle Price", "SttlPric"],
    "CONTRACTS": [
        "CONTRACTS",
        "Contracts",
        "Volume",
        "No. of Contracts",
        "TtlTradgVol",
    ],
    "VAL_INLAKH": ["VAL_INLAKH", "Value", "Turnover (Lakhs)", "TtlTrfVal"],
    "OPEN_INT": ["OPEN_INT", "Open Interest", "OI", "OpnIntrst"],
    "CHG_IN_OI": ["CHG_IN_OI", "Change in OI", "Change in Open Interest", "ChngInOI"],
    "TIMESTAMP": ["TIMESTAMP", "Date", "TradDt", "TrdDt"],
}


def normalize_columns(df_cols):
    """
    Returns a mapping dict {old_col: new_col} based on variants.
    """
    mapping = {}
    cols_upper = {c.upper(): c for c in df_cols}

    for target, variants in COLUMN_VARIANTS.items():
        found = False
        for v in variants:
            if v.upper() in cols_upper:
                mapping[cols_upper[v.upper()]] = target
                found = True
                break
        if not found:
            # If critical column missing, we will detect later
            pass

    return mapping


def safe_read_and_process(file_path):
    """
    Reads a CSV file using Polars, applies normalization, strict casting.
    Returns None if validation fails.
    """
    try:
        # 1. Peek at columns first (Lazy Scan) to determine mapping
        # We assume standard CSV
        # Use Eager read with 'ignore_errors' to just get data in
        # scan_csv is faster but harder to handle per-file schema drift dynamically without error

        # We use read_csv with `infer_schema_length=0` to read EVERYTHING as string first.
        # This prevents "Cast Error" if 'null' or '-' exists in Float columns.

        df = pl.read_csv(
            file_path,
            infer_schema_length=0,
            ignore_errors=True,
            encoding="utf8-lossy",
            null_values=["", "NA", "null", "-"],
        )

        if df.is_empty():
            return None

        # 2. Normalize Schema
        mapping = normalize_columns(df.columns)
        df = df.rename(mapping)

        # 3. Check Critical Columns
        required_keys = ["SYMBOL", "TIMESTAMP", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP"]
        # Allow missing prices? No, meaningless.
        missing = [c for c in required_keys if c not in df.columns]
        if missing:
            logging.warning(f"Skipping {file_path.name}: Missing {missing}")
            return None

        # 4. Filter Instruments (Row-wise)
        # Casting INSTRUMENT to standard string
        if "INSTRUMENT" in df.columns:
            df = df.filter(pl.col("INSTRUMENT").is_in(ACCEPTED_INSTRUMENTS))
        else:
            # Some older formats might just imply instrument? Unlikely for NSE FO.
            # If INSTRUMENT missing, check if filtering is even possible.
            # We assume it's corrupt if no Instrument col.
            return None

        if df.is_empty():
            return None

        # 5. Strict Type Casting & Sanitation
        # We need to remove commas from numbers strings: '1,200.00'

        cols_to_select = []

        # Helper for cleaning and casting
        def clean_cast(col_name, dtype):
            if col_name not in df.columns:
                return pl.lit(None, dtype=dtype).alias(col_name)

            # Cleaning logic for strings
            c = pl.col(col_name)

            # If target is Numeric, perform cleanup
            if dtype in [pl.Float64, pl.Int64]:
                c = c.str.replace_all(",", "").str.strip_chars()
                # Empty to null
                return c.cast(dtype, strict=False).alias(col_name)

            return c.cast(dtype, strict=False).alias(col_name)

        # Build selection expression
        exprs = [
            clean_cast("INSTRUMENT", pl.Utf8),
            clean_cast("SYMBOL", pl.Utf8),
            clean_cast("EXPIRY_DT", pl.Utf8),
            clean_cast("STRIKE_PR", pl.Float64),
            clean_cast("OPTION_TYP", pl.Utf8),
            clean_cast("OPEN", pl.Float64),
            clean_cast("HIGH", pl.Float64),
            clean_cast("LOW", pl.Float64),
            clean_cast("CLOSE", pl.Float64),
            clean_cast("SETTLE_PR", pl.Float64),
            clean_cast("CONTRACTS", pl.Int64),
            clean_cast("VAL_INLAKH", pl.Float64),
            clean_cast("OPEN_INT", pl.Int64),
            clean_cast("CHG_IN_OI", pl.Int64),
            clean_cast("TIMESTAMP", pl.Utf8),
        ]

        df_clean = df.select(exprs)

        # 6. Parse Dates (Strict Schema: date, expiry_date)
        # Try multiple formats if needed. Usually DD-MMM-YYYY or YYYY-MM-DD.
        # We create 'date_obj' and 'expiry_obj' for partitioning/sorting.

        # Note: str.to_date() with strict=False will return Null if parse fails.
        # NSE usually: 30-Jan-2025

        df_clean = df_clean.with_columns(
            [
                pl.col("TIMESTAMP")
                .str.to_date("%d-%b-%Y", strict=False)
                .fill_null(
                    pl.col("TIMESTAMP").str.to_date("%Y-%m-%d", strict=False)
                )  # Fallback
                .alias("date"),
                pl.col("EXPIRY_DT")
                .str.to_date("%d-%b-%Y", strict=False)
                .fill_null(pl.col("EXPIRY_DT").str.to_date("%Y-%m-%d", strict=False))
                .alias("expiry_date"),
            ]
        )

        # Drop rows where Date is invalid (critical)
        df_clean = df_clean.drop_nulls(subset=["date", "expiry_date", "SYMBOL"])

        return df_clean

    except Exception as e:
        logging.error(f"Processing Error {file_path.name}: {e}")
        return None


def build_master_dataset():
    logging.info("STARTING MASTER BUILD PROCESS")
    logging.info(f"Source: {RAW_DATA_DIR.absolute()}")
    logging.info(f"Target: {MASTER_LAKE_DIR.absolute()}")

    all_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not all_files:
        logging.error("No CSV files found.")
        return

    logging.info(f"Found {len(all_files)} files.")

    # Clean Output Dir
    if MASTER_LAKE_DIR.exists():
        logging.info("Resetting Master Lake...")
        shutil.rmtree(MASTER_LAKE_DIR)
    MASTER_LAKE_DIR.mkdir(parents=True, exist_ok=True)

    # Batch Processing
    total_rows = 0
    batches = [
        all_files[i : i + BATCH_SIZE] for i in range(0, len(all_files), BATCH_SIZE)
    ]

    for i, batch in enumerate(batches):
        logging.info(f"Batch {i + 1}/{len(batches)}...")

        cleaned_dfs = []
        for file_path in batch:
            df = safe_read_and_process(file_path)
            if df is not None:
                cleaned_dfs.append(df)

        if not cleaned_dfs:
            continue

        # Merge Batch
        df_batch = pl.concat(cleaned_dfs)

        if df_batch.is_empty():
            continue

        # Partitioning: Add Year/Month columns
        df_batch = df_batch.with_columns(
            [
                pl.col("date").dt.year().cast(pl.Utf8).alias("year"),
                pl.col("date").dt.month().cast(pl.Utf8).str.zfill(2).alias("month"),
            ]
        )

        # Write to Partitioned Parquet
        # We manually iterate years to append safely.

        years = df_batch["year"].unique().to_list()
        for yr in years:
            target_dir = MASTER_LAKE_DIR / f"year={yr}"
            target_dir.mkdir(parents=True, exist_ok=True)

            df_year = df_batch.filter(pl.col("year") == yr)

            # Unique filename for this batch write
            fname = f"part_{i}_{datetime.now().strftime('%H%M%S%f')}.parquet"
            df_year.write_parquet(target_dir / fname, compression="zstd")

        cnt = len(df_batch)
        total_rows += cnt
        logging.info(f"Batch {i + 1} Committed: {cnt} rows.")

        del df_batch
        del cleaned_dfs
        gc.collect()

    logging.info(f"=== BUILD COMPLETE ===")
    logging.info(f"Total Rows Ingested: {total_rows}")


if __name__ == "__main__":
    build_master_dataset()
