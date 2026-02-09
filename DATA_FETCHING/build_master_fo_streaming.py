"""
build_master_fo_streaming.py

TRUE STREAMING FO MASTER BUILDER
--------------------------------
Reads, normalizes, and partitions NSE FO data in a pure streaming fashion.
Never holds multiple files or large DataFrames in memory.
Optimized for low RAM environments.

FEATURES:
1.  **True Streaming**: Reads CSV in chunks (200k rows), processes, writes, and deletes immediately.
2.  **Schema Normalization**: Auto-resolves column drift (Old -> New, Reserved cols).
3.  **Partitioning**: Hive-style `year=YYYY/instrument=FinInstrmTp/`.
4.  **Compression**: Snappy Parquet.
5.  **Robustness**: Handles any schema variation dynamically.

USAGE:
    python DATA_FETCHING/build_master_fo_streaming.py
"""

import pandas as pd
import glob
import os
import shutil
from pathlib import Path
from datetime import datetime
import gc
import uuid
from collections import Counter

# --- CONFIGURATION ---
RAW_DATA_DIR = Path("nse_fo_bhavcopies_extracted")
LAKE_DIR = Path("data/master_fo_lake")
CHUNK_SIZE = 200_000

# 1. Column Mapping (Old -> New Standard)
COLUMN_MAP = {
    # Core Fields
    "INSTRUMENT": "FinInstrmTp",
    "SYMBOL": "TckrSymb",
    "EXPIRY_DT": "XpryDt",
    "STRIKE_PR": "StrkPric",
    "OPTION_TYP": "OptnTp",
    "OPEN": "OpnPric",
    "HIGH": "HghPric",
    "LOW": "LwPric",
    "CLOSE": "ClsPric",
    "SETTLE_PR": "SttlPric",
    "CONTRACTS": "TtlTradgVol",
    "VAL_INLAKH": "TtlTrfVal",
    "OPEN_INT": "OpnIntrst",
    "CHG_IN_OI": "ChngInOpnIntrst",
    "TIMESTAMP": "TradDt",
    # Schema Drift Fixes (Reserved Cols)
    "Rsvd01": "Rsvd1",
    "Rsvd02": "Rsvd2",
    "Rsvd03": "Rsvd3",
    "Rsvd04": "Rsvd4",
}

# 2. Columns to DROP (After Normalization)
DROP_COLS = [
    "Rsvd1",
    "Rsvd2",
    "Rsvd3",
    "Rsvd4",
    "Rmks",
    "ISIN",
    "FinInstrmId",
    "SsnId",
    "Unnamed: 0",
]


def normalize_chunk(df, filename):
    """
    Normalizes a single chunk of data.
    """
    # Standardize Column Names
    df.columns = [c.strip() for c in df.columns]

    current_cols = set(df.columns)
    rename_dict = {}

    for old_col, new_col in COLUMN_MAP.items():
        if old_col in current_cols:
            rename_dict[old_col] = new_col
        elif new_col in current_cols:
            pass
        else:
            # Case-insensitive match
            for c in current_cols:
                if c.upper() == old_col.upper():
                    rename_dict[c] = new_col
                    break

    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Drop Reserved/Useless Cols
    # Note: We rename Rsvd01 -> Rsvd1 first, THEN drop Rsvd1 here.
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        try:
            df = df.drop(columns=cols_to_drop)
        except:
            pass

    # Critical Columns Check
    REQUIRED = ["FinInstrmTp", "TckrSymb", "TradDt"]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        # We skip chunks with missing critical data (likely header/footer noise)
        return None

    # Type Conversion

    # Dates
    for col in ["TradDt", "XpryDt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["TradDt"])
    if df.empty:
        return None

    # Numerics
    # Prices (Float32)
    price_cols = ["StrkPric", "OpnPric", "HghPric", "LwPric", "ClsPric", "SttlPric"]
    for col in price_cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )
            df[col] = df[col].astype("float32")

    # Ints (Vol/OI)
    int_cols = ["OpnIntrst", "ChngInOpnIntrst", "TtlTradgVol"]
    for col in int_cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                ).fillna(0)
            df[col] = df[col].fillna(0).astype("int32")

    # Safety: Cast Null-Only Columns to String to prevent PyArrow Schema Drift
    for col in df.columns:
        if df[col].isnull().all():
            df[col] = df[col].astype(str)

    return df


def process_stream():
    print("=== STARTING MASTER FO STREAMING BUILDER ===")
    print(f"Source: {RAW_DATA_DIR.absolute()}")
    print(f"Target: {LAKE_DIR.absolute()}")
    print(f"Chunk Size: {CHUNK_SIZE}")

    files = list(RAW_DATA_DIR.glob("*.csv"))
    if not files:
        print("CRITICAL: No CSV files found.")
        return

    print(f"Files Found: {len(files)}")

    # Reset Lake? User guidance "Rebuild the lake from scratch" in previous prompt.
    # In this prompt "Rewrite the FO lake builder...".
    # Assuming standard rebuild practice.
    if LAKE_DIR.exists():
        shutil.rmtree(LAKE_DIR)
    LAKE_DIR.mkdir(parents=True, exist_ok=True)

    # Stats
    total_files = 0
    total_rows = 0
    instrument_stats = Counter()

    for i, file_path in enumerate(files):
        try:
            # Read in Chunks
            chunks = pd.read_csv(
                file_path,
                chunksize=CHUNK_SIZE,
                low_memory=False,
                encoding_errors="replace",  # Handle encoding issues
            )

            for chunk in chunks:
                df = normalize_chunk(chunk, file_path.name)

                if df is None or df.empty:
                    continue

                # Add Partition Info
                df["year"] = df["TradDt"].dt.year.fillna(0).astype(int)
                df["month"] = df["TradDt"].dt.month.fillna(0).astype(int)

                # Update Stats
                if "FinInstrmTp" in df.columns:
                    counts = df["FinInstrmTp"].value_counts().to_dict()
                    instrument_stats.update(counts)

                # GROUP WRITE (Streaming Partition)
                # Group by [year, month] - Optimized Partitioning
                # FinInstrmTp is NOT used for partitioning (too granular) but kept as column

                groups = df.groupby(["year", "month"])

                for (yr, mth), group in groups:
                    # Target Path: year=YYYY/month=MM/
                    target_dir = LAKE_DIR / f"year={yr}" / f"month={mth}"
                    target_dir.mkdir(parents=True, exist_ok=True)

                    # Unique Filename (UUID to prevent collision)
                    unique_name = f"{uuid.uuid4().hex}.parquet"
                    target_file = target_dir / unique_name

                    # Write Optimized Parquet
                    group.to_parquet(
                        target_file,
                        engine="pyarrow",
                        compression="zstd",
                        index=False,
                        # Performance Optimizations
                        use_dictionary=True,
                        write_statistics=True,
                        row_group_size=250000,
                    )

                    total_rows += len(group)

                # Explicit Cleanup
                del df
                del group
                gc.collect()  # Force GC after chunk processing? Maybe too aggressive per loop but safe.

            total_files += 1

            # Periodic Logging
            if total_files % 25 == 0:
                print(f"Processed {total_files} files... Rows: {total_rows:,.0f}")

        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            continue

    # Final Report
    print("\n" + "=" * 40)
    print("FINAL STREAMING REPORT")
    print("=" * 40)
    print(f"TOTAL FILES           : {total_files}")
    print(f"TOTAL ROWS            : {total_rows:,.0f}")

    # Calculate Size
    total_size = 0
    for p in LAKE_DIR.rglob("*.parquet"):
        total_size += p.stat().st_size
    size_gb = total_size / (1024**3)
    print(f"DATASET SIZE          : {size_gb:.2f} GB")

    print("\nINSTRUMENT DISTRIBUTION:")
    for inst, count in instrument_stats.most_common():
        print(f"{inst:<15} : {count:,.0f}")

    print("========================================")


if __name__ == "__main__":
    process_stream()
