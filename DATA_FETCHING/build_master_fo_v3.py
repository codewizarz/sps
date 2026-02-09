"""
build_master_fo_v3.py

PRODUCTION-GRADE FO MASTER BUILDER (V3)
---------------------------------------
Rebuilds the Master Lake with FULL INGESTION (No Instrument Filter).
Purpose: Diagnose instrument codes (e.g., STO vs OPTSTK) and capture all data.

FEATURES:
1.  **Strict Normalization**: Renames old columns (INSTRUMENT -> FinInstrmTp).
2.  **No Filtering**: Ingests ALL instruments to debug codes.
3.  **Type Enforcement**: Dates (datetime64), Prices (float32), OI/Vol (int32).
4.  **Partitioning**: Writes to data/master_fo_lake/year=YYYY/month=MM/ (Snappy).
5.  **Statistics**: Prints full distribution of FinInstrmTp at the end.

USAGE:
    python DATA_FETCHING/build_master_fo_v3.py
"""

import pandas as pd
import glob
import os
import shutil
from pathlib import Path
from datetime import datetime
import gc
from collections import Counter

# --- CONFIGURATION ---
RAW_DATA_DIR = Path("nse_fo_bhavcopies_extracted")
LAKE_DIR = Path("data/master_fo_lake")

# 1. Column Mapping (Old -> New Standard)
COLUMN_MAP = {
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
}

# 2. Columns to DROP
DROP_COLS = [
    "Rsvd1",
    "Rsvd2",
    "Rsvd3",
    "Rsvd4",
    "Rsvd01",
    "Rsvd02",
    "Rsvd03",
    "Rsvd04",
    "Rmks",
    "ISIN",
    "FinInstrmId",
    "SsnId",
    "Unnamed: 0",
]


def normalize_and_clean(df, filename):
    """
    Applies strict normalization, renaming, and type casting.
    Raises ValueError if critical conditions fail.
    """

    # A. Rename Columns (Handle variations)
    df.columns = [c.strip() for c in df.columns]

    current_cols = set(df.columns)
    rename_dict = {}

    for old_col, new_col in COLUMN_MAP.items():
        if old_col in current_cols:
            rename_dict[old_col] = new_col
        elif new_col in current_cols:
            pass
        else:
            for c in current_cols:
                if c.upper() == old_col.upper():
                    rename_dict[c] = new_col
                    break

    if rename_dict:
        df = df.rename(columns=rename_dict)

    # B. Validate Required Columns
    # Critical: FinInstrmTp, TckrSymb, TradDt
    REQUIRED = ["FinInstrmTp", "TckrSymb", "TradDt"]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        # Some older files might vary significantly.
        # But for quantitative work, we need tickers and dates.
        raise ValueError(f"Missing Critical Columns: {missing} in file {filename}")

    # C. Drop Useless Columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # D. NO FILTERING - Capture All Instruments

    # E. Strict Type Enforcing

    # 1. Dates
    for date_col in ["TradDt", "XpryDt"]:
        if date_col in df.columns:
            # Try multiple formats if needed, but errors='coerce' handles bad data
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop Invalid Dates? No, preserve raw if TradDt is valid.
    df = df.dropna(subset=["TradDt"])

    # 2. Prices (Float32)
    price_cols = ["StrkPric", "OpnPric", "HghPric", "LwPric", "ClsPric", "SttlPric"]
    for col in price_cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.replace(",", "", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].astype("float32")

    # 3. Volume / OI (Int32)
    int_cols = ["OpnIntrst", "ChngInOpnIntrst", "TtlTradgVol"]
    for col in int_cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.replace(",", "", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = df[col].fillna(0).astype("int32")

    return df


def process_lake():
    print("=== STARTING MASTER FO BUILDER (V3 - NO FILTER) ===")
    print(f"Source: {RAW_DATA_DIR.absolute()}")
    print(f"Target: {LAKE_DIR.absolute()}")

    files = list(RAW_DATA_DIR.glob("*.csv"))
    if not files:
        print("CRITICAL: No CSV files found.")
        return

    print(f"Files Found: {len(files)}")

    # Reset Lake
    if LAKE_DIR.exists():
        shutil.rmtree(LAKE_DIR)
    LAKE_DIR.mkdir(parents=True, exist_ok=True)

    # Metrics
    instrument_counts = Counter()
    total_rows_ingested = 0
    BATCH_SIZE = 50

    batches = [files[i : i + BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]

    for i, batch in enumerate(batches):
        print(f"Processing Batch {i + 1}/{len(batches)}...")

        batch_dfs = []
        for f in batch:
            try:
                df = pd.read_csv(f, low_memory=False)
                df_clean = normalize_and_clean(df, f.name)

                if df_clean is not None and not df_clean.empty:
                    batch_dfs.append(df_clean)

            except Exception as e:
                print(f"❌ FAIL: {f.name} -> {e}")
                # We skip individual bad files but continue batch

        if not batch_dfs:
            continue

        # Combine Batch
        master_batch = pd.concat(batch_dfs, ignore_index=True)

        # Capture Statistics
        if "FinInstrmTp" in master_batch.columns:
            counts = master_batch["FinInstrmTp"].value_counts().to_dict()
            instrument_counts.update(counts)

        # Partitioning
        master_batch["year"] = master_batch["TradDt"].dt.year
        master_batch["month"] = master_batch["TradDt"].dt.month

        # Write Partitioned
        try:
            # Explicit Loop for robust partitioning
            groups = master_batch.groupby(["year", "month"])
            for (curr_year, curr_month), group_df in groups:
                target_path = LAKE_DIR / f"year={curr_year}" / f"month={curr_month}"
                target_path.mkdir(parents=True, exist_ok=True)

                fname = f"data_batch_{i}_{datetime.now().strftime('%H%M%S%f')}.parquet"
                group_df.to_parquet(
                    target_path / fname,
                    engine="pyarrow",
                    compression="snappy",
                    index=False,
                )

            total_rows_ingested += len(master_batch)

        except Exception as e:
            print(f"Error writing batch {i}: {e}")

        del master_batch
        del batch_dfs
        gc.collect()

    print("\n" + "=" * 40)
    print("FINAL REPORT (V3)")
    print("=" * 40)
    print(f"TOTAL FILES INGESTED : {len(files)}")
    print(f"TOTAL ROWS           : {total_rows_ingested}")
    print("\nINSTRUMENT DISTRIBUTION:")

    # Sort by count desc
    for inst, count in instrument_counts.most_common():
        print(f"{inst:<15} : {count:,.0f}")

    print("========================================")


if __name__ == "__main__":
    process_lake()
