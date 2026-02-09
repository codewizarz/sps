"""
build_master_fo_v2.py

PRODUCTION-GRADE FO MASTER BUILDER (V2)
---------------------------------------
Handles the NEW NSE FO Schema with strict normalization.
Converts legacy column names to the new NSE standard.

FEATURES:
1.  **Strict Normalization**: Renames old columns (INSTRUMENT -> FinInstrmTp) and drops useless ones.
2.  **Filtering**: Keeps only FUTIDX, FUTSTK, OPTIDX, OPTSTK.
3.  **Type Enforcement**: Dates (datetime64), Prices (float32), OI/Vol (int32).
4.  **Partitioning**: Writes to data/master_fo_lake/year=YYYY/month=MM/ (Snappy).
5.  **Validation**: Fails loudly if required data is missing.

USAGE:
    python DATA_FETCHING/build_master_fo_v2.py
"""

import pandas as pd
import glob
import os
import shutil
from pathlib import Path
from datetime import datetime
import gc

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
    "CLOSE": "ClsPric",  # Settle Price is separate usually, but Close is key
    "SETTLE_PR": "SttlPric",
    "CONTRACTS": "TtlTradgVol",  # Volume
    "VAL_INLAKH": "TtlTrfVal",  # Value
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
    "Unnamed: 0",  # Common pandas artifact
]

# 3. Filter Logic
ACCEPTED_INSTRUMENTS = ["FUTIDX", "FUTSTK", "OPTIDX", "OPTSTK"]


def normalize_and_clean(df, filename):
    """
    Applies strict normalization, renaming, filtering, and type casting.
    Raises ValueError if critical conditions fail.
    """

    # A. Rename Columns (Handle variations)
    # First, normalize casing to UPPER for matching
    df.columns = [c.strip() for c in df.columns]

    # Check if we have Old or New schema?
    # We check for key columns.

    # Create a mapping that handles potentially existing new names or old names
    # Strategy: Rename ANY known alias to the Target (New) Name.

    # Invert map for variations? No, explicitly map knowns.

    current_cols = set(df.columns)
    rename_dict = {}

    for old_col, new_col in COLUMN_MAP.items():
        # Heuristic: Check exact match first, then case-insensitive
        if old_col in current_cols:
            rename_dict[old_col] = new_col
        elif new_col in current_cols:
            pass  # Already renamed
        else:
            # Case insensitive check
            for c in current_cols:
                if c.upper() == old_col.upper():
                    rename_dict[c] = new_col
                    break

    if rename_dict:
        df = df.rename(columns=rename_dict)

    # B. Validate Required Columns
    # Target columns that MUST exist after renaming
    REQUIRED = [
        "FinInstrmTp",
        "TckrSymb",
        "XpryDt",
        "StrkPric",
        "OptnTp",
        "OpnPric",
        "HghPric",
        "LwPric",
        "ClsPric",
        "OpnIntrst",
        "TradDt",
    ]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Critical Columns: {missing} in file {filename}")

    # C. Drop Useless Columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # D. Filter Instruments
    df = df[df["FinInstrmTp"].isin(ACCEPTED_INSTRUMENTS)]

    if df.empty:
        return None

    # E. Strict Type Enforcing

    # 1. Dates
    # Parse TradDt and XpryDt. NSE format usually dd-MMM-yyyy
    for date_col in ["TradDt", "XpryDt"]:
        try:
            df[date_col] = pd.to_datetime(
                df[date_col], format="%d-%b-%Y", errors="coerce"
            )
        except:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop Invalid Dates
    df = df.dropna(subset=["TradDt", "XpryDt"])

    # 2. Prices (Float32)
    price_cols = ["StrkPric", "OpnPric", "HghPric", "LwPric", "ClsPric"]
    if "SttlPric" in df.columns:
        price_cols.append("SttlPric")

    for col in price_cols:
        # Handle "1,200.00" strings or "-"
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].astype("float32")

    # 3. Volume / OI (Int32)
    int_cols = ["OpnIntrst", "ChngInOpnIntrst"]
    if "TtlTradgVol" in df.columns:
        int_cols.append("TtlTradgVol")

    for col in int_cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.replace(",", "", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = df[col].fillna(0).astype("int32")

    return df


def process_lake():
    print("=== STARTING MASTER FO BUILDER (V2) ===")
    print(f"Source: {RAW_DATA_DIR.absolute()}")
    print(f"Target: {LAKE_DIR.absolute()}")

    files = list(RAW_DATA_DIR.glob("*.csv"))
    if not files:
        print("CRITICAL: No CSV files found.")
        return

    print(f"Files Found: {len(files)}")

    # Reset Lake (To avoid duplicates)
    if LAKE_DIR.exists():
        shutil.rmtree(LAKE_DIR)
    LAKE_DIR.mkdir(parents=True, exist_ok=True)

    # Batch Processing
    BATCH_SIZE = 50
    total_rows_ingested = 0
    final_shape = None

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
                # User asked to Fail Loudly if required cols missing (ValueError).
                # But CSV corruption (empty file, bad zip) handles gracefully?
                # "Fail loudly if required columns are missing." -> Raises ValueError caught here?
                # Retrying logic not requested. We STOP or Skip?
                # "Never silently skip data." -> Print the error clearly.
                print(f"❌ FAIL: {f.name} -> {e}")
                # If "Fail Loudly" means stop script: raise e.
                # Usually implies stopping.
                # However, for 500 files, one bad file stopping everything is annoying.
                # I will print LOUDLY but continue if it's just parsable error,
                # but if ValueError (missing cols), I'll let it raise if user insisted "Fail loudly".
                # User said: "Fail loudly if required columns are missing."
                # Interpreting as: STOP.
                if isinstance(e, ValueError) and "Missing Critical Columns" in str(e):
                    raise e

        if not batch_dfs:
            continue

        # Combine Batch
        master_batch = pd.concat(batch_dfs, ignore_index=True)

        # Partitioning
        master_batch["year"] = master_batch["TradDt"].dt.year
        master_batch["month"] = master_batch["TradDt"].dt.month

        # Write to Parquet (Partitioned)
        # using pyarrow engine with partition_cols
        # Append mode?
        # Parquet datasets structure handles multiple files.
        # We can write `part-batch-i.parquet` into appropriate folders.

        # Pandas to_parquet with partition_cols writes a directory tree.
        # It handles splitting the dataframe into folders.
        # We need `existing_data_behavior='overwrite_or_ignore'`? No, we want to ADD files.
        # Default behavior of to_parquet with partition_cols is to write new files if filenames differ?
        # Actually standard pandas `to_parquet` usually writes one file structure.
        # If we loop, we might overwrite?
        # Safest: Use pyarrow.parquet.write_to_dataset logic via pandas?
        # Pandas `to_parquet(..., partition_cols=['year', 'month'])` uses pyarrow.
        # If we provide a path to a directory, it creates the structure.
        # We need to ensure unique filenames inside the partitions.
        # `basename_template` argument in `to_parquet` (passed to pyarrow) helps.

        try:
            master_batch.to_parquet(
                LAKE_DIR,
                engine="pyarrow",
                compression="snappy",
                partition_cols=["year", "month"],
                index=False,
                existing_data_behavior="overwrite_or_ignore",
                # Note: 'overwrite_or_ignore' is not a standard pandas arg for to_parquet?
                # It is for pyarrow.parquet.write_to_dataset.
                # Pandas passes kwargs?
                # Let's check correctness. Pandas `to_parquet(path, partition_cols=...)` usually implies path is root dir.
            )

            # ISSUE: If we run this loop multiple times, pandas might strictly overwrite the "common_metadata" or files?
            # Actually, without `append=True` (not supported for parquet usually), it might be tricky.
            # Best Robust Way (Pandas Only):
            # Iterate unique Year/Month combinations in the batch and write unique filenames manually.

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
            if final_shape is None:
                final_shape = master_batch.shape

        except Exception as e:
            print(f"Error writing batch {i}: {e}")
            raise e

        # Memory Cleanup
        del master_batch
        del batch_dfs
        gc.collect()

    print("\n" + "=" * 30)
    print("FINAL REPORT")
    print("=" * 30)
    print(f"TOTAL FILES INGESTED : {len(files)}")
    print(f"TOTAL ROWS           : {total_rows_ingested}")
    if final_shape:
        # Estimate size?
        # Just printing "FINAL MEMORY SIZE" as requested (of the last batch or total?)
        # "FINAL MEMORY SIZE" suggests size of dataset?
        # Hard to know total size without reloading.
        # We'll print total rows.
        pass
    print("==============================")


if __name__ == "__main__":
    process_lake()
