import pandas as pd
from pathlib import Path
import sys


def normalize_dataframe(df):
    """
    Normalizes a raw NSE dataframe to the standard quant schema.
    """
    # Map NSE -> standard
    mapping = {
        "TckrSymb": "symbol",
        "StrkPric": "strike",
        "UndrlygPric": "underlying_price",
        "XpryDt": "expiry",
        "OptnTp": "option_type",
        "OpnIntrst": "open_interest",
        "TtlTradgVol": "volume",
        "FinInstrmTp": "instrument_type",
    }

    # Timestamp mapping
    if "BizDt" in df.columns:
        mapping["BizDt"] = "timestamp"
    elif "TradDt" in df.columns:
        mapping["TradDt"] = "timestamp"

    # Option price mapping
    if "SttlmPric" in df.columns:
        mapping["SttlmPric"] = "option_price"
    elif "LastPric" in df.columns:
        mapping["LastPric"] = "option_price"

    df = df.rename(columns=mapping)

    # Standard schema definition
    standard_cols = [
        "symbol",
        "timestamp",
        "strike",
        "underlying_price",
        "option_price",
        "expiry",
        "option_type",
        "open_interest",
        "volume",
        "instrument_type",
    ]

    for col in standard_cols:
        if col not in df.columns:
            # Handle missing fields gracefully
            df[col] = pd.NA

    # Convert timestamp to pandas datetime safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Convert numeric columns safely
    numeric_cols = [
        "strike",
        "underlying_price",
        "option_price",
        "open_interest",
        "volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing underlying_price or strike
    df = df.dropna(subset=["underlying_price", "strike"]).copy()

    return df


def load_normalized_lake(path):
    """
    Recursively loads all parquet files from the path and normalizes them.
    Concatenates them safely and returns a single standardized dataframe.
    """
    lake_dir = Path(path).resolve()

    if not lake_dir.exists():
        print(f"Error: Directory {lake_dir} does not exist.")
        sys.exit(1)

    files = list(lake_dir.rglob("*.parquet"))
    if not files:
        print(f"Error: No parquet files found in {lake_dir}")
        sys.exit(1)

    dfs = []
    for f in files:
        try:
            raw_df = pd.read_parquet(f)
            norm_df = normalize_dataframe(raw_df)
            dfs.append(norm_df)
        except Exception as e:
            print(f"Error processing file {f}: {e}")

    if not dfs:
        print("Error: No valid dataframes loaded from the lake.")
        sys.exit(1)

    final_df = pd.concat(dfs, ignore_index=True)

    print(f"Total files: {len(files)}")
    print(f"Total rows: {len(final_df)}")

    return final_df
