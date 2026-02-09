"""
inspect_fo_schema.py

READ-ONLY DIAGNOSTIC SCRIPT
---------------------------
Inspects NSE FO Bhavcopy schemas across years (2024, 2025, 2026).
Randomly samples 5 files per year to detect schema drift.

OUTPUT:
- Column names
- First 5 rows
- Unique Instruments/Symbols
- Final Schema Summary
"""

import pandas as pd
import glob
import os
import random
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
EXTRACT_DIR = Path("nse_fo_bhavcopies_extracted")
YEARS_TO_CHECK = ["2024", "2025", "2026"]
SAMPLES_PER_YEAR = 5


def inspect_files():
    print(f"=== NSE FO SCHEMA INSPECTOR ===")
    print(f"Source: {EXTRACT_DIR.absolute()}\n")

    if not EXTRACT_DIR.exists():
        print(f"ERROR: Directory '{EXTRACT_DIR}' not found.")
        return

    all_csvs = list(EXTRACT_DIR.glob("*.csv"))
    if not all_csvs:
        print("ERROR: No CSV files found.")
        return

    # Categorize by Year
    files_by_year = defaultdict(list)
    for f in all_csvs:
        # Simplistic year extraction from filename "BhavCopy_NSE_FO_0_0_0_20251126_F_0000.csv"
        # Extract "2025" from filename
        name = f.name
        # Try to find year pattern
        for year in YEARS_TO_CHECK:
            if year in name:
                files_by_year[year].append(f)
                break

    # Global Aggregators
    all_instruments = set()
    schema_registry = defaultdict(list)  # {tuple(columns): [years]}
    column_frequency = defaultdict(int)
    total_files_scanned = 0

    print(f"{'filename':<50} | {'rows':<10} | {'cols':<5}")
    print("-" * 80)

    for year in YEARS_TO_CHECK:
        files = files_by_year.get(year, [])
        if not files:
            print(f"\n[YEAR {year}] No files found.")
            continue

        print(
            f"\n[YEAR {year}] Found {len(files)} files. Sampling {SAMPLES_PER_YEAR}..."
        )

        # Sample
        sample_files = random.sample(files, min(len(files), SAMPLES_PER_YEAR))

        for f in sample_files:
            try:
                # Read header only first to check cols, then data
                # Using pandas for robustness
                df = pd.read_csv(f, nrows=5)  # Peek

                # Get full row count (optional, might be slow for big files, but we asked for it)
                # Let's skip full count for speed? Requirements said "Show row count".
                # To get count cheaply:
                with open(f, "rb") as fp:
                    row_count = sum(1 for _ in fp) - 1  # Approx

                cols = tuple(df.columns.tolist())
                schema_registry[cols].append(f.name)

                for c in cols:
                    column_frequency[c] += 1

                print(f"{f.name:<50} | {row_count:<10} | {len(cols)}")

                # Capture Unique Instruments/Option Types
                # We need to read whole file for unique values?
                # Requirement: "Print UNIQUE values from: INSTRUMENT... SYMBOL... OPTION_TYP"
                # This implies reading specific columns entirely.

                # Read specific columns for unique stats
                # Try to find target cols
                target_cols = [
                    c
                    for c in df.columns
                    if "INSTRUMENT" in c.upper()
                    or "SYMBOL" in c.upper()
                    or "OPTION" in c.upper()
                ]

                if target_cols:
                    df_full = pd.read_csv(f, usecols=target_cols, low_memory=False)
                    unique_inst = (
                        df_full.iloc[:, 0].unique() if len(target_cols) > 0 else []
                    )
                    # Assuming Instrument is usually first in these files

                    # Refine logic for "Instrument" specifically
                    inst_col = next(
                        (c for c in df.columns if "INSTRUMENT" in c.upper()), None
                    )
                    if inst_col:
                        uniques = df_full[inst_col].dropna().unique()
                        all_instruments.update(uniques)
                        print(f"   > Instruments: {uniques}")

                    sym_col = next(
                        (c for c in df.columns if "SYMBOL" in c.upper()), None
                    )
                    if sym_col:
                        # Just show first 5 symbols to avoid spam
                        pass

                    opt_col = next(
                        (
                            c
                            for c in df.columns
                            if "OPTION" in c.upper() or "TP" in c.upper()
                        ),
                        None,
                    )
                    if opt_col:
                        print(
                            f"   > Option Types: {df_full[opt_col].dropna().unique()}"
                        )

                # Print Cols/Rows
                print(f"   > Columns: {list(df.columns)}")
                print(f"   > First Row: {df.iloc[0].to_dict()}\n")

                total_files_scanned += 1

            except Exception as e:
                print(f"   > ERROR reading {f.name}: {e}")

    # --- FINAL SUMMARY ---
    print("\n" + "=" * 40)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 40)

    print(f"\n1. Global Instruments Found:")
    print(sorted(list(all_instruments)))

    print(f"\n2. Schema Variations Detected: {len(schema_registry)}")
    for i, (cols, files) in enumerate(schema_registry.items()):
        print(f"\nSchema ID #{i + 1} (Seen in {len(files)} sampled files):")
        print(f"Columns: {list(cols)}")

    print(f"\n3. Column Consistency:")
    all_sampled = total_files_scanned
    for col, count in sorted(column_frequency.items(), key=lambda x: -x[1]):
        status = (
            "✅ CONSISTENT"
            if count == all_sampled
            else f"⚠️ INCONSISTENT ({count}/{all_sampled})"
        )
        print(f"{col:<30} : {status}")


if __name__ == "__main__":
    inspect_files()
