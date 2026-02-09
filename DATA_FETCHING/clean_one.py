import pandas as pd
import glob
from pathlib import Path

# Use the correct directory from extraction step
# Path was "nse_fo_bhavcopies_extracted" in the previous script
# glob supports Path objects in newer python or string
EXTRACT_DIR = "nse_fo_bhavcopies_extracted"
files = glob.glob(f"{EXTRACT_DIR}/*.csv")

dfs = []

if not files:
    print(f"No CSV files found in {EXTRACT_DIR}. Did you run extract_fo_bhavcopes.py?")
else:
    print(f"Found {len(files)} files. Processing...")

    for file in files:
        try:
            df = pd.read_csv(file)

            # Check if columns exist before filtering to avoid KeyErrors
            if "OptnTp" not in df.columns:
                continue

            # Keep only options
            df = df[df["OptnTp"].isin(["CE", "PE"])]

            # Keep only liquid contracts
            df = df[df["TtlTradgVol"] > 50]
            df = df[df["OpnIntrst"] > 500]

            # Focus on index first
            df = df[df["TckrSymb"].isin(["NIFTY", "BANKNIFTY"])]

            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    if dfs:
        master = pd.concat(dfs)
        output_file = "clean_options.parquet"
        master.to_parquet(output_file)
        print(f"Saved {len(master)} rows to {output_file}")
    else:
        print("No data matched criteria.")
