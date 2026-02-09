import pandas as pd
import glob
from tqdm import tqdm

files = glob.glob("nse_fo_bhavcopies_extracted/*.csv")

dfs = []
print(f"Found {len(files)} files.")

if not files:
    print("No files found!")
else:
    for file in tqdm(files):
        try:
            df = pd.read_csv(file, low_memory=False)

            # Check mandatory cols
            if "OptnTp" not in df.columns:
                continue

            # Keep only options
            df = df[df["OptnTp"].isin(["CE", "PE"])]

            # Remove dead contracts
            df = df[df["TtlTradgVol"] > 10]
            df = df[df["OpnIntrst"] > 100]

            # Convert dates
            df["TradDt"] = pd.to_datetime(df["TradDt"])
            df["XpryDt"] = pd.to_datetime(df["XpryDt"])

            dfs.append(df)
        except Exception as e:
            pass  # Skip bad files

    if dfs:
        master = pd.concat(dfs, ignore_index=True)

        master.sort_values("TradDt", inplace=True)

        master.to_parquet("MASTER_OPTIONS.parquet")

        print("DONE — Your Quant Database Is Born.")
    else:
        print("No valid data processed.")
