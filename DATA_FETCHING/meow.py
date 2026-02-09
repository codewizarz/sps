import pandas as pd
import glob
from tqdm import tqdm

files = glob.glob("nse_fo_bhavcopies_extracted/*.csv")

dfs = []

for file in tqdm(files):
    df = pd.read_csv(file, low_memory=False)

# Keep Futures + Options BOTH
df["TradDt"] = pd.to_datetime(df["TradDt"])
df["XpryDt"] = pd.to_datetime(df["XpryDt"])

dfs.append(df)

master = pd.concat(dfs, ignore_index=True)

master.sort_values("TradDt", inplace=True)

master.to_parquet("MASTER_FO_FULL.parquet")

print("TRUE MASTER CREATED")
