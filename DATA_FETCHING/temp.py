import pandas as pd
df = pd.read_csv("nse_fo_bhavcopies_extracted/BhavCopy_NSE_FO_0_0_0_20240101_F_0000.csv")

print(df["FinInstrmTp"].value_counts())
