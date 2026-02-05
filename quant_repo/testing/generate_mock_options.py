import numpy as np
import pandas as pd
from pathlib import Path
import polars as pl
import shutil


def generate_mock_options_data():
    dataset_path = Path.cwd() / "temp_options_data"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    dataset_path.mkdir(parents=True)

    # Define parameters
    symbol = "NIFTY"
    expiry = "2024-01-25"
    strikes = [21000, 21100]
    rights = ["CE", "PE"]
    dates = pd.date_range("2024-01-01", periods=10, freq="1min")

    rows = []

    for d in dates:
        for k in strikes:
            for r in rights:
                # Mock Prices
                spot = 21050 + np.random.randn() * 10
                intrinsic = max(0, spot - k) if r == "CE" else max(0, k - spot)
                premium = intrinsic + np.random.uniform(50, 100)

                rows.append(
                    {
                        "timestamp": d.value,  # nanos
                        "symbol": symbol,
                        "expiry": expiry,
                        "strike": float(k),
                        "right": r,
                        "bid": round(premium - 0.5, 2),
                        "ask": round(premium + 0.5, 2),
                        "volume": 100,
                        "IV": 0.15,
                        "OI": 100000,
                    }
                )

    df = pl.DataFrame(rows)

    # Write partitioned Parquet
    # We want: root / symbol=NIFTY / expiry=2024-01-25 / ...
    # Polars partition_by is simpler to iterate and write

    partitions = df.partition_by(["symbol", "expiry"], include_key=False)

    # Note: polars write_parquet doesn't support hive partitioning write directly in simple API always,
    # but we can do it manually or use pyarrow dataset.
    # Let's do it manually for control since we only have one combo here.

    for key, group in df.group_by(["symbol", "expiry"]):
        s, e = key
        # group is a dataframe
        path = dataset_path / f"symbol={s}" / f"expiry={e}"
        path.mkdir(parents=True, exist_ok=True)

        # Write file
        group.write_parquet(path / "data.parquet")

    print(f"[MOCK] Generated data at {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    generate_mock_options_data()
