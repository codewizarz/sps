import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
from datetime import date
import shutil
from quant_repo.data.pipeline import DataPipeline, Catalog


def test_data_pipeline():
    print("[TEST] Data Ingestion Pipeline...")

    test_root = Path("./temp_data_pipeline")
    if test_root.exists():
        shutil.rmtree(test_root)

    pipeline = DataPipeline(root_path=str(test_root))

    # 1. Create Valid Data (2 days, 1 expiry)
    valid_data = pl.DataFrame(
        {
            "date": [date(2023, 1, 1), date(2023, 1, 2)],
            "symbol": ["NIFTY", "NIFTY"],
            "expiry": [date(2023, 1, 5), date(2023, 1, 5)],
            "strike": [18000, 18000],
            "type": ["CE", "CE"],
            "bid": [100.0, 105.0],
            "ask": [102.0, 107.0],
            "close": [101.0, 106.0],
            "volume": [1000, 1200],
        }
    )

    success = pipeline.ingest(valid_data, symbol="NIFTY")
    assert success
    print("Ingestion Successful.")

    # 2. Check Directory Structure
    # Expect: temp_data_pipeline/symbol=NIFTY/expiry=2023-01-05/date=2023-01-01/...
    # Note: Polars write_parquet with partition_cols might create slightly different depending on version,
    # but hive structure is standard.

    # 3. Query via Catalog
    catalog = Catalog(root_path=str(test_root))

    # Load Date 1
    df_d1 = catalog.load_range("NIFTY", "2023-01-01", "2023-01-01")
    assert len(df_d1) == 1
    assert df_d1["date"][0] == date(2023, 1, 1)
    print("Query 1 Successful.")

    # Load Range (both days)
    df_range = catalog.load_range("NIFTY", "2023-01-01", "2023-01-05")
    assert len(df_range) == 2
    print("Query Range Successful.")

    # 4. Test Audit Rejection (Bad Data)
    bad_data = pl.DataFrame(
        {
            "date": [date(2023, 1, 3)],
            "symbol": ["NIFTY"],
            "expiry": [date(2023, 1, 2)],  # Broken Expiry
            "strike": [18000],
            "type": ["CE"],
            "bid": [100.0],
            "ask": [90.0],  # Crossed
            "close": [0.0],  # Zero Price
            "volume": [100],
        }
    )

    print("Attempting to ingest Bad Data (Should Fail)...")
    success_bad = pipeline.ingest(bad_data, symbol="NIFTY")
    assert not success_bad
    print("Bad Data correctly rejected.")

    # Cleanup
    if test_root.exists():
        shutil.rmtree(test_root)

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_data_pipeline()
