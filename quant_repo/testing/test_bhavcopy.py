import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
from datetime import date
import shutil
import io
import zipfile
from unittest.mock import MagicMock
from quant_repo.data.bhavcopy import BhavcopyLoader


def create_mock_zip(csv_content: str) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("fo01JAN2023bhav.csv", csv_content)
    b.seek(0)
    return b.read()


def test_bhavcopy_loader():
    print("[TEST] Historical Bhavcopy Pipeline...")

    test_dir = Path("./temp_bhav_history")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    loader = BhavcopyLoader(root_path=str(test_dir))

    # 1. Mock Network
    mock_csv = """INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP
OPTIDX,NIFTY,05-JAN-2023,18000.00,CE,100.0,150.0,90.0,120.0,120.0,5000,40000.00,100000,5000,02-JAN-2023
OPTIDX,NIFTY,05-JAN-2023,18000.00,PE,80.0,90.0,50.0,60.0,60.0,4000,30000.00,200000,-1000,02-JAN-2023
FUTIDX,NIFTY,25-JAN-2023,0.00,XX,18100.0,18200.0,18050.0,18150.0,18150.0,2000,20000.00,50000,100,02-JAN-2023
"""
    zip_bytes = create_mock_zip(mock_csv)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = zip_bytes

    loader.session.get = MagicMock(return_value=mock_response)

    # 2. Run for specific date
    test_date = date(2023, 1, 2)
    success = loader.download_and_process_date(test_date)

    assert success
    print("Download & Process Successful.")

    # 3. Verify URL Construction
    # Expected: .../2023/JAN/fo02JAN2023bhav.csv.zip (Assuming logic uses 02 for day)
    # The actual date passed was Jan 2.
    args, _ = loader.session.get.call_args
    # Check if URL contains expected parts
    assert "2023/JAN/fo02JAN2023bhav.csv.zip" in args[0]
    print("URL Construction Correct.")

    # 4. Verify Storage & Schema
    # Check: temp_bhav_history/symbol=NIFTY/year=2023/2023-01-02.parquet
    expected_file = test_dir / "symbol=NIFTY" / "year=2023" / "2023-01-02.parquet"
    assert expected_file.exists()

    df_loaded = pl.read_parquet(expected_file)
    print(f"Loaded Parquet columns: {df_loaded.columns}")

    assert "oi" in df_loaded.columns
    assert "volume" in df_loaded.columns
    assert len(df_loaded) == 2  # 2 Options (Feature filtered out)
    assert df_loaded["type"].to_list() == ["CE", "PE"]

    print("Schema Normalization Correct (Futures filtered, columns mapped).")

    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_bhavcopy_loader()
