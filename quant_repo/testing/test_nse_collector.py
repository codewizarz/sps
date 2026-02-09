import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import pandas as pd
import shutil
from unittest.mock import MagicMock
from quant_repo.data.nse_collector import NSECollector


def test_nse_collector():
    print("[TEST] NSE Option Chain Collector...")

    test_dir = Path("./temp_nse_snapshots")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    collector = NSECollector(output_dir=str(test_dir))

    # 1. Mock the Network Call
    # We want to test parsing logic and saving logic.

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "records": {
            "timestamp": "10-Jan-2025 10:00:00",
            "data": [
                {
                    "expiryDate": "12-Jan-2025",
                    "strikePrice": 18000,
                    "CE": {
                        "strikePrice": 18000,
                        "expiryDate": "12-Jan-2025",
                        "bidprice": 100,
                        "askPrice": 102,
                        "lastPrice": 101,
                        "impliedVolatility": 15.5,
                        "totalTradedVolume": 5000,
                        "openInterest": 100000,
                        "changeinOpenInterest": 500,
                    },
                    "PE": {
                        "strikePrice": 18000,
                        "expiryDate": "12-Jan-2025",
                        "bidprice": 80,
                        "askPrice": 82,
                        "lastPrice": 81,
                        "impliedVolatility": 16.5,
                        "totalTradedVolume": 4000,
                        "openInterest": 90000,
                        "changeinOpenInterest": -200,
                    },
                }
            ],
        }
    }

    collector.session.get = MagicMock(return_value=mock_response)

    # 2. Fetch
    df = collector.fetch_option_chain("NIFTY")

    assert not df.empty
    assert len(df) == 2  # 1 CE, 1 PE
    assert "timestamp" in df.columns
    assert df["strike"].iloc[0] == 18000
    print("Parsing Successful.")

    # 3. Save
    collector.save_snapshot(df)

    # Check File
    # Expected: temp_nse_snapshots/2025-01-10/10-00-00.parquet
    expected_dir = test_dir / "2025-01-10"
    assert expected_dir.exists()

    parquet_files = list(expected_dir.glob("*.parquet"))
    assert len(parquet_files) == 1
    print(f"Snapshot Saved: {parquet_files[0]}")

    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_nse_collector()
