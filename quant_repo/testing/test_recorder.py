import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
from datetime import datetime, timedelta
import shutil
from quant_repo.data.recorder import MarketRecorder


def test_market_recorder():
    print("[TEST] Market Recorder (Data Advantage)...")

    # Force Polars to use simple ASCII tables
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")

    # 1. Setup Test Dir
    test_dir = Path.cwd() / "temp_data_recorder_test"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    recorder = MarketRecorder(data_dir=test_dir, buffer_size=5)

    # 2. Simulate Ticks
    # 2 Days of data
    base_time = datetime(2025, 1, 1, 9, 15)

    ticks = []

    # Day 1
    for i in range(10):
        ticks.append(
            {
                "timestamp": base_time + timedelta(seconds=i),
                "symbol": "NIFTY25JAN18000CE",
                "bid": 100.0 + i,
                "ask": 101.0 + i,
                "last": 100.5 + i,
                "volume": 50 + i,
                "oi": 1000 + i * 10,
            }
        )

    # Day 2
    base_time_2 = datetime(2025, 1, 2, 9, 15)
    for i in range(10):
        ticks.append(
            {
                "timestamp": base_time_2 + timedelta(seconds=i),
                "symbol": "NIFTY25JAN18000CE",
                "bid": 200.0 + i,
                "ask": 201.0 + i,
                "last": 200.5 + i,
                "volume": 20 + i,
                "oi": 2000 + i * 10,
            }
        )

    # 3. Feed Recorder
    # Buffer size is 5. We define 20 total. Should induce ~4 flushes.
    for t in ticks:
        recorder.on_tick(t)

    # Force flush remainder
    recorder.flush()

    # 4. Verify Files
    raw_path = test_dir / "raw" / "options_tick"

    day1_path = raw_path / "date=2025-01-01"
    day2_path = raw_path / "date=2025-01-02"

    assert day1_path.exists(), "Day 1 partition missing"
    assert day2_path.exists(), "Day 2 partition missing"

    # 5. Read Back check
    # Polars scan of partition
    df1 = pl.read_parquet(str(day1_path / "*.parquet"))
    print(f"\nDay 1 Records: {len(df1)}")
    # print(df1.head(3))

    assert len(df1) == 10
    assert df1["bid"][0] == 100.0

    df2 = pl.read_parquet(str(day2_path / "*.parquet"))
    print(f"\nDay 2 Records: {len(df2)}")
    assert len(df2) == 10
    assert df2["bid"][0] == 200.0

    # 6. Test Consolidation
    print("\nTesting Consolidation...")
    cons_path = recorder.consolidate("2025-01-01")
    assert cons_path.exists()

    df_cons = pl.read_parquet(cons_path)
    assert len(df_cons) == 10

    # Cleanup
    shutil.rmtree(test_dir)
    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_market_recorder()
