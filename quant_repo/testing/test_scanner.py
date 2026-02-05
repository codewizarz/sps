import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pandas as pd
from quant_repo.research.edges import VRPEdge, SkewEdge
from quant_repo.research.scanner import EdgeScanner


def test_edge_scanner():
    print("[TEST] Automated Edge Scanner...")

    # 1. Generate Mock History (3 Years)
    # 252 * 3 = 756 days
    dates = pd.date_range("2021-01-01", periods=756, freq="B")
    timestamps = dates.view(np.int64)

    np.random.seed(42)

    # VRP Setup: IV consistently higher than RV
    # IV ~ 20%, RV ~ 15%
    iv = 0.20 + np.random.normal(0, 0.02, 756)
    rv = 0.15 + np.random.normal(0, 0.02, 756)

    # Skew Setup: Noisy
    skew = np.random.normal(0.02, 0.05, 756)

    # Spot Returns for Skew crash calc
    # Generally flat, one crash
    spot_ret = np.random.normal(0, 0.01, 756)
    spot_ret[100] = -0.05  # Crash

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "iv_atm": iv,
            "realized_vol": rv,
            "iv_skew": skew,
            "spot_return": spot_ret,
        }
    )

    # 2. Run Scanner
    scanner = EdgeScanner(edges=[VRPEdge(), SkewEdge()])
    report = scanner.scan(df)

    # 3. Validation
    print("\n=== Edge Report ===")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN"):
        print(report)

    # Assertions
    # VRP should be GOLD/SILVER (IV > RV consistently)
    vrp_row = report.filter(pl.col("edge_name") == "VRP_Short_Straddle")
    assert len(vrp_row) > 0

    sharpe = vrp_row["sharpe"][0]
    print(f"\nVRP Sharpe: {sharpe:.2f}")
    assert sharpe > 0.5

    print("[TEST] SUCCESS")


if __name__ == "__main__":
    test_edge_scanner()
