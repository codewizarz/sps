import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
from datetime import date
from quant_repo.validation.reality import RealityDetector


def test_reality_detector():
    print("[TEST] Reality Check...")

    # 1. Market Data
    # 3 Days
    market_data = pl.DataFrame(
        {
            "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "symbol": ["NIFTY", "NIFTY", "NIFTY"],
            "high": [105.0, 110.0, 100.0],
            "low": [95.0, 100.0, 90.0],
            "bid": [99.0, 104.0, 94.0],
            "ask": [101.0, 106.0, 96.0],  # Spread 2.0
            "volume": [1000, 5000, 1000],
        }
    )

    # 2. Trade Log (Simulated)
    # Trade 1: Acceptable. Buy at 101 (Ask). Vol 10.
    # Trade 2: Optimistic. Buy at 100 (Mid/Low), Real Ask is 106. Huge Optimism.
    # Trade 3: Volume Violation. Buy 500 lots when Vol is 1000 (50%).
    # Trade 4: Price Violation. Buy at 80 (Below Low 90).

    trade_log = pl.DataFrame(
        {
            "date": [
                date(2023, 1, 1),
                date(2023, 1, 2),
                date(2023, 1, 3),
                date(2023, 1, 3),
            ],
            "symbol": ["NIFTY", "NIFTY", "NIFTY", "NIFTY"],
            "action": ["BUY", "BUY", "BUY", "BUY"],
            "quantity": [10, 10, 500, 10],
            "exec_price": [101.0, 100.0, 95.0, 80.0],
            "assumed_spread": [2.0, 2.0, 2.0, 2.0],
        }
    )

    detector = RealityDetector()
    report = detector.check_execution(trade_log, market_data)

    print(f"Optimism Score: {report.optimism_score:.4%}")
    print(f"Volume Violations: {report.volume_violations}")
    print(f"Suspicious Trades: {len(report.suspicious_trades)}")

    print("\nSuspicious Trades Table:")
    print(
        report.suspicious_trades.select(
            [
                "date",
                "exec_price",
                "real_market_price",
                "optimism_per_unit",
                "vol_violation",
                "price_violation",
            ]
        )
    )

    # Assertions
    # T2 (Date 1/2): Exec 100, Ask 106. Optimism = 6 per unit. Should be flagged.
    # T3 (Date 1/3): Qty 500 / Vol 1000 = 0.5 > 0.1. Vol Violation.
    # T4 (Date 1/3): Exec 80 < Low 90. Price Violation.

    # T1 is fine.

    assert report.volume_violations >= 1
    assert len(report.suspicious_trades) >= 3  # T2, T3, T4
    assert report.optimism_score > 0  # Positive means we were optimistic

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_reality_detector()
