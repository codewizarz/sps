import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pytest
from quant_repo.analytics.tail_risk import TailRiskAnalyzer


def test_tail_risk_detector():
    print("[TEST] Tail Risk Detector...")

    analyzer = TailRiskAnalyzer()

    # Generate Synthetic Data (100 Days)
    n = 100
    dates = pl.date_range(
        start=pl.date(2023, 1, 1), end=pl.date(2023, 4, 10), interval="1d", eager=True
    )[:n]

    # 1. Prices (Gap Event at index 50)
    # Vol is approx 1% daily
    returns = np.random.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(returns))
    open_prices = close.copy()  # Usually open = close prev

    # Create Gap Shock
    # Index 50: Open is 5% lower than Prev Close (5 sigma gap)
    open_prices[50] = close[49] * 0.95
    close[50] = open_prices[50]  # Market stays down

    # 2. IV (Vol Explosion at index 70)
    iv = np.full(n, 0.20)
    iv[70] = 0.40  # 100% Increase (0.2 -> 0.4)

    # 3. Spread (Liquidity Freeze at index 90)
    bid = close - 0.05
    ask = close + 0.05  # Spread 0.10 normally

    ask[90] = close[90] + 0.50  # Spread 0.55 (~5x widening)

    # 4. Skew (Skew Shock at index 20)
    skew = np.full(n, 0.05)
    skew[20] = 0.15  # 3x widening

    df = pl.DataFrame(
        {
            "date": dates,
            "open": open_prices,
            "close": close,
            "iv": iv,
            "bid": bid,
            "ask": ask,
            "skew": skew,
        }
    )

    # Run Detection
    print("\n--- Detecting Shocks ---")
    df_res = analyzer.detect_shocks(df)

    # Check Gap Shock (Index 50)
    print(f"Index 50 Gap Shock: {df_res['is_gap_shock'][50]}")
    assert df_res["is_gap_shock"][50] == True

    # Check Vol Explosion (Index 70)
    print(f"Index 70 Vol Explosion: {df_res['is_vol_explosion'][70]}")
    assert df_res["is_vol_explosion"][70] == True

    # Check Liquidity Freeze (Index 90)
    print(f"Index 90 Liquidity Freeze: {df_res['is_liquidity_freeze'][90]}")
    assert df_res["is_liquidity_freeze"][90] == True

    # Check Skew Shock (Index 20)
    print(f"Index 20 Skew Shock: {df_res['is_skew_shock'][20]}")
    assert df_res["is_skew_shock"][20] == True

    # Check Stats
    stats = analyzer.analyze_shock_stats(df_res)
    print("\nStats:", stats)
    assert stats["gap_shocks"] >= 1
    assert stats["vol_explosions"] >= 1
    assert stats["liquidity_freezes"] >= 1

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_tail_risk_detector()
