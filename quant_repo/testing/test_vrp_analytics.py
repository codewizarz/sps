import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pytest
from quant_repo.analytics.vrp_stats import VRPAnalyzer


def test_vrp_analytics():
    print("[TEST] VRP Analytics Engine...")

    analyzer = VRPAnalyzer()

    # 1. Generate Synthetic Data
    # 100 days
    # Days 0-50: "Calm" (Returns ~0, IV High) -> VRP Positive
    # Days 50-70: "Crash" (Returns High Vol, IV High) -> VRP Low/Negative
    # Days 70-100: "Calm"

    n = 100
    dates = pl.date_range(
        start=pl.date(2023, 1, 1), end=pl.date(2023, 4, 10), interval="1d", eager=True
    )[:n]

    # Prices
    returns = np.zeros(n)
    # Calm: tiny random moves
    returns[0:50] = np.random.normal(0, 0.005, 50)
    # Crash: big moves
    returns[50:70] = np.random.normal(0, 0.03, 20)
    # Calm
    returns[70:100] = np.random.normal(0, 0.005, 30)

    price = 100 * np.exp(np.cumsum(returns))

    # Implied Vol (Annualized)
    # Constant high IV for simplicity to isolate RV effect
    iv = np.full(n, 0.30)  # 30% Vol

    # Regimes
    regimes = ["BULL_QUIET"] * 50 + ["CRISIS"] * 20 + ["BULL_QUIET"] * 30

    df = pl.DataFrame({"date": dates, "close": price, "iv": iv, "regime": regimes})

    # 2. Calculate VRP (Window = 10 days for this short sample)
    print("\n--- Testing Calculation ---")
    df_res = analyzer.calculate_vrp(df, lookahead_window=10)

    print(df_res.head(5))
    print(df_res.filter(pl.col("regime") == "CRISIS").head(5))

    # Verification Logic:
    # In Calm period (0-40), RV should be low (~0.005 * sqrt(252) ~= 0.08)
    # IV is 0.30. VRP should be approx 0.22 (Positive)

    # In Pre-Crash period (rows 40-50), FUTURE RV includes the crash.
    # So VRP should drop before the crash regime starts in the data if we looked ahead?
    # Actually, row 45's RV_future looks at 46-55. 50-55 are crash. So RV rises.
    # Row 55's RV_future looks at 56-65 (Crash). RV High.

    # Let's check average VRP in Calm vs Crisis rows
    # Careful: 'Regime' column is contemporaneous. 'VRP' column depends on Future RV.
    # So a 'CRISIS' day (high vol today) might have Low Future Vol (Normalization) -> High VRP?
    # Or High Future Vol (Continuation) -> Low VRP.

    stats_dist = analyzer.analyze_distribution(df_res)
    print("\nLikely Positive VRP Mean:", stats_dist["mean_vrp"])
    assert (
        stats_dist["mean_vrp"] > 0
    )  # Generally selling 30% vol against 0.5% daily returns is profitable

    # 3. Analyze by Regime
    print("\n--- Testing Segmentation ---")
    seg_df = analyzer.analyze_by_segment(df_res, "regime")
    print(seg_df)

    # We expect BULL_QUIET to have higher VRP than CRISIS (where realized moves match IV)
    bull_vrp = seg_df.filter(pl.col("regime") == "BULL_QUIET")["mean_vrp"][0]
    # crisis_vrp might be lower or negative depending on how much 30% IV underpriced the 3% moves
    # 3% daily ~= 47% annualized. So IV(30) - RV(47) = -17%.

    assert bull_vrp > 0.05

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_vrp_analytics()
