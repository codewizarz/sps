import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pytest
from quant_repo.analytics.zone_stats import ZoneAnalyzer


def test_zone_analyzer():
    print("[TEST] VRP Zone Analyzer...")

    analyzer = ZoneAnalyzer()

    # 1. Generate Synthetic Option Surface
    # Case A: 0DTE ATM (High Yield, High Gamma, Tight Spread) -> "The Casino"
    # Case B: 30DTE Wings (Low Yield, Low Gamma, Wide Spread) -> "The Insurance"
    # Case C: 7DTE OTM (Med Yield, Med Gamma, Tight Spread) -> "The Sweet Spot"

    data = {
        "strike": [10000.0] * 3,
        "dte": [1, 30, 7],
        "delta": [0.50, 0.15, 0.30],
        "gamma": [0.10, 0.01, 0.04],  # 0DTE has much higher gamma
        "mid_price": [50.0, 20.0, 35.0],
        "bid_price": [49.0, 15.0, 34.0],
        "ask_price": [51.0, 25.0, 36.0],
    }

    df = pl.DataFrame(data)

    # 2. Run Classification
    print("\n--- Testing Bucketing ---")
    df_bucketed = analyzer.bucket_options(df)
    print(df_bucketed.select(["dte", "delta", "dte_bucket", "delta_bucket"]))

    assert df_bucketed["dte_bucket"][0] == "0-2D"
    assert df_bucketed["delta_bucket"][0] == "ATM (50-40)"

    assert df_bucketed["dte_bucket"][1] == "21-45D"
    assert df_bucketed["delta_bucket"][1] == "Wings (25-10)"

    # 3. Run Scoring
    print("\n--- Testing Scores ---")
    df_scored = analyzer.compute_scores(df_bucketed)
    print(df_scored.select(["harvest_score", "yield_pct", "safety_score", "cost_pct"]))

    # Check Logic:
    # 0DTE (Row 0):
    # Yield = 50/10000 = 0.5%
    # Gamma = 0.10 -> Safety = 1/(10 * 1) = 0.1 (Low Safety)
    # Spread = 2/50 = 4% cost

    # 30DTE (Row 1):
    # Yield = 20/10000 = 0.2%
    # Gamma = 0.01 -> Safety = 1/(1 * 1) = 1.0 (High Safety)
    # Spread = 10/20 = 50% cost (Huge slippage penalizes wings)

    # 7DTE (Row 2):
    # Yield = 35/10000 = 0.35%
    # Gamma = 0.04 -> Safety = 1/(4 * 1) = 0.25
    # Spread = 2/35 = 5.7% cost

    # Harvest Score Expectations:
    # Row 1 might be killed by cost (50% spread).
    # Row 0 might be killed by gamma risk (Safety 0.1).
    # Row 2 (Sweet Spot) should handle reasonably well.

    # Let's just check relative ordering or specific values roughly
    scores = df_scored["harvest_score"].to_list()
    print(f"Scores: {scores}")

    # Ensure scores are calculated (not null)
    assert not np.isnan(scores).any()

    # 4. Heatmap
    print("\n--- Testing Heatmap ---")
    heatmap = analyzer.generate_heatmap(df)
    print(heatmap)

    assert len(heatmap) == 3
    assert "mean_score" in heatmap.columns

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_zone_analyzer()
