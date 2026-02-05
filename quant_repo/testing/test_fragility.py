import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pytest
from quant_repo.validation.fragility import FragilityAnalyzer


def test_fragility_analyzer():
    print("[TEST] Fragility Analyzer...")

    analyzer = FragilityAnalyzer()

    # 1. Test Execution Stress
    print("\n--- Test Execution Stress ---")
    # Mock Trades: PnL = 100, Cost = 10
    # Net PnL = 100. (We assume PnL column includes cost usually, but logic in analyzer expects PnL and Cost separate to reverse engineer Revenue?)
    # Let's check logic: PnL + Cost = Revenue. New PnL = Revenue - (Cost * Mult).
    # If trades_df["pnl"] is Net PnL.

    trades_df = pl.DataFrame(
        {
            "pnl": [100.0, 200.0],  # Net PnL
            "slippage_cost": [10.0, 20.0],  # Cost incurred
        }
    )
    # Total PnL = 300. Total Cost = 30.
    # At 2x Slippage: Cost becomes 60. Extra Cost = 30.
    # New Net PnL should be 300 - 30 = 270.

    df_exec = analyzer.stress_execution(trades_df, slippage_multipliers=[1.0, 2.0, 5.0])
    print(df_exec)

    pnl_1x = df_exec.filter(pl.col("slippage_mult") == 1.0)["total_pnl"][0]
    pnl_2x = df_exec.filter(pl.col("slippage_mult") == 2.0)["total_pnl"][0]

    assert pnl_1x == 300.0
    assert pnl_2x == 270.0

    # 2. Test Parameter Stress
    print("\n--- Test Parameter Stress ---")

    # Mock Backtest: Metric = x + y
    def mock_backtest(params):
        return params["x"] + params["y"]

    param_grid = {"x": [1, 2], "y": [10, 20]}
    # Combinations: (1,10)=11, (1,20)=21, (2,10)=12, (2,20)=22

    df_params = analyzer.stress_parameters(mock_backtest, param_grid)
    print(df_params)

    assert df_params.height == 4
    assert df_params["metric"].min() == 11
    assert df_params["metric"].max() == 22

    # 3. Test Fragility Score
    print("\n--- Test Fragility Score ---")
    score = analyzer.compute_fragility_score(df_params)
    print(f"Fragility Score (CV): {score}")

    assert score > 0.0

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_fragility_analyzer()
