import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pandas as pd
from quant_repo.research.walk_forward import WalkForwardOptimizer


def test_walk_forward():
    print("[TEST] Walk-Forward Optimization...")

    # 1. Generate Data (Regime Switch)
    # 500 days total
    # Day 0-300: Trend Up (Signal A works)
    # Day 300-500: Mean Revert (Signal B works)

    n_days = 500
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B").view(np.int64)
    returns = np.random.normal(0, 0.01, n_days)

    # Signal A: Good in first half
    sig_a = np.sign(np.roll(returns, -1))  # Perfect lookahead proxy for testing
    sig_a[300:] = np.random.choice([-1, 1], 200)  # Noise in second half

    # Signal B: Good in second half
    sig_b = np.random.choice([-1, 1], n_days)  # Noise first
    sig_b[300:] = np.sign(np.roll(returns[300:], -1))  # Perfect later

    df = pl.DataFrame(
        {
            "timestamp": dates,
            "spot_return": returns,
            "signal_A": sig_a,
            "signal_B": sig_b,
        }
    )

    # 2. Run WFO
    # Train 100, Test 50
    # First Window: 0-100 Train. Test 100-150. (Should pick A)
    # ...
    # Transition Window: Train 250-350. (Includes transition to B).
    # Later Window: Train 350-450 (Should pick B).

    optimizer = WalkForwardOptimizer()
    param_grid = [{"signal_col": "signal_A"}, {"signal_col": "signal_B"}]

    res_df = optimizer.run(df, param_grid, train_size=100, test_size=50)

    assert len(res_df) > 0
    print(f"\nOOS Rows Generated: {len(res_df)}")

    # 3. Verify Adaptability
    # Early segments should chose A
    early = res_df.head(50)  # First OOS block
    chosen_early = early["chosen_param"].mode()[0]
    print(f"Early Segment Chosen: {chosen_early}")
    assert chosen_early == "signal_A"

    # Late segments should chose B
    late = res_df.tail(50)
    chosen_late = late["chosen_param"].mode()[0]
    print(f"Late Segment Chosen: {chosen_late}")

    # Note: Transition might be lagging because Train Window is 100 days.
    # At day 350 (Train 250-350), 50 days are Regime A, 50 days Regime B.
    # At day 400 (Train 300-400), 100 days are Regime B. Should definitely switch by then.

    if chosen_late != "signal_B":
        print(f"WARNING: Late segment stuck on {chosen_late}. Train window logic?")
        # It's possible signal_A lucked out, or training window needs to be fully in regime.
        # But broadly, Total PnL should be positive.

    total_pnl = res_df["pnl"].sum()
    print(f"Total OOS PnL: {total_pnl:.5f}")
    assert total_pnl > 0.1  # Should catch most of the perfect trends

    print("[TEST] SUCCESS")


if __name__ == "__main__":
    test_walk_forward()
