import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import pandas as pd
import numpy as np
import polars as pl
from quant_repo.portfolio.construction import PortfolioOptimizer


def test_portfolio_optimizer():
    print("[TEST] Portfolio Construction Engine...")

    optimizer = PortfolioOptimizer()

    # 1. Generate Correlated Data
    np.random.seed(42)
    n = 1000

    # Strat A: Trend
    ret_a = np.random.normal(0.001, 0.02, n)

    # Strat B: Trend (Highly correlated to A)
    # R ~ 0.9
    ret_b = ret_a * 0.9 + np.random.normal(0, 0.005, n)

    # Strat C: Hedge (Uncorrelated, Positive Skew)
    # Long Vol: Small bleed, occasional big pop
    ret_c = np.random.normal(-0.0005, 0.01, n)
    # Add skew (outliers)
    indices = np.random.choice(n, size=20, replace=False)
    ret_c[indices] = 0.10  # +10% gains

    df_returns = pd.DataFrame({"Trend_A": ret_a, "Trend_B": ret_b, "Hedge_C": ret_c})

    print("\n--- Correlation Matrix ---")
    print(df_returns.corr().round(2))

    print("\n--- Skewness ---")
    print(df_returns.skew().round(2))

    # 2. Optimize Weights
    weights = optimizer.optimize_weights(df_returns, use_convexity=True)

    print("\n--- Optimal Weights (HRP + Convexity) ---")
    for s, w in weights.items():
        print(f"{s}: {w:.4f}")

    # 3. Assertions
    # A and B are one cluster. C is another.
    # Naive Risk Parity would see A and B as separate high vol assets.
    # HRP sees them as one group.

    # Combined weight of A+B should be roughly equal to C (2 clusters)
    # But C has positive skew, so it should get a BOOST.

    w_trend = weights["Trend_A"] + weights["Trend_B"]
    w_hedge = weights["Hedge_C"]

    print(f"\nTrend Cluster Weight: {w_trend:.4f}")
    print(f"Hedge Cluster Weight: {w_hedge:.4f}")

    # Hedge should have significant weight despite being 1 asset vs 2
    # Because A and B share risk.
    assert w_hedge > 0.40  # At least 40% to the hedge

    # Convexity Check
    # Disable convexity to see difference
    weights_no_conv = optimizer.optimize_weights(df_returns, use_convexity=False)
    print(f"Hedge Weight (No Convexity): {weights_no_conv['Hedge_C']:.4f}")

    assert weights["Hedge_C"] > weights_no_conv["Hedge_C"]

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_portfolio_optimizer()
