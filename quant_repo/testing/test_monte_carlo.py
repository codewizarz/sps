import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import numpy as np
from quant_repo.analytics.monte_carlo import MonteCarloEngine, SimulationConfig


def test_monte_carlo():
    print("[TEST] Monte Carlo Engine...")

    # 1. Generate Mock Trade PnLs
    # 500 Trades
    # 60% Win Rate. Win=1000, Loss=-1000.
    np.random.seed(42)
    n_trades = 500
    pnl = np.where(np.random.rand(n_trades) > 0.4, 1000, -1000).astype(float)

    # Add a fat tail loss manually to source data?
    # No, let the Monte Carlo find clusters of the -1000s.

    engine = MonteCarloEngine()

    # 2. Run Baseline (Light Stress)
    config = SimulationConfig(
        n_sims=1000, initial_capital=50_000, prob_skip_fill=0.0, prob_slippage_shock=0.0
    )

    res_base = engine.run(pnl, config)

    print("\n--- Baseline ---")
    print(f"Median DD: {res_base.median_dd:.2f}")
    print(f"Worst Case DD (99%): {res_base.worst_case_dd:.2f}")
    print(f"Ruin Prob: {res_base.ruin_probability:.2%}")

    assert res_base.worst_case_dd > res_base.median_dd

    # 3. Run Stressed (High Shock)
    config_stress = SimulationConfig(
        n_sims=1000,
        initial_capital=50_000,
        prob_skip_fill=0.10,  # 10% of winners skipped
        prob_slippage_shock=0.05,  # 5% of trades shocked
        shock_factor=3.0,  # Big shocks
    )

    res_stress = engine.run(pnl, config_stress)

    print("\n--- Stressed ---")
    print(f"Median DD: {res_stress.median_dd:.2f}")
    print(f"Worst Case DD (99%): {res_stress.worst_case_dd:.2f}")
    print(f"Ruin Prob: {res_stress.ruin_probability:.2%}")

    # Assert Stress worsens outcomes
    assert res_stress.median_dd > res_base.median_dd
    assert res_stress.worst_case_dd > res_base.worst_case_dd
    assert res_stress.var_99_equity < res_base.var_99_equity

    print("[TEST] SUCCESS")


if __name__ == "__main__":
    test_monte_carlo()
