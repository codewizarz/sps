import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from quant_repo.portfolio.allocation import (
    CapitalAllocator,
    AllocatorConfig,
    StrategyMetrics,
    PortfolioState,
)


def test_allocation():
    print("[TEST] Capital Allocation Model...")

    allocator = CapitalAllocator()

    # 1. Base Case: Good Market, Good Edge
    metrics = StrategyMetrics(win_rate=0.55, avg_win=1000, avg_loss=-800)
    # Kelly: b = 1.25. p = 0.55. q = 0.45.
    # Full Kelly = 0.55 - (0.45 / 1.25) = 0.55 - 0.36 = 0.19 (19% Capital)

    state = PortfolioState(
        current_equity=100_000,
        current_vol=0.15,  # On Target
        current_drawdown=0.0,
    )

    config = AllocatorConfig(
        target_vol=0.15,
        kelly_fraction=0.5,  # Half Kelly -> ~9.5%
        max_dd_limit=0.20,
    )

    alloc_base = allocator.calculate_allocation_scaler(config, metrics, state)
    print(f"Base Allocation (Leverage): {alloc_base:.4f}")
    assert 0.08 < alloc_base < 0.11  # Expect ~0.095

    # 2. High Volatility Scenario
    # Vol doubles to 30%. Sizing should halve.
    state_high_vol = PortfolioState(100_000, 0.30, 0.0)
    alloc_vol = allocator.calculate_allocation_scaler(config, metrics, state_high_vol)
    print(f"High Vol Allocation: {alloc_vol:.4f}")
    assert abs(alloc_vol - (alloc_base / 2)) < 0.01

    # 3. Convex Drawdown Control
    # 50% of Limit (10% DD out of 20% Limit)
    # Expected Scaler: (1 - 0.5)^2 = 0.25 (Quarter of base)
    state_dd = PortfolioState(90_000, 0.15, 0.10)
    alloc_dd = allocator.calculate_allocation_scaler(config, metrics, state_dd)
    print(f"50% Drawdown Allocation: {alloc_dd:.4f}")

    expected_dd_alloc = alloc_base * 0.25
    assert abs(alloc_dd - expected_dd_alloc) < 0.01

    # 4. Near Ruin Check
    # 99% of Limit
    state_ruin = PortfolioState(80_000, 0.15, 0.198)
    alloc_ruin = allocator.calculate_allocation_scaler(config, metrics, state_ruin)
    print(f"99% Drawdown Allocation: {alloc_ruin:.6f}")
    assert alloc_ruin < 0.001

    print("[TEST] SUCCESS")


if __name__ == "__main__":
    test_allocation()
