import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from quant_repo.execution.simulation import ExecutionSimulator, QuoteTick, OrderSide


def test_execution_simulation():
    print("[TEST] Execution Simulation Layer...")

    # Setup
    sim = ExecutionSimulator(
        baseline_iv=0.20, vol_spread_sensitivity=1.0, impact_coefficient=0.1
    )

    # Mock Tick
    # Mid = 100, Spread = 0.50 (Bid 99.75, Ask 100.25)
    tick = QuoteTick(
        bid_price=99.75,
        ask_price=100.25,
        bid_size=100,
        ask_size=100,
        mid_price=100.0,
        daily_volume=10_000,
    )

    # 1. Normal Vol, Small Size
    print("\n[TEST] 1. Baseline Scenario (IV=20%, Size=10)")
    res_base = sim.simulate(tick, OrderSide.BUY, size=10, current_iv=0.20)
    print(f"  Ask: {tick.ask_price}")
    print(f"  Fill: {res_base.fill_price}")
    print(f"  Slippage: {res_base.slippage_cost:.2f}")

    # Expect small impact. Price should be near 100.25 + epsilon
    assert res_base.fill_price >= tick.ask_price

    # 2. High Vol Scenario
    print("\n[TEST] 2. High Volatility (IV=40%)")
    # Spread should widen. Factor = (0.4/0.2)^1 = 2x.
    # New Spread = 0.5 * 2 = 1.0.
    # New Ask = 100 + 0.5 = 100.5.

    res_vol = sim.simulate(tick, OrderSide.BUY, size=10, current_iv=0.40)
    print(f"  Fill: {res_vol.fill_price}")

    # Expect significantly worse price than baseline
    assert res_vol.fill_price > res_base.fill_price
    assert res_vol.fill_price >= 100.50  # Approx check

    # 3. Large Size Scenario
    print("\n[TEST] 3. Whale Order (Size=5000)")
    # 50% of volume! huge impact.
    res_whale = sim.simulate(tick, OrderSide.BUY, size=5000, current_iv=0.20)
    print(f"  Fill: {res_whale.fill_price}")
    print(f"  Impact bps: {res_whale.impact_bps:.0f}")

    assert res_whale.fill_price > res_base.fill_price
    # Impact should be large

    print("[TEST] SUCCESS")


if __name__ == "__main__":
    test_execution_simulation()
