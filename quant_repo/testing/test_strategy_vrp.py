import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pandas as pd
from quant_repo.strategies.vrp import RobustVRPStrategy, TradeInstruction
from quant_repo.portfolio.allocation import PortfolioState


def test_vrp_strategy():
    print("[TEST] Robust VRP Strategy...")

    strategy = RobustVRPStrategy()

    # 1. Generate Data for Regimes
    # Create 500 days history to stabilize rolling windows
    dates = pd.date_range("2021-01-01", periods=500, freq="B")

    # A. Bull Quiet Data (Last day)
    # Price Rising, IV Low.
    price = np.linspace(100, 150, 500)
    iv = np.linspace(0.20, 0.12, 500)  # Falling IV -> Rank Low
    rv = iv - 0.02

    df_bull = pl.DataFrame({"timestamp": dates, "close": price, "iv": iv, "rv": rv})

    # State: Healthy
    state_healthy = PortfolioState(
        current_equity=100_000, current_vol=0.10, current_drawdown=0.0
    )

    # Check Bull Quiet Instruction
    instr_bull = strategy.generate_instruction(df_bull, state_healthy)
    print(f"\nScenario: Bull Quiet")
    print(f"Instruction: {instr_bull}")

    assert instr_bull.structure_type == "IRON_CONDOR"
    assert instr_bull.action == "OPEN"
    assert instr_bull.allocation_scalar > 0.0

    # B. Crisis Data
    # Last few days VRP turns negative
    rv_crisis = rv.copy()
    rv_crisis[-10:] = iv[-10:] + 0.10  # Huge RV spike over IV

    df_crisis = pl.DataFrame(
        {
            "timestamp": dates,
            "close": price,  # Price check unimportant for Crisis override
            "iv": iv,
            "rv": rv_crisis,
        }
    )

    instr_crisis = strategy.generate_instruction(df_crisis, state_healthy)
    print(f"\nScenario: Crisis")
    print(f"Instruction: {instr_crisis}")

    assert instr_crisis.structure_type == "CASH"
    assert instr_crisis.action == "FLATTEN"
    assert instr_crisis.allocation_scalar == 0.0

    # C. Capital Sizing Check (High Drawdown)
    state_dd = PortfolioState(
        current_equity=100_000,
        current_vol=0.10,
        current_drawdown=0.08,  # 8% Drawdown (Limit is 10%)
    )
    # Using Bull market data, but bad portfolio state
    instr_dd = strategy.generate_instruction(df_bull, state_dd)
    print(f"\nScenario: High Drawdown (8% on 10% limit)")
    print(f"Instruction: {instr_dd}")

    # Convex Sizing: (1 - 0.8)^2 = 0.04 factor.
    # Base size is based on Kelly (0.3 of Full) + Vol.
    # Should be much smaller than instr_bull
    print(f"Base Scalar: {instr_bull.allocation_scalar:.4f}")
    print(f"DD Scalar: {instr_dd.allocation_scalar:.4f}")

    assert instr_dd.allocation_scalar < (instr_bull.allocation_scalar * 0.10)

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_vrp_strategy()
