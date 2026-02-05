import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pytest
from quant_repo.strategies.vrp import RobustVRPStrategy
from quant_repo.portfolio.allocation import PortfolioState


def test_professional_vrp():
    print("[TEST] Professional VRP Strategy...")

    strategy = RobustVRPStrategy()

    # Mock Market Data (Need enough for Regime calc, but we assume regime mock mostly)
    # We will just reuse the structure from test_strategy_vrp but verify new logic

    # 1. Setup Data for BULL_QUIET (Low Vol, Rising Price)
    n = 300
    dates = pl.date_range(
        start=pl.date(2023, 1, 1), end=pl.date(2024, 1, 1), interval="1d", eager=True
    )
    # Ensure length matches n
    dates = dates[:n]

    close = np.linspace(100, 150, n) + np.random.normal(0, 1, n)
    iv = np.linspace(0.15, 0.12, n)

    # DataFrame
    market_data = pl.DataFrame(
        {
            "date": dates,
            "close": close,
            "iv": iv,
            "rv": iv - 0.02,  # Positive VRP
        }
    ).with_columns(
        [
            pl.col("close").rolling_mean(50).alias("sma_50"),
            pl.col("close").rolling_mean(200).alias("sma_200"),
            pl.lit(0.5).alias("iv_rank"),  # Mock
        ]
    )

    # 2. Test Circuit Breaker
    print("\n--- Testing Circuit Breaker ---")
    state_dd = PortfolioState(
        current_equity=100000, current_vol=0.15, current_drawdown=0.04
    )  # 4% DD
    instr_dd = strategy.generate_instruction(market_data, state_dd)

    print(f"Instruction: {instr_dd.action} - {instr_dd.description}")
    assert instr_dd.action == "FREEZE"
    assert "CIRCUIT BREAKER" in instr_dd.description

    # 3. Test Skew Adaptability (Steep Skew -> Ratio Spread)
    print("\n--- Testing Skew Logic (Steep) ---")
    state_ok = PortfolioState(
        current_equity=100000, current_vol=0.15, current_drawdown=0.01
    )

    # Inject Steep Skew signal into market_data
    market_data_steep = market_data.with_columns(
        pl.lit(0.10).alias("skew_slope")  # > 0.08
    )

    instr_steep = strategy.generate_instruction(market_data_steep, state_ok)
    print(f"Instruction: {instr_steep.structure_type}")
    assert instr_steep.structure_type == "PUT_RATIO_SPREAD_1x2"

    # 4. Test Skew Logic (Flat -> Iron Condor)
    print("\n--- Testing Skew Logic (Flat) ---")
    market_data_flat = market_data.with_columns(
        pl.lit(0.01).alias("skew_slope")  # < 0.03
    )
    instr_flat = strategy.generate_instruction(market_data_flat, state_ok)
    print(f"Instruction: {instr_flat.structure_type}")
    assert instr_flat.structure_type == "IRON_CONDOR"

    # 5. Gamma Kill Switch
    print("\n--- Testing Gamma Kill Switch ---")
    # Positions with DTE: [45, 30, 15]
    has_risk = strategy.check_gamma_risk([45, 30, 15])
    print(f"Has Gamma Risk ([45, 30, 15]): {has_risk}")
    assert has_risk == True

    safe_positions = strategy.check_gamma_risk([45, 30, 25])
    print(f"Has Gamma Risk ([45, 30, 25]): {safe_positions}")
    assert safe_positions == False

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_professional_vrp()
