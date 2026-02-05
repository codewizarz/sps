import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pytest
from unittest.mock import MagicMock
from quant_repo.strategies.vrp_deployable import DeployableVRPStrategy
from quant_repo.features.vol_state import VolatilityState


class MockVolManager:
    def __init__(self, target_state):
        self.target_state = target_state

    def detect_state(self, df):
        # Return df with 'vol_state' column set to target_state
        return df.with_columns(pl.lit(self.target_state.value).alias("vol_state"))

    def get_exposure_scalar(self, state):
        # Replicate logic or call original? simpler to replicate for test isolation
        if state == VolatilityState.SUPPRESSED:
            return 0.5
        if state == VolatilityState.EXPANDING:
            return 0.0
        if state == VolatilityState.PANIC:
            return 0.0
        if state == VolatilityState.NORMALIZATION:
            return 2.0
        return 1.0


def test_deployable_vrp():
    print("[TEST] Deployable VRP Strategy...")

    strategy = DeployableVRPStrategy(equity=1000000.0)

    # 1. Simulate Suppressed State
    print("\n--- Test Scenario: SUPPRESSED ---")

    # Inject Mock Manager
    strategy.vol_manager = MockVolManager(VolatilityState.SUPPRESSED)

    # Dummy Data (Content doesn't matter now as we mock detection)
    df_data = pl.DataFrame({"close": [100.0]})

    signals = strategy.generate_signals(df_data, "2023-01-01")

    for s in signals:
        print(f" Signal: {s.order_type} {s.instrument_type} | {s.rationale}")

    # Expectation: BUY HEDGE, SELL INCOME
    has_hedge = any("HEDGE" in s.rationale for s in signals if s.order_type == "BUY")
    has_income = any("INCOME" in s.rationale for s in signals if s.order_type == "SELL")

    assert has_hedge, "Expected Hedge in Suppressed State"
    assert has_income, "Expected Income in Suppressed State"

    # 2. Simulate Panic State
    print("\n--- Test Scenario: PANIC ---")

    strategy.vol_manager = MockVolManager(VolatilityState.PANIC)
    signals_panic = strategy.generate_signals(df_data, "2023-03-15")

    for s in signals_panic:
        print(f" Signal: {s.order_type} {s.instrument_type} | {s.rationale}")

    # Expectation: MONETIZE HEDGE, NO INCOME
    has_monetize = any("MONETIZE" in s.rationale for s in signals_panic)
    has_income_panic = any("INCOME" in s.rationale for s in signals_panic)

    assert has_monetize, "Expected Monetization in Panic"
    assert not has_income_panic, "Expected NO Income in Panic"

    # 3. Simulate Normalization
    print("\n--- Test Scenario: NORMALIZATION ---")
    strategy.vol_manager = MockVolManager(VolatilityState.NORMALIZATION)
    signals_norm = strategy.generate_signals(df_data, "2023-04-01")

    for s in signals_norm:
        print(f" Signal: {s.order_type} {s.instrument_type} | {s.rationale}")

    # Expectation: MONETIZE (if any), MASSIVE INCOME (Scalar 2.0)
    has_aggressive_income = any("Scalar 2.0" in s.rationale for s in signals_norm)
    assert has_aggressive_income, "Expected Aggressive Income (2.0x) in Normalization"

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_deployable_vrp()
