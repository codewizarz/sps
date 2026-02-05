import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import pytest
from quant_repo.risk.hedging import HedgeManager, HedgeType, VolatilityState


def test_hedge_manager():
    print("[TEST] Tail Hedging Manager...")

    manager = HedgeManager()

    # 1. Test Suppressed State (Buy Insurance)
    state_suppressed = VolatilityState.SUPPRESSED
    instr_suppressed = manager.select_hedge(state_suppressed)

    print(
        f"State: SUPPRESSED -> {instr_suppressed.hedge_type}, Alloc: {instr_suppressed.allocation_pct}"
    )
    assert instr_suppressed.hedge_type == HedgeType.LONG_PUT
    assert instr_suppressed.allocation_pct == 0.01

    # 2. Test Expanding State (Ratio Spread)
    state_expanding = VolatilityState.EXPANDING
    instr_expanding = manager.select_hedge(state_expanding)

    print(
        f"State: EXPANDING -> {instr_expanding.hedge_type}, Alloc: {instr_expanding.allocation_pct}"
    )
    assert instr_expanding.hedge_type == HedgeType.RATIO_SPREAD

    # 3. Test Panic State (Monetize)
    state_panic = VolatilityState.PANIC
    instr_panic = manager.select_hedge(state_panic)

    print(f"State: PANIC -> {instr_panic.hedge_type}")
    assert instr_panic.hedge_type == HedgeType.CLOSE_ALL
    assert instr_panic.allocation_pct == 0.0

    # 4. Test Normalization (Monetize/None)
    state_norm = VolatilityState.NORMALIZATION
    instr_norm = manager.select_hedge(state_norm)

    print(f"State: NORMALIZATION -> {instr_norm.hedge_type}")
    assert instr_norm.hedge_type == HedgeType.CLOSE_ALL

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_hedge_manager()
