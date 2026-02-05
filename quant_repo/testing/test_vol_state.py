import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pytest
from quant_repo.features.vol_state import VolatilityStateManager, VolatilityState


def test_vol_state_manager():
    print("[TEST] Volatility State Manager...")

    manager = VolatilityStateManager()

    # Generate Synthetic Cycle
    # 100 days of history
    # 0-30: Suppressed (Low Vol)
    # 30-50: Expanding (Rising)
    # 50-60: Panic (Spike + Inversion)
    # 60-80: Normalization (Falling)
    # 80-100: Return to suppression

    n = 100
    iv = np.zeros(n)
    vix_3m = np.zeros(n)

    # 1. Suppressed
    # Flat line, no noise to guarantee slope <= 0.001
    iv[0:30] = np.linspace(0.10, 0.10, 30)
    vix_3m[0:30] = 0.12  # Contango

    # 2. Expanding
    iv[30:50] = np.linspace(0.10, 0.25, 20)
    vix_3m[30:50] = np.linspace(0.12, 0.26, 20)

    # 3. Panic (Spike)
    iv[50:60] = np.linspace(0.25, 0.80, 10)  # Huge spike
    vix_3m[50:60] = 0.50  # Backwardation (iv > vix_3m)

    # 4. Normalization (Crush)
    iv[60:80] = np.linspace(0.80, 0.30, 20)  # Falling fast
    vix_3m[60:80] = 0.40  # Normalizing

    # 5. Tail
    iv[80:100] = np.linspace(0.30, 0.15, 20)
    vix_3m[80:100] = 0.20

    df = pl.DataFrame({"iv": iv, "vix_3m": vix_3m})

    # Detect
    df_res = manager.detect_state(df)

    # Verification

    # Check Suppressed Phase (Index 10)
    state_suppressed = df_res["vol_state"][10]
    print(f"Index 10 State: {state_suppressed}")
    assert state_suppressed == VolatilityState.SUPPRESSED.value
    assert manager.get_exposure_scalar(VolatilityState.SUPPRESSED) == 0.5

    # Check Expanding Phase (Index 40)
    # IV is rising. Rank likely low-mid.
    state_expanding = df_res["vol_state"][40]
    print(f"Index 40 State: {state_expanding}")
    # Note: Depending on rank calc, might vary, but slope is positive.
    assert state_expanding == VolatilityState.EXPANDING.value
    assert manager.get_exposure_scalar(VolatilityState.EXPANDING) == 0.0

    # Check Panic Phase (Index 55)
    # IV ~0.50. Rank High. Inversion likely.
    state_panic = df_res["vol_state"][55]
    print(f"Index 55 State: {state_panic}")
    assert state_panic == VolatilityState.PANIC.value
    assert manager.get_exposure_scalar(VolatilityState.PANIC) == 0.0

    # Check Normalization Phase (Index 70)
    # IV ~0.55 but Falling (-0.025 per step).
    state_norm = df_res["vol_state"][70]
    print(f"Index 70 State: {state_norm}")
    assert state_norm == VolatilityState.NORMALIZATION.value
    assert manager.get_exposure_scalar(VolatilityState.NORMALIZATION) == 2.0

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_vol_state_manager()
