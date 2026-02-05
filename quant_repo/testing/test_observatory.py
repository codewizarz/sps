import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pytest
from quant_repo.analytics.observatory import VolatilityObservatory


def test_volatility_observatory():
    print("[TEST] Volatility Observatory...")

    observatory = VolatilityObservatory()

    # Generate Synthetic History
    # 20 days of Normal, 1 day of Panic

    dates = pl.date_range(
        start=pl.date(2023, 1, 1), end=pl.date(2023, 1, 25), interval="1d", eager=True
    )
    n = len(dates)

    # Normal Regime features
    iv = np.full(n, 0.15)
    iv_put_25 = np.full(n, 0.18)  # Skew normal
    vix_3m = np.full(n, 0.17)  # Contango (Back > Front)
    rv = np.full(n, 0.10)  # Positive VRP

    # Panic Day (Last Day)
    iv[-1] = 0.40  # Huge spike
    vix_3m[-1] = 0.35  # Backwardation (Front > Back)
    rv[-1] = 0.45  # Negative VRP (Realized > Implied potentially, or just high)

    df = pl.DataFrame(
        {"date": dates, "iv": iv, "iv_put_25": iv_put_25, "vix_3m": vix_3m, "rv": rv}
    )

    # 1. Analyze Normal Day
    print("\n--- Analyzing Normal Day ---")
    normal_date = str(dates[10])  # Middle of normal period
    status_normal = observatory.analyze_market(df, normal_date)

    print(f"Date: {status_normal.date}")
    print(f"Regime: {status_normal.regime}")
    print(f"Term Slope: {status_normal.term_structure_slope:.4f}")
    print(f"Warnings: {status_normal.warnings}")

    assert status_normal.term_structure_slope < 0  # Contango
    assert status_normal.vrp_spread > 0  # Positive VRP
    assert len(status_normal.warnings) == 0

    # 2. Analyze Panic Day
    print("\n--- Analyzing Panic Day ---")
    panic_date = str(dates[-1])
    status_panic = observatory.analyze_market(df, panic_date)

    print(f"Date: {status_panic.date}")
    print(f"Regime: {status_panic.regime}")
    print(f"Term Slope: {status_panic.term_structure_slope:.4f}")
    print(f"Warnings: {status_panic.warnings}")

    assert status_panic.term_structure_slope > 0  # Backwardation
    assert "Backwardation Alert (Panic Structure)" in status_panic.warnings
    assert "Negative VRP (Selling is -EV)" in status_panic.warnings

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_volatility_observatory()
