import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pytest
from quant_repo.analytics.dealer_proxy import DealerPositioningSystem


def test_dealer_proxy():
    print("[TEST] Dealer Positioning Proxy...")

    proxy = DealerPositioningSystem()
    spot = 100.0

    # Generate Synthetic Option Chain
    # Scenario: Puts at 90 (Dealer Short Puts -> Long Gamma)
    #           Calls at 110 (Dealer Short Calls -> Short Gamma)
    #           Spot at 100.

    # 90 Put: OI 1000, Gamma 0.05. GEX = 0.05 * 1000 * 100 * 100 * (+1) = +500,000 ?
    # Logic in code: Gamma * OI * 100 * Spot * Sign.
    # 0.05 * 1000 * 100 * 100 * 1 = 500,000.

    # 110 Call: OI 1000, Gamma 0.05. GEX = 0.05 * 1000 * 100 * 100 * (-1) = -500,000.

    # Net GEX should be near 0 if balanced.

    df = pl.DataFrame(
        {
            "strike": [90.0, 100.0, 110.0, 120.0],
            "type": ["PUT", "PUT", "CALL", "CALL"],
            "open_interest": [2000, 500, 2000, 500],
            "gamma": [0.05, 0.08, 0.05, 0.02],
        }
    )

    # 1. Standard Calculation
    print("\n--- Standard GEX ---")
    profile = proxy.calculate_gex(df, spot, retail_net_short=False)

    print(f"Total GEX: ${profile.total_gex_dollars:,.2f}")
    print(f"Regime: {profile.regime}")
    print(f"Flip Level: {profile.zero_gamma_level}")
    print(f"Magnets: {profile.dominant_magnets}")

    # Calculation Check:
    # 90 Put: 0.05 * 2000 * 100 * 100 * (+1) = +1,000,000
    # 100 Put: 0.08 * 500 * 100 * 100 * (+1) = +400,000
    # 110 Call: 0.05 * 2000 * 100 * 100 * (-1) = -1,000,000
    # 120 Call: 0.02 * 500 * 100 * 100 * (-1) = -100,000
    # Net: 1M + 400k - 1M - 100k = +300,000.

    expected_gex = 300000.0
    assert profile.total_gex_dollars == expected_gex
    assert profile.regime == "Long Gamma"

    # Magnets should be 90 and 110 (2000 OI each)
    assert 90.0 in profile.dominant_magnets
    assert 110.0 in profile.dominant_magnets

    # Flip Level check
    # 90: +1M (Cum +1M)
    # 100: +400k (Cum +1.4M)
    # 110: -1M (Cum +0.4M)
    # 120: -100k (Cum +0.3M)

    # Wait, my flip logic was: Scan per-strike GEX and find sign change.
    # 90: +1M
    # 100: +400k
    # 110: -1M -> Sign Flip here! from + to -.
    # So Flip Level should be 110.0.

    assert profile.zero_gamma_level == 110.0

    # 2. Retail Net Short Calculation (Inverted)
    print("\n--- Retail Short (Inverted) GEX ---")
    profile_inv = proxy.calculate_gex(df, spot, retail_net_short=True)

    # Result should be inverted: -300,000
    print(f"Total GEX (Inv): ${profile_inv.total_gex_dollars:,.2f}")
    assert profile_inv.total_gex_dollars == -300000.0
    assert profile_inv.regime == "Short Gamma"

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_dealer_proxy()
