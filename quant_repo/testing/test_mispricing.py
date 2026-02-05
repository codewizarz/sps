import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pandas as pd
from quant_repo.signals.mispricing import MispricingDetector
from quant_repo.signals.definitions import SignalType, Direction


def test_vol_arb_logic():
    print("[TEST] 1. Volatility Arbitrage Logic...")

    # 1. Generate Synthetic Data
    # 100 Days of data
    # Underlying follows a random walk (low vol)
    # IV stays high consistently

    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    timestamps = dates.view(np.int64)  # nanos

    # Low RV: Prices don't move much
    # Spot = 100 * exp(cumsum(random * 0.005)) roughly 0.5% daily ~ 8% annual
    np.random.seed(42)
    returns = np.random.normal(0, 0.005, 200)
    price_path = 100 * np.exp(np.cumsum(returns))

    # High IV: 20% flat
    iv_path = np.ones(200) * 0.20

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "instrument_id": "OPT-TEST",
            "spot_price": price_path,
            "iv": iv_path,
        }
    ).lazy()

    detector = MispricingDetector()

    # Run Detector
    # Window=20 for RV, Window=50 for Z-Score
    signals = detector.detect_vol_arb(
        df,
        underlying_col="spot_price",
        iv_col="iv",
        rv_window=20,
        z_window=50,
        threshold=1.5,
    )

    print(f"[TEST] Generated {len(signals)} signals")
    # We expect signals because IV (20%) > RV (~8%). Spread is POSITIVE.
    # Z-Score should be POSITIVE.
    # Direction should be SHORT (Sell expensive Vol).

    assert len(signals) > 0
    last_sig = signals[-1]
    print(f"  Last Signal: {last_sig.direction} Strength={last_sig.strength:.2f}")
    print(f"  Metadata: {last_sig.metadata}")

    assert last_sig.signal_type == SignalType.VOL_ARBITRAGE
    assert last_sig.direction == Direction.SHORT

    print("[TEST] Vol Arb Passed.")


def test_parity_logic():
    print("[TEST] 2. Put-Call Parity Logic...")

    # S = 100, K = 100, r = 0, T = 1 year
    # Parity: C - P = S - K (since r=0, df=1) => C - P = 0
    # Let's make C = 10, P = 5. diff = 5. Real parity diff should be 0.
    # Error = 5. > cost 0.5. Signal!

    ts = pd.Timestamp("2024-01-01").value
    expiry = pd.Timestamp("2025-01-01").value

    calls = pl.DataFrame(
        {"timestamp": [ts], "expiry": [expiry], "strike": [100.0], "close": [10.0]}
    ).lazy()

    puts = pl.DataFrame(
        {"timestamp": [ts], "expiry": [expiry], "strike": [100.0], "close": [5.0]}
    ).lazy()

    spots = pl.DataFrame({"timestamp": [ts], "close_spot": [100.0]}).lazy()

    detector = MispricingDetector()
    signals = detector.detect_parity_violations(
        calls, puts, spots, r=0.0, cost_threshold=0.5
    )

    print(f"[TEST] Generated {len(signals)} signals")
    assert len(signals) == 1
    sig = signals[0]

    # Market Diff: 10 - 5 = 5
    # Theo Diff: 100 - 100 = 0
    # Error = 5. Positive. C is expensive relative to P/S.
    # Should SHORT the synthetic (Sell C, Buy P).
    print(f"  Signal: {sig.direction} Strength={sig.strength:.2f}")

    assert sig.direction == Direction.SHORT
    print("[TEST] Parity Passed.")


if __name__ == "__main__":
    test_vol_arb_logic()
    test_parity_logic()
