import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pandas as pd
from quant_repo.features.regime import RegimeClassifier, MarketRegime


def test_regime_classifier():
    print("[TEST] Regime Classifier...")

    # 1. Generate Data (700 days total)

    # Phase 1: Bull Quiet (0-500)
    # Price rises, Low IV
    # Length 500 to satisfy 252-day rolling window
    price_bull = np.linspace(100, 150, 500) + np.random.normal(0, 1, 500)
    iv_bull = np.linspace(0.15, 0.12, 500)  # Low/Falling IV
    rv_bull = iv_bull - 0.02  # Positive VRP

    # Phase 2: Bear Volatile (500-600)
    # Price Crashes, SMA 50 crosses below 200 eventually
    price_bear = np.linspace(150, 100, 100) + np.random.normal(0, 2, 100)
    iv_bear = np.linspace(0.20, 0.40, 100)  # Spiking IV
    rv_bear = iv_bear - 0.05

    # Phase 3: Crisis (600-700)
    # Price choppy, but RV > IV
    price_crisis = np.linspace(100, 90, 100) + np.random.normal(0, 5, 100)
    iv_crisis = 0.30 * np.ones(100)
    rv_crisis = 0.40 * np.ones(100)  # Negative VRP (-10%)

    # Stitch
    prices = np.concatenate([price_bull, price_bear, price_crisis])
    ivs = np.concatenate([iv_bull, iv_bear, iv_crisis])
    rvs = np.concatenate([rv_bull, rv_bear, rv_crisis])

    dates = pd.date_range("2021-01-01", periods=len(prices), freq="B")

    df = pl.DataFrame({"timestamp": dates, "close": prices, "iv": ivs, "rv": rvs})

    classifier = RegimeClassifier()
    df_res = classifier.detect_regime(df)

    # 2. Check Classifications

    # A. Bull Quiet Check (Index ~400)
    # At 400: Bull trend is mature. IV rank window is full.
    row = df_res.row(400, named=True)
    print(
        f"Index 400: Close={row['close']:.2f}, IVRank={row['iv_rank_pct']:.2f}, Regime={row['regime']}"
    )
    assert row["regime"] == MarketRegime.BULL_QUIET.value

    # B. Bear Volatile Check (Index 550) - In Bear Phase (500-600)
    # Price falling fast. IV Spiked.
    row_bear = df_res.row(550, named=True)
    print(f"Index 550 Regime: {row_bear['regime']}")
    # Assert broadly Bearish (Quiet or Volatile depends on IV Rank threshold)
    assert "BEAR" in row_bear["regime"]

    # C. Crisis Check (Index 650) - In Crisis Phase (600-700)
    # RV (0.40) > IV (0.30). Diff = -0.10.
    row_crisis = df_res.row(650, named=True)
    print(f"Index 650 Regime: {row_crisis['regime']}")

    assert row_crisis["regime"] == MarketRegime.CRISIS.value

    print("[TEST] SUCCESS")


if __name__ == "__main__":
    test_regime_classifier()
