import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import pytest
from quant_repo.analytics.gamma_regime import GammaRegimeClassifier, GammaRegime
from quant_repo.analytics.dealer_proxy import DealerPositioningSystem


def test_gamma_regime():
    print("[TEST] Gamma Regime Classifier...")

    dealer_system = DealerPositioningSystem()
    classifier = GammaRegimeClassifier(dealer_system)
    spot = 100.0

    # Scene 1: Clear Long Gamma (Puts dominate)
    df_long = pl.DataFrame(
        {
            "strike": [90.0, 100.0],
            "type": ["PUT", "PUT"],
            "open_interest": [5000, 1000],
            "gamma": [0.05, 0.05],
        }
    )

    print("\n--- Testing Long Gamma ---")
    signal_long = classifier.classify(df_long, spot)
    print(f"Regime: {signal_long.regime}")
    print(f"GEX: {signal_long.gex_normalized}")

    assert signal_long.regime == GammaRegime.LONG_GAMMA
    assert signal_long.gex_normalized > 0

    # Scene 2: Clear Short Gamma (Calls dominate)
    df_short = pl.DataFrame(
        {"strike": [110.0], "type": ["CALL"], "open_interest": [5000], "gamma": [0.05]}
    )

    print("\n--- Testing Short Gamma ---")
    signal_short = classifier.classify(df_short, spot)
    print(f"Regime: {signal_short.regime}")
    print(f"GEX: {signal_short.gex_normalized}")

    assert signal_short.regime == GammaRegime.SHORT_GAMMA
    assert signal_short.gex_normalized < 0

    # Scene 3: Transition Zone (Balanced, Flip near Spot)
    # Put at 99 (Positive), Call at 101 (Negative). Net approx 0.
    # Flip level likely around 100.
    df_transition = pl.DataFrame(
        {
            "strike": [99.0, 101.0],
            "type": ["PUT", "CALL"],
            "open_interest": [5000, 5000],
            "gamma": [0.05, 0.05],
        }
    )

    # dealer_proxy logic:
    # 99 Put: +GEX
    # 101 Call: -GEX
    # Flip level logic in dealer_proxy: Strike where sign flips.
    # Sorted Strikes: 99, 101.
    # 99: +GEX
    # 101: -GEX. Flip at 101?
    # If Spot is 100, and Flip is 101, dist is 1%.
    # Transition threshold default is 0.5%.
    # Let's check what dealer proxy calculates as flip.

    print("\n--- Testing Transition Zone ---")
    # Forcing flip EXACTLY at spot (100) requires a strike at 100 where sign changes?
    # Dealer Proxy logic finds "Strike where sign changes".
    # If we have 99 (+), 100 (-), flip is 100.
    # Spot 100. Dist 0%. Should be TRANSITION.

    df_trans_force = pl.DataFrame(
        {
            "strike": [99.0, 100.0],
            "type": ["PUT", "CALL"],
            "open_interest": [
                1000,
                2000,
            ],  # Call heavy -> Short Gamma overall, but flip happens
            "gamma": [0.05, 0.05],
        }
    )
    # 99 Put: +GEX
    # 100 Call: -GEX.
    # Flip at 100.
    # Spot 100.

    signal_trans = classifier.classify(df_trans_force, spot)
    print(f"Regime: {signal_trans.regime}")
    print(f"Flip Level implicitly used: {100.0}")  # Based on logic trace

    assert signal_trans.regime == GammaRegime.TRANSITION

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_gamma_regime()
