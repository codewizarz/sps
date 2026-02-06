import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import pytest
from quant_repo.analytics.ranking import OpportunityScorer
from quant_repo.analytics.gamma_regime import GammaRegime


def test_opportunity_scorer():
    print("[TEST] Mispricing Severity Scorer...")

    scorer = OpportunityScorer()

    # Scene 1: The "Perfect" Trade
    # Rare (3 sigma), High Edge, Long Gamma Regime, Tight Spreads, Low Risk.
    sig_perfect = {
        "id": "SIG_001",
        "z_score": 3.0,  # 60 pts (capped 100 at 5)
        "strength": 0.8,  # 80 pts
        "strategy_type": "SHORT_VOL",
    }
    mkt_perfect = {
        "bid_ask_spread": 0.05,
        "price": 100.0,  # 5bps spread -> Score ~95
        "tail_risk_prob": 0.01,  # 1% -> Score ~98
    }
    regime = GammaRegime.LONG_GAMMA  # Matches Short Vol -> 100 pts

    print("\n--- Scoring Perfect Trade ---")
    res = scorer.score_opportunity(sig_perfect, regime, mkt_perfect)
    print(f"Composite: {res.composite_score:.2f}")
    print(f"Breakdown: {res.breakdown}")

    # Calc:
    # Rarity: 3 * 20 = 60 * 0.3 = 18
    # Edge: 80 * 0.3 = 24
    # Regime: 100 * 0.2 = 20
    # Liq: (100 - 5) = 95 * 0.1 = 9.5
    # Tail: (100 - 2) = 98 * 0.1 = 9.8
    # Total: 18 + 24 + 20 + 9.5 + 9.8 = 81.3

    assert res.composite_score > 80.0

    # Scene 2: The "Suicide" Trade
    # Short Vol in Short Gamma Regime.
    sig_bad = {
        "id": "SIG_002",
        "z_score": 1.0,  # 20 pts
        "strength": 0.5,  # 50 pts
        "strategy_type": "SHORT_VOL",
    }
    regime_bad = GammaRegime.SHORT_GAMMA  # Mismatch -> 0 pts

    print("\n--- Scoring Bad Regime Trade ---")
    res_bad = scorer.score_opportunity(sig_bad, regime_bad, mkt_perfect)
    print(f"Composite: {res_bad.composite_score:.2f}")
    print(f"Regime Score: {res_bad.breakdown['regime']}")

    # Regime score should be 0.
    assert res_bad.breakdown["regime"] == 0.0
    assert res_bad.composite_score < res.composite_score

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_opportunity_scorer()
