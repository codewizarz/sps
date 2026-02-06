import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from quant_repo.analytics.gamma_regime import GammaRegime


@dataclass
class ScoredOpportunity:
    signal_id: str
    composite_score: float  # 0 to 100
    breakdown: Dict[str, float]  # Component scores


class OpportunityScorer:
    """
    Ranks trading opportunities based on weighted multi-factor scoring.
    """

    def __init__(self):
        # Configuration
        self.weights = {
            "rarity": 0.30,
            "edge": 0.30,
            "regime": 0.20,
            "liquidity": 0.10,
            "tail_risk": 0.10,
        }

    def score_opportunity(
        self, signal: Dict[str, Any], regime: GammaRegime, market_stats: Dict[str, Any]
    ) -> ScoredOpportunity:
        """
        Calculates composite score for a single trade signal.
        signal: {z_score, expected_pnl, strategy_type, ...}
        market_stats: {spread, price, tail_risk_prob}
        """

        # 1. Rarity Score (0-100)
        # Cap at 5 sigma. 1 sigma = 20 pts.
        z_score = abs(signal.get("z_score", 0.0))
        score_rarity = min(z_score * 20.0, 100.0)

        # 2. Edge Score (0-100)
        # Log scaling of PnL? Or simple linear cap.
        # Let's assume expected yield %. 1% = 20pts? 5% = 100pts?
        # Or absolute PnL. Let's use annualized yield or just raw PnL if standardized.
        # Simplification: Signal gives 'strength' 0-1.
        strength = signal.get("strength", 0.0)
        score_edge = min(strength * 100.0, 100.0)

        # 3. Regime Alignment Score (0 or 100)
        # Strategy types: "SHORT_VOL", "LONG_VOL"
        strategy_type = signal.get("strategy_type", "SHORT_VOL")

        score_regime = 50.0  # Neutral default

        if regime == GammaRegime.LONG_GAMMA:  # Safe for Short Vol
            if strategy_type == "SHORT_VOL":
                score_regime = 100.0
            elif strategy_type == "LONG_VOL":
                score_regime = 0.0
        elif regime == GammaRegime.SHORT_GAMMA:  # Dangerous for Short Vol
            if strategy_type == "SHORT_VOL":
                score_regime = 0.0
            elif strategy_type == "LONG_VOL":
                score_regime = 100.0
        elif regime == GammaRegime.TRANSITION:
            score_regime = 50.0

        # 4. Liquidity Score (0-100)
        # Spread / Price. If > 0.5%, score drops.
        spread = market_stats.get("bid_ask_spread", 0.0)
        price = market_stats.get("price", 100.0)
        if price > 0:
            rel_spread = spread / price
            # 10bps (0.001) -> 90 score. 100bps (0.01) -> 0 score.
            # Formula: 100 - (rel_spread * 10000)
            score_liquidity = max(100.0 - (rel_spread * 10000.0), 0.0)
        else:
            score_liquidity = 0.0

        # 5. Tail Risk Penalty (0-100, where 100 is Safe)
        # tail_risk_prob: 0.0 to 1.0.
        tail_prob = market_stats.get("tail_risk_prob", 0.0)
        # If 0% prob -> 100 score.
        # If 50% prob -> 0 score.
        score_tail = max(100.0 - (tail_prob * 200.0), 0.0)

        # Weighted Sum
        composite = (
            score_rarity * self.weights["rarity"]
            + score_edge * self.weights["edge"]
            + score_regime * self.weights["regime"]
            + score_liquidity * self.weights["liquidity"]
            + score_tail * self.weights["tail_risk"]
        )

        return ScoredOpportunity(
            signal_id=signal.get("id", "unknown"),
            composite_score=composite,
            breakdown={
                "rarity": score_rarity,
                "edge": score_edge,
                "regime": score_regime,
                "liquidity": score_liquidity,
                "tail": score_tail,
            },
        )
