from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import polars as pl
from quant_repo.features.vol_state import VolatilityState


class HedgeType(Enum):
    NONE = "NONE"
    LONG_PUT = "LONG_PUT"  # 5-Delta Puts (Black Swan)
    RATIO_SPREAD = "RATIO_SPREAD"  # 1x2 Put Ratio (Backspread)
    DEBIT_SPREAD = "DEBIT_SPREAD"  # ATM Put Spread (Directional)
    CLOSE_ALL = "CLOSE_ALL"  # Monetize Hedges


@dataclass
class HedgeInstruction:
    hedge_type: HedgeType
    allocation_pct: float  # Percent of portfolio equity to allocate/spend
    details: str


class HedgeManager:
    """
    Tail Hedging Manager.
    Protects portfolio from Extinction Events based on Volatility Regime.
    """

    def __init__(self, target_protection_pct: float = 0.20):
        self.target_protection_pct = target_protection_pct  # Protect against 20% crash

    def select_hedge(
        self, vol_state: VolatilityState, current_drawdown: float = 0.0
    ) -> HedgeInstruction:
        """
        Determines the optimal hedge structure based on Market State.
        """

        # 1. Panic / Normalization -> Monetize/Remove Hedges
        # In Panic, volatility is overpriced. We should be selling the expensive hedges we bought earlier.
        # In Normalization, risk has passed. Run naked or minimal.
        if vol_state in [VolatilityState.PANIC, VolatilityState.NORMALIZATION]:
            return HedgeInstruction(
                hedge_type=HedgeType.CLOSE_ALL,
                allocation_pct=0.0,
                details="Monetize/Close Hedges in High Vol",
            )

        # 2. Suppressed -> Buy Insurance (Cheap)
        # "Black Swan Unit": Buy 5-Delta Puts.
        # Cost is low, convexity is high.
        if vol_state == VolatilityState.SUPPRESSED:
            return HedgeInstruction(
                hedge_type=HedgeType.LONG_PUT,
                allocation_pct=0.01,  # Spend 1% of equity per month on insurance
                details="Buy 5-Delta Puts (Cheap Skew)",
            )

        # 3. Expanding -> Fund with Credit (Expensive)
        # "Ratio Backspread": Sell 1 Near Put / Buy 2 Far Puts.
        # OTM puts are expensive, so we sell them to fund the wings.
        if vol_state == VolatilityState.EXPANDING:
            return HedgeInstruction(
                hedge_type=HedgeType.RATIO_SPREAD,
                allocation_pct=0.005,  # Net cost should be near zero or small debit
                details="1x2 Ratio Put Spread (Funded Hedge)",
            )

        # Default
        return HedgeInstruction(HedgeType.NONE, 0.0, "No Hedge Needed")

    def calculate_required_notional(
        self, portfolio_value: float, index_level: float
    ) -> float:
        """
        Returns estimated Notional of Nifty Puts required to protect Portfolio.
        Example: If Port = 1Cr, Beta = 0.5 -> Need 50L Notional protection?
        Simplification: 1:1 Hedge for now.
        """
        return portfolio_value
