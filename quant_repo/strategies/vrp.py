from dataclasses import dataclass, field
from typing import List, Optional, Dict
import polars as pl
import pandas as pd
import numpy as np

from quant_repo.features.regime import RegimeClassifier, MarketRegime
from quant_repo.portfolio.allocation import (
    CapitalAllocator,
    AllocatorConfig,
    PortfolioState,
    StrategyMetrics,
)


@dataclass
class TradeInstruction:
    structure_type: str  # "IRON_CONDOR", "PUT_RATIO", "JADE_LIZARD", "CASH"
    action: str  # "OPEN", "CLOSE"
    allocation_scalar: float
    description: str


class RobustVRPStrategy:
    """
    Regime-Adaptive Volatility Risk Premium Strategy (Professional Grade).
    Features:
    - Gamma Kill Switch (<21 DTE)
    - Skew-Adaptive Structure Selection (Ratio Spread vs IC)
    - Circuit Breaker (Max Drawdown Freeze)
    """

    def __init__(self):
        self.regime_classifier = RegimeClassifier()
        self.allocator = CapitalAllocator()
        self.allocator_config = AllocatorConfig(
            target_vol=0.15,
            kelly_fraction=0.30,  # Conservative
            max_dd_limit=0.10,  # Hard strategy kill level
            use_convex_dd_control=True,
        )

        # Long-term performance metrics (Calibrated)
        self.strategy_metrics = StrategyMetrics(
            win_rate=0.75,
            avg_win=500.0,
            avg_loss=-1000.0,
        )

    def generate_instruction(
        self, market_data: pl.DataFrame, portfolio_state: PortfolioState
    ) -> Optional[TradeInstruction]:
        """
        Analyzes market data and returns a trading instruction based on Regime and Structure Logic.
        """

        # 0. Circuit Breaker: Daily Drawdown Freeze
        # If we lost > 3% from peak recently, we pause entries.
        # Assuming portfolio_state has 'current_drawdown'
        if portfolio_state.current_drawdown > 0.03:
            return TradeInstruction(
                structure_type="CASH",
                action="FREEZE",
                allocation_scalar=0.0,
                description=f"CIRCUIT BREAKER: Drawdown {portfolio_state.current_drawdown:.1%}. Trading Paused.",
            )

        # 1. Detect Regime
        df_regime = self.regime_classifier.detect_regime(market_data)
        current_regime_str = df_regime.tail(1)["regime"].item()
        if current_regime_str == "UNKNOWN":
            return None
        regime = MarketRegime(current_regime_str)

        # 2. Capital Allocation
        size_scalar = self.allocator.calculate_allocation_scaler(
            self.allocator_config, self.strategy_metrics, portfolio_state
        )

        # 3. Skew Analysis (Simulated for this implementation)
        # In prod, this would read from `market_data` IV columns
        # signal: Skew Slope (25D Put IV - 25D Call IV)
        # Pseudo-logic based on mock data availability
        skew_slope = 0.05  # Default "Normal"
        if "skew_slope" in market_data.columns:
            skew_slope = market_data.tail(1)["skew_slope"].item()

        is_steep_skew = skew_slope > 0.08
        is_flat_skew = skew_slope < 0.03

        # 4. Strategy Logic Switch

        if regime == MarketRegime.CRISIS:
            # DEFENSE: Hard Close
            return TradeInstruction(
                structure_type="CASH",
                action="FLATTEN",
                allocation_scalar=0.0,
                description="CRISIS DETECTED - Negative VRP. Hard Exit.",
            )

        elif regime == MarketRegime.BULL_QUIET:
            # HARVEST
            # Logic: If Skew is Steep -> Put Ratio (Free Hedge). If Flat -> Iron Condor.
            if is_steep_skew:
                return TradeInstruction(
                    structure_type="PUT_RATIO_SPREAD_1x2",
                    action="OPEN",
                    allocation_scalar=size_scalar,
                    description=f"BULL QUIET (Steep Skew) - 1x2 Ratio Spread. Financing Tails.",
                )
            else:
                return TradeInstruction(
                    structure_type="IRON_CONDOR",
                    action="OPEN",
                    allocation_scalar=size_scalar,
                    description=f"BULL QUIET (Flat Skew) - Iron Condor. Harvesting Belly.",
                )

        elif regime == MarketRegime.BULL_VOLATILE:
            # TREND UP + VOL
            # Ratio Spread is ideal here too (long vega tails)
            return TradeInstruction(
                structure_type="PUT_RATIO_SPREAD_1x2",
                action="OPEN",
                allocation_scalar=size_scalar * 0.7,  # Reduced size
                description=f"BULL VOLATILE - Volatility Robust Structure (Ratio).",
            )

        elif regime == MarketRegime.BEAR_QUIET:
            # DRIFT DOWN
            # Call Credit Spreads (Jade Lizard if Skew Inverted)
            return TradeInstruction(
                structure_type="CALL_CREDIT_SPREAD",
                action="OPEN",
                allocation_scalar=size_scalar,
                description="BEAR QUIET - Fading rallies.",
            )

        elif regime == MarketRegime.BEAR_VOLATILE:
            # CRASH
            return TradeInstruction(
                structure_type="CASH",
                action="FLATTEN",
                allocation_scalar=0.0,
                description="BEAR VOLATILE - Gamma Trap Avoidance. Flat.",
            )

        elif regime == MarketRegime.SIDEWAYS:
            # CHOP
            return TradeInstruction(
                structure_type="IRON_CONDOR",
                action="OPEN",
                allocation_scalar=size_scalar,
                description="SIDEWAYS - Balanced Exposure.",
            )

        return None

    def check_gamma_risk(self, positions_dte: List[int]) -> bool:
        """
        GAMMA KILL SWITCH.
        Returns True if ANY position is inside the 'Gamma Kill Zone' (< 21 DTE).
        Usage: If True, strategy should issue CLOSE orders for those trades immediately.
        """
        for dte in positions_dte:
            if dte < 21:
                return True
        return False
