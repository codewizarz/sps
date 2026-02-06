from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
from quant_repo.analytics.gamma_regime import GammaRegime


class KillStatus(Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"  # Temporary condition (e.g., waiting for liquidity)
    KILLED = "KILLED"  # Permanent stop (requires manual intervention)


@dataclass
class KillSwitchConfig:
    max_drawdown_multiplier: float = 1.5
    liquidity_spread_threshold: float = 0.01  # 1% spread is huge, likely pause
    slippage_tolerance_x: float = 3.0  # If slippage is 3x expected, pause
    strategy_type: str = "SHORT_VOL"  # Default to fragile strat type


class SmartKillSwitch:
    """
    Advanced circuit breaker monitoring behavioral (PnL) and environmental (Regime, Liquidity) triggers.
    """

    def __init__(self, config: KillSwitchConfig = KillSwitchConfig()):
        self.config = config
        self.status = KillStatus.ACTIVE
        self.reason = ""
        self.manual_override = False

    def reset(self):
        """Manually reset a KILLED switch."""
        self.status = KillStatus.ACTIVE
        self.reason = ""
        self.manual_override = False

    def check_status(
        self,
        pnl_stats: Dict[str, float],
        regime: GammaRegime,
        market_liquidity: Dict[str, float],
        execution_metrics: Dict[str, float],
    ) -> KillStatus:
        """
        Evaluates all triggers and returns current status.
        Status is sticky for KILLED (requires reset).
        Status is ephemeral for PAUSED (auto-resolves).
        """

        if self.status == KillStatus.KILLED:
            return KillStatus.KILLED

        # 1. Drawdown Check (Hard Kill)
        # Expects: 'current_drawdown_pct' (positive float, e.g. 0.15 for 15% DD)
        #          'expected_drawdown_pct' (e.g. 0.10)
        curr_dd = pnl_stats.get("current_drawdown_pct", 0.0)
        exp_dd = pnl_stats.get("expected_drawdown_pct", 0.10)

        limit = exp_dd * self.config.max_drawdown_multiplier

        if curr_dd > limit:
            self.status = KillStatus.KILLED
            self.reason = f"Drawdown {curr_dd:.1%} exceeds limit {limit:.1%} ({self.config.max_drawdown_multiplier}x Expected)"
            return self.status

        # 2. Regime Check (Pause)
        # If Short Vol strategy meets Short Gamma regime -> Pause
        # If Long Vol strategy meets Long Gamma regime -> Maybe suboptimal but not fatal?
        # Primarily protecting Short Vol from Trends.
        if self.config.strategy_type == "SHORT_VOL":
            if regime == GammaRegime.SHORT_GAMMA:
                # We return PAUSED, but do not set self.status to permanent KILLED
                self.reason = "Regime Mismatch: Short Vol in Short Gamma Environment"
                return KillStatus.PAUSED

        # 3. Liquidity Check (Pause)
        spread = market_liquidity.get("bid_ask_spread_pct", 0.0)
        if spread > self.config.liquidity_spread_threshold:
            self.reason = f"Liquidity Dried Up: Spread {spread:.2%} > {self.config.liquidity_spread_threshold:.2%}"
            return KillStatus.PAUSED

        # 4. Slippage Check (Pause)
        realized = execution_metrics.get("realized_slippage_pct", 0.0)
        modeled = execution_metrics.get("modeled_slippage_pct", 0.0001)
        if modeled > 0 and realized > (modeled * self.config.slippage_tolerance_x):
            self.reason = f"Execution Anomaly: Slippage {realized:.4%} > {self.config.slippage_tolerance_x}x Expected"
            return KillStatus.PAUSED

        # All Clear
        self.reason = ""
        return KillStatus.ACTIVE
