import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GapReport:
    suspicious_trades: pl.DataFrame
    avg_slippage_error: float
    optimism_score: float  # Positive = Backtest was too optimistic
    volume_violations: int
    spread_violations: int


class RealityDetector:
    """
    Audits backtest execution against market reality.
    Detects unrealistic fills, volume limit breaches, and spread optimism.
    """

    def __init__(self, expected_slippage_ticks: float = 1.0):
        self.expected_slippage_ticks = expected_slippage_ticks

    def check_execution(
        self,
        trade_log: pl.DataFrame,
        market_data: pl.DataFrame,
        tick_size: float = 0.05,
    ) -> GapReport:
        """
        trade_log: [date, symbol, exec_price, quantity, action, assumed_spread (opt)]
        market_data: [date, symbol, bid, ask, volume, high, low]
        """

        # Join on date and symbol
        # Assuming daily resolution for this implementation,
        # but could be timestamp-based for higher fidelity.

        joined = trade_log.join(market_data, on=["date", "symbol"], how="left")

        if len(joined) == 0:
            return GapReport(pl.DataFrame(), 0.0, 0.0, 0, 0)

        # 1. Price Reality Check (Did we trade outside High/Low range?)
        # A rigid backtest might fill at 'Close' when High was lower (impossible).
        price_violation = (pl.col("exec_price") > pl.col("high")) | (
            pl.col("exec_price") < pl.col("low")
        )

        # 2. Slippage Gap
        # Calculate theoretical mid price or use Bid/Ask depending on action
        # If BUY, real cost was ASK. If SELL, real revenue was BID.
        # Sim might have filled at Mid.

        # Define 'real_ref_price'
        real_ref_expr = (
            pl.when(pl.col("action") == "BUY")
            .then(pl.col("ask"))
            .otherwise(pl.col("bid"))
        )

        joined = joined.with_columns(real_ref_expr.alias("real_market_price"))

        # Slippage Gap = Simulated Fill - Real Ref Price (signed based on direction)
        # For BUY: If Sim Fill < Ask, we were optimistic (Gap positive/negative logic needs care).
        # Let's define Error = (Benefit in Sim) - (Benefit in Reality)
        # BUY: Sim Price 100, Real Ask 101 -> We saved 1. Optimistic. (Real Cost > Sim Cost) -> Sim PnL > Real PnL
        # SELL: Sim Price 100, Real Bid 99 -> We gained 1 more. Optimistic. (Sim Rev > Real Rev)

        # Optimism = (Action == BUY) * (Real Ask - Sim Price) + (Action == SELL) * (Sim Price - Real Bid)
        optimism_expr = (
            pl.when(pl.col("action") == "BUY")
            .then(pl.col("real_market_price") - pl.col("exec_price"))
            .otherwise(pl.col("exec_price") - pl.col("real_market_price"))
        )

        joined = joined.with_columns(optimism_expr.alias("optimism_per_unit"))

        # 3. Volume Check
        # Did we take too much liquidity?
        # Threshold: 10% of day's volume (aggressive for large days)
        vol_violation = (pl.col("quantity").abs() / pl.col("volume")) > 0.10

        # 4. Spread Check (if assumed_spread is present)
        spread_violation = pl.lit(False)
        if (
            "assumed_spread" in joined.columns
            and "bid" in joined.columns
            and "ask" in joined.columns
        ):
            real_spread = pl.col("ask") - pl.col("bid")
            # If we assumed a spread smaller than reality by 50%
            spread_violation = pl.col("assumed_spread") < (real_spread * 0.5)

        joined = joined.with_columns(
            [
                price_violation.alias("price_violation"),
                vol_violation.alias("vol_violation"),
                spread_violation.alias("spread_violation"),
            ]
        )

        # Aggregation
        suspicious = joined.filter(
            pl.col("price_violation")
            | pl.col("vol_violation")
            | pl.col("spread_violation")
            | (pl.col("optimism_per_unit") > (5 * tick_size))
        )

        avg_slippage_error = joined["optimism_per_unit"].mean()

        # Overall Optimism Score based on total PnL impact approximation
        # Sum(Optimism * Qty)
        total_optimistic_value = (
            joined["optimism_per_unit"] * joined["quantity"].abs()
        ).sum()
        total_value_traded = (joined["exec_price"] * joined["quantity"].abs()).sum()

        if total_value_traded > 0:
            optimism_score = total_optimistic_value / total_value_traded
        else:
            optimism_score = 0.0

        vol_violations_count = joined.filter(pl.col("vol_violation")).height
        spread_violations_count = joined.filter(pl.col("spread_violation")).height

        return GapReport(
            suspicious_trades=suspicious,
            avg_slippage_error=float(avg_slippage_error)
            if avg_slippage_error is not None
            else 0.0,
            optimism_score=float(
                optimism_score
            ),  # Percentage of volume that was 'free money'
            volume_violations=vol_violations_count,
            spread_violations=spread_violations_count,
        )
