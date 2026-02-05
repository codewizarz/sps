from dataclasses import dataclass
import numpy as np
from enum import Enum


class OrderSide(Enum):
    BUY = 1
    SELL = -1


@dataclass
class QuoteTick:
    # Minimal representation for simulation
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    mid_price: float
    daily_volume: int  # Needed for impact model


@dataclass
class SimulationResult:
    fill_price: float
    filled_qty: int
    slippage_cost: float
    impact_bps: float


class ExecutionSimulator:
    """
    Simulates execution realism:
    1. Crosses Spread (Always)
    2. Widens Spread based on Volatility (Vol Regime)
    3. Adds Market Impact based on Size vs Volume (Square Root Law)
    """

    def __init__(
        self,
        baseline_iv: float = 0.15,
        vol_spread_sensitivity: float = 1.0,  # Alpha
        impact_coefficient: float = 0.5,
    ):  # Beta
        self.baseline_iv = baseline_iv
        self.vol_sensitivity = vol_spread_sensitivity
        self.impact_coeff = impact_coefficient

    def simulate(
        self, tick: QuoteTick, side: OrderSide, size: int, current_iv: float
    ) -> SimulationResult:
        """
        Calculates theoretical execution price.
        """
        # 1. Volatility Adjustment
        # If IV is high, Market Makers widen quotes to protect against gamma/vega.
        # Widen factor = (Current IV / Baseline) ^ Sensitivity

        iv_ratio = current_iv / self.baseline_iv
        widen_factor = max(1.0, iv_ratio**self.vol_sensitivity)

        original_spread = tick.ask_price - tick.bid_price
        new_spread = original_spread * widen_factor

        # Adjust quotes centered on Mid
        adj_bid = tick.mid_price - (new_spread / 2)
        adj_ask = tick.mid_price + (new_spread / 2)

        # 2. Base Price (Crossing)
        # We fill at THE MARKET's Price.
        # Buy -> Fill at Ask. Sell -> Fill at Bid.
        base_price = adj_ask if side == OrderSide.BUY else adj_bid

        # Limit check: If price crossed mid, clamp? No, let spread widen.

        # 3. Market Impact (Square Root Law)
        # Impact ~ Vol * sqrt(Size / Volume)
        # We assume daily volume is available. If 0, use fallback.
        volume = max(1000, tick.daily_volume)  # Avoid div by zero

        participation = size / volume
        # Basic sqrt model: Sigma * sqrt(Part/10) ??
        # Let's use: Impact (bps) = Coeff * IV * sqrt(Participation)

        impact_bps = self.impact_coeff * current_iv * np.sqrt(participation)

        impact_cost = base_price * impact_bps

        if side == OrderSide.BUY:
            final_price = base_price + impact_cost
        else:
            final_price = base_price - impact_cost

        slippage = abs(
            final_price - (tick.ask_price if side == OrderSide.BUY else tick.bid_price)
        )

        return SimulationResult(
            fill_price=round(final_price, 2),
            filled_qty=size,  # Assuming full fill for now, could model partial
            slippage_cost=slippage * size,
            impact_bps=impact_bps * 10000,
        )
