import numpy as np
from typing import Protocol


# Protocol for structure analysis
class StructureVegaCalculator(Protocol):
    def calculate_unit_vega(self, structure) -> float: ...


class VolTargetSizer:
    """
    Determines trade size based on Volatility Targeting.
    """

    def __init__(self, target_vol_pct_per_trade: float = 0.001):
        """
        target_vol_pct_per_trade: Percentage of NAV to risk in Vega terms per trade.
        e.g., 0.1% of NAV. If NAV=1M, TargetVega = 1000.
        If Structure Unit Vega = 50, Size = 20 lots.
        """
        self.target_vol_pct = target_vol_pct_per_trade

    def calculate_size(self, equity: float, unit_vega: float) -> int:
        if unit_vega == 0:
            # Vega-neutral trade? Fallback to fixed risk % of equity?
            # For now, if 0 vega, return 0 or fallback sizing (TODO)
            # Assuming pure volatility strategies here where vega is key.
            # If purely Gamma scalping (Vega neutral), this sizer needs Gamma targeting.
            return 1  # Fallback or Raise

        target_vega_exposure = equity * self.target_vol_pct

        # We use abs() because Short Vol (Negative Vega) also consumes risk budget magnitude
        qty = abs(target_vega_exposure / unit_vega)

        return int(np.floor(qty))
