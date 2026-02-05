from typing import List, Optional
from quant_repo.portfolio.risk import TradeStructure, TradeLeg, RiskException
from quant_repo.portfolio.selection import StrikeSelector, OptionChain


class StructureFactory:
    """
    Converts High-Level Strategy Intents into Concrete TradeStructures.
    """

    def __init__(self, selector: StrikeSelector):
        self.selector = selector

    def create_iron_condor(
        self,
        chain: OptionChain,
        short_delta: float = 0.30,
        wing_width_deltas: float = 0.15,
    ) -> TradeStructure:
        """
        Creates an Iron Condor:
        - Sell Strangle (Short Call @ 30 Delta, Short Put @ 30 Delta)
        - Buy Wings (Long Call @ 15 Delta, Long Put @ 15 Delta)
        """
        # Deltas
        # Short Put: -0.30, Long Put: -0.15 (Further OTM, lower delta mag)
        # Short Call: 0.30, Long Call: 0.15 (Further OTM)

        s_put = self.selector.select(chain, -short_delta, "PUT")
        l_put = self.selector.select(
            chain, -(short_delta - wing_width_deltas), "PUT"
        )  # e.g. -0.15

        s_call = self.selector.select(chain, short_delta, "CALL")
        l_call = self.selector.select(
            chain, short_delta - wing_width_deltas, "CALL"
        )  # e.g. 0.15

        if not all([s_put, l_put, s_call, l_call]):
            raise RiskException("Could not find all legs for Iron Condor")

        legs = [
            TradeLeg(s_put.instrument_id, -1, "PUT"),
            TradeLeg(l_put.instrument_id, 1, "PUT"),
            TradeLeg(s_call.instrument_id, -1, "CALL"),
            TradeLeg(l_call.instrument_id, 1, "CALL"),
        ]

        return TradeStructure(legs=legs, strategy_type="IRON_CONDOR")

    def create_vertical_spread(
        self, chain: OptionChain, kind: str, short_delta: float, width_deltas: float
    ) -> TradeStructure:
        """
        Credit Spread: Sell high delta, Buy low delta.
        """
        # Normalize deltas based on kind
        if kind == "PUT":
            # Puts have negative delta.
            # Short 30 Delta Put -> Target -0.30
            # Buy 15 Delta Put -> Target -0.15
            target_short = -abs(short_delta)
            target_long = -abs(short_delta - width_deltas)
        else:
            target_short = abs(short_delta)
            target_long = abs(short_delta - width_deltas)

        short_leg = self.selector.select(chain, target_short, kind)
        long_leg = self.selector.select(chain, target_long, kind)

        if not all([short_leg, long_leg]):
            raise RiskException(f"Could not find legs for Vertical {kind}")

        legs = [
            TradeLeg(short_leg.instrument_id, -1, kind),
            TradeLeg(long_leg.instrument_id, 1, kind),
        ]

        return TradeStructure(legs=legs, strategy_type="VERTICAL_SPREAD")
