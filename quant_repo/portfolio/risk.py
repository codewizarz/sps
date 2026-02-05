from dataclasses import dataclass
from typing import List, Protocol
from enum import Enum


class RiskException(Exception):
    pass


@dataclass
class TradeLeg:
    instrument_id: str
    ratio: int  # +1 for Buy, -1 for Sell
    option_kind: str  # "CALL" or "PUT" (or "CE"/"PE")


@dataclass
class TradeStructure:
    legs: List[TradeLeg]
    strategy_type: str


class RiskFilter:
    """
    Enforces risk management rules on proposed trade structures.
    """

    def validate(self, structure: TradeStructure) -> bool:
        """
        Validates the structure against risk rules.
        Raises RiskException if invalid.
        """
        self._check_naked_tails(structure)
        return True

    def _check_naked_tails(self, structure: TradeStructure):
        """
        Rule: No Naked Short Calls/Puts without Defined Risk.
        Simplified check: Sum of ratios should not be negative for a given Right?
        No, Short Vertical is defined risk but net ratio is 0.

        Naked Short Call: Sell 1 Call. Net Ratio -1.
        Naked Short Strangle: Sell 1 Call, Sell 1 Put.

        This requires knowing the Strike to determine if it's a spread or naked.
        For now, let's enforce a simple rule: Net quantity per option type must be >= 0
        UNLESS explicitly ALLOWED (which we default to NO for this safety filter).

        Actually, Vertical Spreads (Buy 1 ATM, Sell 1 OTM) -> Net 0. Safe.
        Ratio Spreads (Buy 1 ATM, Sell 2 OTM) -> Net -1. Unsafe Tail.
        """

        calls = [leg for leg in structure.legs if leg.option_kind in ["CALL", "CE"]]
        puts = [leg for leg in structure.legs if leg.option_kind in ["PUT", "PE"]]

        net_calls = sum(leg.ratio for leg in calls)
        net_puts = sum(leg.ratio for leg in puts)

        if net_calls < 0:
            raise RiskException(f"Naked Call Exposure detected: Net {net_calls}")

        if net_puts < 0:
            raise RiskException(f"Naked Put Exposure detected: Net {net_puts}")
