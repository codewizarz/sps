from dataclasses import dataclass
from typing import List, Optional, Protocol
import numpy as np


# Protocol for Option info to avoid tight coupling with specific Instrument classes here
@dataclass
class OptionInfo:
    instrument_id: str
    strike: float
    option_kind: str
    delta: float
    bid_ask_spread: float
    open_interest: int


class OptionChain:
    """
    Helper to query options by Delta/Strike.
    """

    def __init__(self, options: List[OptionInfo]):
        self.options = options
        # Sort by strike for easier lookup
        self.options.sort(key=lambda x: x.strike)

    def get_closest_delta(self, target_delta: float, kind: str) -> Optional[OptionInfo]:
        """Finds option with delta closest to target."""
        candidates = [o for o in self.options if o.option_kind == kind]
        if not candidates:
            return None

        # Closest by abs diff
        best = min(candidates, key=lambda x: abs(x.delta - target_delta))
        return best

    def get_closest_strike(
        self, target_strike: float, kind: str
    ) -> Optional[OptionInfo]:
        candidates = [o for o in self.options if o.option_kind == kind]
        if not candidates:
            return None
        best = min(candidates, key=lambda x: abs(x.strike - target_strike))
        return best


class StrikeSelector:
    """
    Selects specific options based on delta targets and liquidity constraints.
    """

    def __init__(self, min_oi: int = 100, max_spread_pct: float = 0.05):
        self.min_oi = min_oi
        self.max_spread_pct = max_spread_pct

    def select(
        self, chain: OptionChain, target_delta: float, kind: str
    ) -> Optional[OptionInfo]:
        """
        Selects an option matching the criteria.
        Returns None if no liquid option found.
        """
        # 1. Find best fit
        opt = chain.get_closest_delta(target_delta, kind)
        if not opt:
            return None

        # 2. Check Liquidity
        if opt.open_interest < self.min_oi:
            # Try finding neighbors? For now, strict fail.
            print(
                f"[StrikeSelector] Rejected {opt.instrument_id}: Low OI ({opt.open_interest})"
            )
            return None

        # Spread check (assuming price info available or passed, here using scalar spread)
        # if opt.bid_ask_spread > ...

        return opt
