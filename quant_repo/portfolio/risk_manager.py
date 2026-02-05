from typing import List, Protocol
from quant_repo.portfolio.risk import TradeStructure
from quant_repo.portfolio.kill_switch import KillSwitch, AccountState
from quant_repo.portfolio.sizing import VolTargetSizer


class PortfolioRiskManager:
    """
    Central gatekeeper for trade approval and sizing.
    """

    def __init__(
        self,
        kill_switch: KillSwitch,
        sizer: VolTargetSizer,
        max_net_vega_pct: float = 0.01,  # 1% of NAV
        max_margin_pct: float = 0.50,
    ):  # 50% Margin Util
        self.kill_switch = kill_switch
        self.sizer = sizer
        self.max_net_vega_pct = max_net_vega_pct
        self.max_margin_pct = max_margin_pct

    def check_and_size(
        self,
        structure: TradeStructure,
        state: AccountState,
        structure_unit_vega: float,
        structure_unit_margin: float,
    ) -> int:
        """
        Determines the safe quantity for a trade. Returns 0 if rejected.
        """
        # 1. Kill Switch
        if self.kill_switch.check(state):
            print("[RiskManager] Check Failed: Kill Switch Active")
            return 0

        # 2. Initial Sizing (Vol Target)
        qty = self.sizer.calculate_size(state.equity, structure_unit_vega)
        if qty == 0:
            return 0

        # 3. Aggregation Checks (Pro-Forma)
        # Check Max Vega
        proposed_trade_vega = qty * structure_unit_vega
        # Using abs() logic depends on if we cap Directional or Magnitude.
        # Usually we cap Net Vega (Directional Risk).

        new_net_vega = state.net_vega + proposed_trade_vega
        vega_limit = state.equity * self.max_net_vega_pct

        if abs(new_net_vega) > vega_limit:
            # Resize
            # How much room do we have?
            # room = limit - current (if same sign) or limit + current (if reducing risk)
            # Simplification: Reject if pushes over limit, or simple ratio reduction?
            # Let's simple reduce:
            # We need |current + q*unit| <= limit
            # This is complex to solve generically for "signed" reduction.
            # Conservative approach: Max additional Vega magnitude

            print(
                f"[RiskManager] Vega Limit Hit: New {new_net_vega:.2f} > Limit {vega_limit:.2f}"
            )
            # Try to resize to fit exactly?
            # Or just reject for now to be safe.
            return 0

        # 4. Check Margin
        proposed_margin = qty * structure_unit_margin
        new_margin = state.margin_used + proposed_margin
        margin_limit = state.equity * self.max_margin_pct

        if new_margin > margin_limit:
            # Resize
            # q * unit <= limit - used
            available = margin_limit - state.margin_used
            if available <= 0:
                return 0
            max_qty_margin = int(available / structure_unit_margin)
            qty = min(qty, max_qty_margin)
            print(f"[RiskManager] Resized for Margin: {qty}")

        return qty
