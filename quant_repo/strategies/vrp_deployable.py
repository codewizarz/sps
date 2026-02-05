from typing import List, Dict, Optional
import polars as pl
from dataclasses import dataclass
from quant_repo.features.vol_state import VolatilityStateManager, VolatilityState
from quant_repo.analytics.zone_stats import ZoneAnalyzer
from quant_repo.risk.hedging import HedgeManager, HedgeInstruction, HedgeType


@dataclass
class TradeInstruction:
    order_type: str  # "BUY", "SELL"
    instrument_type: str  # "PUT", "CALL", "FUTURE"
    strike: float
    expiry_date: str  # YYYY-MM-DD
    qty: int
    rationale: str


class DeployableVRPStrategy:
    """
    Flagship VRP Strategy.
    Integrates State Detection, Zone Analysis, and Tail Hedging.
    """

    def __init__(self, equity: float):
        self.equity = equity
        self.vol_manager = VolatilityStateManager()
        self.zone_analyzer = ZoneAnalyzer()
        self.hedge_manager = HedgeManager()

    def generate_signals(
        self, market_data: pl.DataFrame, current_date: str
    ) -> List[TradeInstruction]:
        """
        Orchestrates the trading logic for a single day.

        Args:
            market_data: DataFrame with columns [date, close, iv, iv_rank, etc.] + Option Chain data
        """
        instructions = []

        # 1. Detect State & Exposure
        # Assuming market_data has the feature columns needed by detect_state
        # For simplicity in this shell, we assume detection was run or we run it on latest row
        # (In prod, we'd pass the full history df to detect_state)

        df_state = self.vol_manager.detect_state(market_data)
        current_state_val = df_state["vol_state"][-1]
        current_state = VolatilityState(
            current_state_val
        )  # Convert string to Enum if needed, assuming match

        exposure_scalar = self.vol_manager.get_exposure_scalar(current_state)

        print(
            f"[Strategy] Date: {current_date} | State: {current_state} | Scalar: {exposure_scalar}x"
        )

        # 2. Manage Hedges (Protective Shield)
        hedge_instr = self.hedge_manager.select_hedge(current_state)

        if (
            hedge_instr.hedge_type != HedgeType.NONE
            and hedge_instr.hedge_type != HedgeType.CLOSE_ALL
        ):
            instructions.append(
                TradeInstruction(
                    order_type="BUY",
                    instrument_type="HEDGE_STRUCTURE",  # Abstraction for specific leg generation
                    strike=0.0,  # Placeholder, would be calculated by delta
                    expiry_date="MONTHLY_FAR",
                    qty=1,  # simplified
                    rationale=f"HEDGE: {hedge_instr.details}",
                )
            )
        elif hedge_instr.hedge_type == HedgeType.CLOSE_ALL:
            instructions.append(
                TradeInstruction(
                    order_type="SELL",
                    instrument_type="HEDGE_STRUCTURE",
                    strike=0.0,
                    expiry_date="ALL",
                    qty=0,
                    rationale="MONETIZE HEDGES",
                )
            )

        # 3. Income Generation (if Exposure > 0)
        if exposure_scalar > 0:
            # Run Zone Analysis
            # We filter for options available today
            # df_options = market_data.filter(pl.col("type") == "OPTION") ...
            # For this mock, we assume zone_analyzer returns the 'Best Zone' descriptor

            # Simulated Zone Result
            best_dte = "3-7D"
            best_delta = "Near OTM (40-25)"

            # Creating the Short Instruction
            # "Selling Premium"
            base_qty = int((self.equity * 0.25) / 100000)  # Mock sizing 25%
            scaled_qty = int(base_qty * exposure_scalar)

            if scaled_qty > 0:
                instructions.append(
                    TradeInstruction(
                        order_type="SELL",
                        instrument_type="PUT",
                        strike=0.0,  # Would be valid strike
                        expiry_date=f"WEEKLY_{best_dte}",
                        qty=scaled_qty,
                        rationale=f"INCOME: Short {best_dte} {best_delta} Put (Scalar {exposure_scalar})",
                    )
                )

        return instructions
