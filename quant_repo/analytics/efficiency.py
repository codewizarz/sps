import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class EfficiencyReport:
    rom_annualized: float
    hedge_drag_bps: float
    convexity_score: float  # Correlation with Vol
    tail_risk_ratio: float  # CVaR / Capital


class CapitalEfficiencyAnalyzer:
    """
    Analyzes the 'physics' of capital usage: Margin efficiency, Drag, and Convexity.
    """

    def analyze(
        self,
        trades: pl.DataFrame,
        capital_log: pl.DataFrame,
        margin_log: pl.DataFrame,
        market_data: Optional[pl.DataFrame] = None,
    ) -> EfficiencyReport:
        """
        trades: [date, pnl, strategy_type, premium_paid]
        capital_log: [date, equity]
        margin_log: [date, margin_used]
        market_data: [date, iv_change] (Needed for convexity)
        """

        # 1. Return on Margin (ROM)
        # Annualized PnL / Mean Margin
        # Total PnL
        total_pnl = trades["pnl"].sum()

        # Time period in years
        if len(capital_log) > 1:
            start_date = capital_log["date"].min()
            end_date = capital_log["date"].max()
            days = (end_date - start_date).days
            if days < 1:
                days = 1
            years = days / 365.25
        else:
            years = 1 / 365.25  # Fallback

        avg_margin = margin_log["margin_used"].mean()
        if avg_margin == 0:
            avg_margin = 1.0  # Prevent div/0

        rom_abs = total_pnl / avg_margin
        rom_annualized = rom_abs / years

        # 2. Hedge Drag
        # Sum of premium paid for 'HEDGE' strategies / Avg Equity
        # Assuming trades has 'strategy_type' column
        if "strategy_type" in trades.columns and "premium_paid" in trades.columns:
            hedge_cost = trades.filter(pl.col("strategy_type") == "HEDGE")[
                "premium_paid"
            ].sum()
        else:
            hedge_cost = 0.0

        avg_equity = capital_log["equity"].mean()
        if avg_equity == 0:
            avg_equity = 1.0

        drag_abs = hedge_cost / avg_equity
        drag_annualized = drag_abs / years
        hedge_drag_bps = drag_annualized * 10000

        # 3. Convexity Score
        # Correlation of Daily PnL with Volatility Change
        convexity = 0.0
        if market_data is not None and "iv_change" in market_data.columns:
            # We need daily PnL series joined with market data
            daily_pnl = (
                trades.group_by("date")
                .agg(pl.col("pnl").sum().alias("daily_pnl"))
                .sort("date")
            )

            # Join with market data
            merged = daily_pnl.join(market_data, on="date", how="inner")

            if len(merged) > 10:
                # Calculate correlation
                corr = np.corrcoef(merged["daily_pnl"], merged["iv_change"])[0, 1]
                if not np.isnan(corr):
                    convexity = corr

        # 4. Tail Risk Ratio (CVaR / Capital)
        # Using daily PnL distribution
        daily_pnl_vals = (
            trades.group_by("date").agg(pl.col("pnl").sum())["pnl"].to_numpy()
        )

        if len(daily_pnl_vals) > 0:
            var_99 = np.percentile(
                daily_pnl_vals, 1
            )  # 1st percentile (negative number)
            # CVaR is mean of values <= VaR
            cvar_99 = daily_pnl_vals[daily_pnl_vals <= var_99].mean()

            # Ratio relative to current equity
            current_equity = capital_log["equity"][-1]
            if current_equity > 0:
                tail_risk_ratio = abs(cvar_99) / current_equity
            else:
                tail_risk_ratio = 1.0  # Blown up
        else:
            tail_risk_ratio = 0.0

        return EfficiencyReport(
            rom_annualized=float(rom_annualized),
            hedge_drag_bps=float(hedge_drag_bps),
            convexity_score=float(convexity),
            tail_risk_ratio=float(tail_risk_ratio),
        )
