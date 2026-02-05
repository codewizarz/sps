import polars as pl
import numpy as np
from typing import Dict, List, Callable, Any, Optional
import itertools


class FragilityAnalyzer:
    """
    Fragility Analysis System.
    Tests robustness against Parameter Shifts, Execution Stress, and Capital Scaling.
    """

    def stress_execution(
        self,
        trades_df: pl.DataFrame,
        slippage_multipliers: List[float] = [1.0, 2.0, 5.0],
    ) -> pl.DataFrame:
        """
        Recalculates Strategy PnL under different slippage assumptions.
        Expects trades_df to have: 'pnl', 'slippage_cost' (or 'cost').
        If 'slippage_cost' missing, assumes a fixed bps cost or similar.
        """
        # Assume 'slippage_cost' exists in trades_df, representing the cost incurred at 1x
        if "slippage_cost" not in trades_df.columns:
            # Fallback: Estimate slippage as 0.05% of notional if not present?
            # Or just return empty. Let's assume input has it or we calculcate it.
            # For robustness, let's assume we are stressing the 'transaction_cost' column
            cost_col = (
                "transaction_cost"
                if "transaction_cost" in trades_df.columns
                else "cost"
            )
            if cost_col not in trades_df.columns:
                print("Warning: No cost column found to stress.")
                return pl.DataFrame()
        else:
            cost_col = "slippage_cost"

        results = []
        original_pnl = trades_df["pnl"].sum()

        for mult in slippage_multipliers:
            # New Cost = Old Cost * Multiplier
            # New PnL = Revenue - New Cost
            # Revenue = PnL + Old Cost

            # vectorized calc
            df_stress = trades_df.with_columns(
                (pl.col(cost_col) * mult).alias("stressed_cost")
            )

            df_stress = df_stress.with_columns(
                (pl.col("pnl") + pl.col(cost_col) - pl.col("stressed_cost")).alias(
                    "stressed_pnl"
                )
            )

            total_pnl = df_stress["stressed_pnl"].sum()

            results.append(
                {
                    "slippage_mult": mult,
                    "total_pnl": total_pnl,
                    "pnl_retention": total_pnl / original_pnl
                    if original_pnl != 0
                    else 0.0,
                }
            )

        return pl.DataFrame(results)

    def stress_parameters(
        self,
        backtest_func: Callable[[Dict[str, Any]], float],
        param_grid: Dict[str, List[Any]],
    ) -> pl.DataFrame:
        """
        Runs a grid sweep of parameters to measure 'Neighbor Stability'.
        backtest_func: Function taking a config dict and returning a metric (e.g. Sharpe).
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        results = []

        for combo in combinations:
            params = dict(zip(keys, combo))
            try:
                metric = backtest_func(params)
                row = params.copy()
                row["metric"] = metric
                results.append(row)
            except Exception as e:
                print(f"Backtest failed for {params}: {e}")

        return pl.DataFrame(results)

    def stress_capital(
        self, backtest_func: Callable[[float], float], aum_levels: List[float]
    ) -> pl.DataFrame:
        """
        Tests scalability by varying AUM.
        backtest_func: Function taking 'equity' and returning Sharpe/PnL.
        """
        results = []
        for aum in aum_levels:
            try:
                metric = backtest_func(aum)
                results.append({"aum": aum, "metric": metric})
            except Exception as e:
                print(f"Capital stress failed for {aum}: {e}")

        return pl.DataFrame(results)

    def compute_fragility_score(
        self, df_param_results: pl.DataFrame, metric_col: str = "metric"
    ) -> float:
        """
        Calculates Fragility Score based on parameter sensitivity.
        Defined as: Coefficient of Variation (StdDev / Mean) of the metric across the grid.
        Lower is better (more stable).
        """
        if df_param_results.height == 0:
            return 0.0

        std_dev = df_param_results[metric_col].std()
        mean_val = df_param_results[metric_col].mean()

        if mean_val == 0:
            return float("inf")

        # CV
        fragility = std_dev / abs(mean_val)
        return fragility
