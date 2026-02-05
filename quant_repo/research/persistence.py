import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class EdgeMetrics:
    total_sharpe: float
    stability_score: float  # Annualized Sharpe / Std(Annual Sharpe)
    decay_slope: float  # Linear regression slope of Annual Sharpes
    win_year_pct: float  # % of years positive


class PersistenceAnalyzer:
    """
    Analyzes PnL series for stability and decay.
    """

    def analyze(self, df_pnl: pl.DataFrame) -> EdgeMetrics:
        # df_pnl has 'timestamp', 'pnl_proxy'

        # 1. Annual grouping
        # Timestamps are likely ns.
        df_years = df_pnl.with_columns(
            [pl.from_epoch(pl.col("timestamp"), time_unit="ns").dt.year().alias("year")]
        )

        annual_stats = (
            df_years.group_by("year")
            .agg(
                [
                    pl.col("pnl_proxy").sum().alias("total_pnl"),
                    pl.col("pnl_proxy").mean().alias("mean_return"),
                    pl.col("pnl_proxy").std().alias("std_return"),
                    pl.col("pnl_proxy").count().alias("count"),
                ]
            )
            .sort("year")
        )

        # Calculate Annual Sharpes
        # sharpe = (mean * 252) / (std * sqrt(252)) = mean/std * sqrt(252)
        annual_stats = annual_stats.with_columns(
            [
                ((pl.col("mean_return") / pl.col("std_return")) * np.sqrt(252))
                .fill_nan(0.0)
                .alias("sharpe")
            ]
        )

        sharpes = annual_stats["sharpe"].to_list()

        if len(sharpes) == 0:
            return EdgeMetrics(0, 0, 0, 0)

        # 2. Metrics
        avg_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes) if len(sharpes) > 1 else 1.0
        stability = avg_sharpe / std_sharpe if std_sharpe > 0 else 0.0

        # Decay Slope (Linear Reg of Sharpe vs Year Index)
        if len(sharpes) > 1:
            x = np.arange(len(sharpes))
            y = np.array(sharpes)
            slope, _ = np.polyfit(x, y, 1)
        else:
            slope = 0.0

        win_years = len([s for s in sharpes if s > 0])
        win_pct = win_years / len(sharpes)

        return EdgeMetrics(
            total_sharpe=float(avg_sharpe),  # Avg annual sharpe
            stability_score=float(stability),
            decay_slope=float(slope),
            win_year_pct=float(win_pct),
        )
