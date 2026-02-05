import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple


class VRPAnalyzer:
    """
    VRP Analytics Engine.
    Quantifies the Volatility Risk Premium (IV - Future RV).
    """

    def calculate_vrp(
        self, df: pl.DataFrame, lookahead_window: int = 21, clean_outliers: bool = True
    ) -> pl.DataFrame:
        """
        Computes VRP = IV_t - RV_{t->t+window}

        Args:
            df: DataFrame with ['date', 'close', 'iv'] (and optional 'regime', 'dte')
            lookahead_window: Number of trading days for Realized Vol calculation (e.g. 21 for monthly)

        Returns:
            DataFrame with added columns: ['close_return', 'rv_future', 'vrp']
        """
        # 1. Calculate Log Returns
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_ret")
        )

        # 2. Calculate Future Realized Volatility
        # We want the volatility of the NEXT 'lookahead_window' days relative to row t.
        # Shift returns backwards so that at row t, we see returns of t+1...t+window
        # Actually simplest is: Rolling Std of returns, shifted BACK by window size.
        # But standard rolling_std uses PAST data.
        # So we calculate rolling_std on PAST, then shift the result BACKWARDS by `window` rows.
        # Wait, if we use rolling_std(21), it covers t-20 to t.
        # If we shift this by -21, row t gets rolling_std calculated at t+21 (covering t+1 to t+21). Correct.

        # Annualization Factor: sqrt(252)
        ann_factor = np.sqrt(252)

        # Calculate Rolling Std (Backward looking first)
        df = df.with_columns(
            (pl.col("log_ret").rolling_std(lookahead_window) * ann_factor).alias(
                "rv_rolling"
            )
        )

        # Shift to align with "Future"
        # We shift by negative window size.
        df = df.with_columns(
            pl.col("rv_rolling").shift(-lookahead_window).alias("rv_future")
        )

        # 3. Calculate VRP
        df = df.with_columns((pl.col("iv") - pl.col("rv_future")).alias("vrp"))

        # Filter nulls (last 21 days will have no future RV)
        df_clean = df.drop_nulls(subset=["vrp"])

        if clean_outliers:
            # Simple clip to remove bad data (e.g. VRP > 200% or < -200%)
            df_clean = df_clean.filter(pl.col("vrp").abs() < 2.0)

        return df_clean

    def analyze_distribution(self, df: pl.DataFrame) -> Dict[str, float]:
        """
        Returns core distribution stats: Mean, Median, WinRate, Tail Risk (P5).
        """
        vrp_col = df["vrp"]

        stats = {
            "count": len(df),
            "mean_vrp": vrp_col.mean(),
            "median_vrp": vrp_col.median(),
            "std_vrp": vrp_col.std(),
            "win_rate": (vrp_col > 0).mean(),  # Freq of positive premium
            "p05": vrp_col.quantile(0.05),  # Tail Risk (Worst case for sellers)
            "p95": vrp_col.quantile(0.95),  # Best case
            "inversion_freq": (vrp_col < 0).mean(),  # How often is premium negative?
        }
        return stats

    def analyze_by_segment(self, df: pl.DataFrame, segment_col: str) -> pl.DataFrame:
        """
        Group statistics by a segment (e.g., 'regime', 'bucket_dte').
        """
        if segment_col not in df.columns:
            print(f"[VRP] Segment column '{segment_col}' not found.")
            return pl.DataFrame()

        # Aggregation
        return (
            df.group_by(segment_col)
            .agg(
                [
                    pl.col("vrp").count().alias("count"),
                    pl.col("vrp").mean().alias("mean_vrp"),
                    (pl.col("vrp") > 0).mean().alias("win_rate"),
                    pl.col("vrp").quantile(0.05).alias("p05_risk"),
                    pl.col("vrp").quantile(0.50).alias("median_vrp"),
                ]
            )
            .sort("mean_vrp", descending=True)
        )
