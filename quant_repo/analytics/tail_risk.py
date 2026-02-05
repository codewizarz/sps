import polars as pl
import numpy as np
from typing import Dict, List, Optional


class TailRiskAnalyzer:
    """
    Tail Risk Detector.
    Identifies 'Extinction Events' for option sellers.
    """

    def detect_shocks(self, df: pl.DataFrame, rolling_window: int = 20) -> pl.DataFrame:
        """
        Scans for Vol Explosions, Gaps, Skew Shocks, and Liquidity Freezes.
        """
        # 1. Volatility Explosion ("The Vix Spike")
        # IV > 50% jump from yesterday
        df = df.with_columns(
            (pl.col("iv") / pl.col("iv").shift(1) - 1.0).alias("iv_chg_pct")
        )

        df = df.with_columns((pl.col("iv_chg_pct") > 0.50).alias("is_vol_explosion"))

        # 2. Gap Move ("The Overnight Risk")
        # Gap > 3 * Rolling Std of Daily Returns (Close-to-Close or Close-to-Open check)
        # Usually Gap is Open_t - Close_{t-1}
        # Rolling Std is based on Close-to-Close returns typically

        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_ret")
        )

        df = df.with_columns(
            pl.col("log_ret").rolling_std(rolling_window).alias("sigma")
        )

        df = df.with_columns(
            (
                (pl.col("open") - pl.col("close").shift(1)) / pl.col("close").shift(1)
            ).alias("gap_pct")
        )

        # Gap > 3 Sigma
        df = df.with_columns(
            (pl.col("gap_pct").abs() > (3.0 * pl.col("sigma")))
            .fill_null(False)
            .alias("is_gap_shock")
        )

        # 3. Skew Shock ("The Put Bid")
        # If 'skew' column exists (Put IV - ATM IV)
        if "skew" in df.columns:
            df = df.with_columns(
                pl.col("skew").rolling_mean(rolling_window).alias("avg_skew")
            )
            df = df.with_columns(
                (pl.col("skew") > (1.5 * pl.col("avg_skew")))
                .fill_null(False)
                .alias("is_skew_shock")
            )
        else:
            df = df.with_columns(pl.lit(False).alias("is_skew_shock"))

        # 4. Liquidity Collapse ("The Wide Spread")
        # If 'spread' column exists or bid/ask
        has_spread = "spread" in df.columns
        has_bid_ask = "ask" in df.columns and "bid" in df.columns

        if has_spread:
            pass  # Use existing
        elif has_bid_ask:
            df = df.with_columns((pl.col("ask") - pl.col("bid")).alias("spread"))
            has_spread = True

        if has_spread:
            df = df.with_columns(
                pl.col("spread").rolling_mean(rolling_window).alias("avg_spread")
            )
            # Spread > 3x Normal
            df = df.with_columns(
                (pl.col("spread") > (3.0 * pl.col("avg_spread")))
                .fill_null(False)
                .alias("is_liquidity_freeze")
            )
        else:
            df = df.with_columns(pl.lit(False).alias("is_liquidity_freeze"))

        return df

    def analyze_shock_stats(self, df: pl.DataFrame) -> Dict:
        """
        Summary statistics of detected shocks.
        """
        stats = {
            "vol_explosions": df["is_vol_explosion"].sum(),
            "gap_shocks": df["is_gap_shock"].sum(),
            "skew_shocks": df["is_skew_shock"].sum(),
            "liquidity_freezes": df["is_liquidity_freeze"].sum(),
            "max_vol_spike": df["iv_chg_pct"].max(),
            "max_gap_sigma": (df["gap_pct"].abs() / df["sigma"]).max()
            if "sigma" in df.columns
            else 0.0,
        }
        return stats
