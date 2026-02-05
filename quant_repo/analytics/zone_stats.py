import polars as pl
import numpy as np
from typing import Dict, List, Optional


class ZoneAnalyzer:
    """
    VRP Zone Analyzer.
    Identifies the 'Safest Zones' on the volatility surface.
    """

    def bucket_options(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Adds 'dte_bucket' and 'delta_bucket' columns to the dataframe.
        """
        # DTE Buckets
        # 0-2 (Expiry), 3-7 (Weekly), 8-21 (Monthly), 21-45 (Far)
        df = df.with_columns(
            pl.when(pl.col("dte") <= 2)
            .then(pl.lit("0-2D"))
            .when((pl.col("dte") > 2) & (pl.col("dte") <= 7))
            .then(pl.lit("3-7D"))
            .when((pl.col("dte") > 7) & (pl.col("dte") <= 21))
            .then(pl.lit("8-21D"))
            .when((pl.col("dte") > 21) & (pl.col("dte") <= 45))
            .then(pl.lit("21-45D"))
            .otherwise(pl.lit(">45D"))
            .alias("dte_bucket")
        )

        # Delta Buckets (Absolute Delta)
        # 50-40 (ATM), 40-25 (Near OTM), 25-10 (Wings), 10-5 (Tail)
        df = df.with_columns(pl.col("delta").abs().alias("abs_delta"))

        df = df.with_columns(
            pl.when(pl.col("abs_delta") >= 0.40)
            .then(pl.lit("ATM (50-40)"))
            .when((pl.col("abs_delta") < 0.40) & (pl.col("abs_delta") >= 0.25))
            .then(pl.lit("Near OTM (40-25)"))
            .when((pl.col("abs_delta") < 0.25) & (pl.col("abs_delta") >= 0.10))
            .then(pl.lit("Wings (25-10)"))
            .when((pl.col("abs_delta") < 0.10) & (pl.col("abs_delta") >= 0.05))
            .then(pl.lit("Tail (10-5)"))
            .otherwise(pl.lit("Deep OTM (<5)"))
            .alias("delta_bucket")
        )

        return df

    def compute_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Computes Yield, Safety, Cost, and Harvest Score per option row.
        """
        # 1. Yield = Premium / Strike (Pct Return on Notional, approx Margin proxy)
        # Using mid price for premium
        df = df.with_columns(
            (pl.col("mid_price") / pl.col("strike")).alias("yield_pct")
        )

        # 2. Risk Metrics
        # Safety = 1 / (Gamma * 100 * (1 + Tail_Risk))
        # Gamma is small number (e.g. 0.05), so multiply by 100 to scale.
        # Tail Risk proxy: If 'kurtosis' column exists, use it. Else 0.

        if "kurtosis" in df.columns:
            tail_factor = 1.0 + pl.col("kurtosis")
        else:
            tail_factor = 1.0

        # Avoid division by zero if gamma is 0 (unlikely but safe)
        gamma_scaled = pl.max_horizontal(pl.col("gamma").abs() * 100, 0.01)

        df = df.with_columns((1.0 / (gamma_scaled * tail_factor)).alias("safety_score"))

        # 3. Liquidity Cost
        # Cost = (Ask - Bid) / Mid
        df = df.with_columns(
            (
                (pl.col("ask_price") - pl.col("bid_price"))
                / (pl.col("mid_price") + 1e-6)
            ).alias("cost_pct")
        )

        # 4. Harvest Score
        # Score = Yield * Safety * (1 - Cost)
        # Normalized logic: High Yield is good, High Safety is good, Low Cost is good.

        df = df.with_columns(
            (
                pl.col("yield_pct")
                * pl.col("safety_score")
                * (1.0 - pl.col("cost_pct"))
            ).alias("harvest_score")
        )

        return df

    def generate_heatmap(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Aggregates metrics by Zone to create a heatmap.
        """
        df_bucketed = self.bucket_options(df)
        df_scored = self.compute_scores(df_bucketed)

        heatmap = (
            df_scored.group_by(["dte_bucket", "delta_bucket"])
            .agg(
                [
                    pl.col("harvest_score").mean().alias("mean_score"),
                    pl.col("yield_pct").mean().alias("mean_yield"),
                    pl.col("safety_score").mean().alias("mean_safety"),
                    pl.col("cost_pct").mean().alias("mean_cost"),
                    pl.col("mid_price").count().alias("count"),
                ]
            )
            .sort("mean_score", descending=True)
        )

        return heatmap
