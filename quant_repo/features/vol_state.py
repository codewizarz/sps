from enum import Enum, auto
import polars as pl
import numpy as np


class VolatilityState(Enum):
    SUPPRESSED = "SUPPRESSED"  # The Coil: Low IV, Low VVIX. Explosion risk.
    EXPANDING = "EXPANDING"  # The Breakout: Rising IV. Gamma risk.
    PANIC = "PANIC"  # The Spike: Extreme IV, Inversion. Insolvency risk.
    NORMALIZATION = "NORMALIZATION"  # The Crush: Failing IV. Golden opportunity.
    UNKNOWN = "UNKNOWN"


class VolatilityStateManager:
    """
    Classifies the volatility cycle into 4 discrete phases:
    Suppressed -> Expanding -> Panic -> Normalization
    """

    def detect_state(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Expects columns: 'iv' (Annualized Implied Vol).
        Optional: 'vix_3m' (for Term Structure), 'rv' (Realized).
        Returns DataFrame with 'vol_state' column.
        """
        required = ["iv"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return df.with_columns(
                pl.lit(VolatilityState.UNKNOWN.value).alias("vol_state")
            )

        # 1. Feature Engineering
        # IV Rank (Rolling 252 - 1 Year)
        df_feats = df.with_columns(
            [
                pl.col("iv").rolling_min(252).alias("iv_min_252"),
                pl.col("iv").rolling_max(252).alias("iv_max_252"),
            ]
        )

        # IV Slope (5-day linear regression approx via diff)
        # Using simple change for speed/robustness: IV - IV(5d ago)
        df_feats = df_feats.with_columns(
            [
                (
                    (pl.col("iv") - pl.col("iv_min_252"))
                    / (pl.col("iv_max_252") - pl.col("iv_min_252") + 1e-6)
                ).alias("iv_rank_pct"),
                (pl.col("iv") - pl.col("iv").shift(5)).alias("iv_change_5d"),
            ]
        )

        # Term Structure Ratio (if available, else assume 1.0)
        # Ratio > 1.0 means Inversion (Front > Back) -> Panic
        if "vix_3m" in df.columns:
            df_feats = df_feats.with_columns(
                (pl.col("iv") / pl.col("vix_3m")).alias("term_struct_ratio")
            )
        else:
            df_feats = df_feats.with_columns(pl.lit(1.0).alias("term_struct_ratio"))

        # 2. Logic Mapping via Polars Expressions

        is_suppressed = (pl.col("iv_rank_pct") < 0.20) & (
            pl.col("iv_change_5d") <= 0.001
        )
        is_expanding = (
            pl.col("iv_change_5d") > 0.001
        )  # threshold for meaningful expansion

        # Panic: High Rank + Inversion (or very high rank)
        is_panic = (pl.col("iv_rank_pct") > 0.90) | (pl.col("term_struct_ratio") > 1.1)

        # Normalization: Rank High but falling hard
        # Need to detect 'Was Panic'. For stateless vectorized ops, we look at:
        # Rank is Elevated (>50%) AND Slope is Sharply Negative
        is_normalization = (pl.col("iv_rank_pct") > 0.40) & (
            pl.col("iv_change_5d") < -0.01
        )  # Dropping > 1 vol point in week

        # Evaluation Order Priorities:
        # Panic > Normalization > Suppressed > Expanding (Default to Expanding if moving?)
        # Let's align with cycle logic.

        state_expr = (
            pl.when(is_panic)
            .then(pl.lit(VolatilityState.PANIC.value))
            .when(is_normalization)
            .then(pl.lit(VolatilityState.NORMALIZATION.value))
            .when(is_suppressed)
            .then(pl.lit(VolatilityState.SUPPRESSED.value))
            .when(is_expanding)
            .then(pl.lit(VolatilityState.EXPANDING.value))
            .otherwise(
                pl.lit(VolatilityState.EXPANDING.value)
            )  # Default state for middle ground or noise
        )

        return df_feats.with_columns(state_expr.alias("vol_state"))

    def get_exposure_scalar(self, state: VolatilityState) -> float:
        """
        Returns suggested exposure multiplier 0.0 - 2.0 based on Vol State.
        """
        if state == VolatilityState.SUPPRESSED:
            return 0.5  # Reduce size, risk of explosion
        elif state == VolatilityState.EXPANDING:
            return 0.0  # Freeze, don't sell into expansion
        elif state == VolatilityState.PANIC:
            return 0.0  # Kill, cash is king
        elif state == VolatilityState.NORMALIZATION:
            return 2.0  # Max Aggression, the Crush
        return 1.0
