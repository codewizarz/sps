import polars as pl
from enum import Enum, auto


class MarketRegime(Enum):
    BULL_QUIET = "BULL_QUIET"
    BULL_VOLATILE = "BULL_VOLATILE"
    BEAR_QUIET = "BEAR_QUIET"
    BEAR_VOLATILE = "BEAR_VOLATILE"
    SIDEWAYS = "SIDEWAYS"
    CRISIS = "CRISIS"
    UNKNOWN = "UNKNOWN"


class RegimeClassifier:
    """
    Classifies market state based on Volatility, Trend, and Stress metrics.
    """

    def detect_regime(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Expects columns: 'close', 'iv', 'rv' (optional).
        Returns DataFrame with 'regime' column.
        """
        required = ["close", "iv"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[Regime] Missing columns: {missing}")
            return df.with_columns(pl.lit("UNKNOWN").alias("regime"))

        # 1. Trend (SMA 50 vs 200)
        # Note: Requires enough history. We'll use 'close'

        # Calculate Rolling metrics (Lazy or Eager). Assuming Eager for now.

        df_feats = df.with_columns(
            [
                pl.col("close").rolling_mean(50).alias("sma_50"),
                pl.col("close").rolling_mean(200).alias("sma_200"),
                # IV Rank (Rolling 252)
                (
                    (pl.col("iv") - pl.col("iv").rolling_min(252))
                    / (
                        pl.col("iv").rolling_max(252)
                        - pl.col("iv").rolling_min(252)
                        + 1e-6
                    )
                ).alias("iv_rank_pct"),
            ]
        )

        # If 'rv' exists, calc VRP
        if "rv" in df.columns:
            df_feats = df_feats.with_columns((pl.col("iv") - pl.col("rv")).alias("vrp"))
        else:
            df_feats = df_feats.with_columns(pl.lit(0.0).alias("vrp"))

        # 2. Logic Mapping
        # Use pl.when().then() chain

        # Conditions
        is_crisis = (
            pl.col("vrp") < -0.05
        )  # Negative VRP > 5% points (e.g. IV=20, RV=26)

        is_bull = (pl.col("close") > pl.col("sma_50")) & (
            pl.col("sma_50") > pl.col("sma_200")
        )
        is_bear = (pl.col("close") < pl.col("sma_50")) & (
            pl.col("sma_50") < pl.col("sma_200")
        )

        is_high_vol = pl.col("iv_rank_pct") > 0.80
        is_low_vol = pl.col("iv_rank_pct") <= 0.80  # Normal/Low

        # Prioritize CRISIS first
        regime_expr = (
            pl.when(is_crisis)
            .then(pl.lit(MarketRegime.CRISIS.value))
            # Bull Cases
            .when(is_bull & is_low_vol)
            .then(pl.lit(MarketRegime.BULL_QUIET.value))
            .when(is_bull & is_high_vol)
            .then(pl.lit(MarketRegime.BULL_VOLATILE.value))
            # Bear Cases
            .when(is_bear & is_low_vol)
            .then(pl.lit(MarketRegime.BEAR_QUIET.value))
            .when(is_bear & is_high_vol)
            .then(pl.lit(MarketRegime.BEAR_VOLATILE.value))
            # Fallback
            .otherwise(pl.lit(MarketRegime.SIDEWAYS.value))
        )

        return df_feats.with_columns(regime_expr.alias("regime"))
