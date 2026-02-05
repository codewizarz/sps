import polars as pl
import numpy as np


def compute_realized_vol(
    df: pl.DataFrame, price_col: str = "close", window: int = 20
) -> pl.DataFrame:
    """
    Computes Annualized Realized Volatility using Log Returns.

    Formula: sqrt(252) * std_dev(ln(P_t / P_{t-1}))
    """
    return df.with_columns(
        [(pl.col(price_col) / pl.col(price_col).shift(1)).log().alias("log_return")]
    ).with_columns(
        [
            (pl.col("log_return").rolling_std(window) * np.sqrt(252)).alias(
                "realized_vol"
            )
        ]
    )


def compute_zscores(df: pl.DataFrame, col_name: str, window: int = 20) -> pl.DataFrame:
    """
    Computes Rolling Z-Score for a given column.

    Z = (X - Mean) / StdDev
    """
    mean_col = pl.col(col_name).rolling_mean(window)
    std_col = pl.col(col_name).rolling_std(window)

    return df.with_columns(
        [((pl.col(col_name) - mean_col) / std_col).alias(f"{col_name}_zscore")]
    )


def compute_iv_rv_spread(df: pl.DataFrame, iv_col: str, rv_col: str) -> pl.DataFrame:
    """
    Computes the spread between Implied and Realized Volatility.
    Spread = IV - RV
    """
    return df.with_columns([(pl.col(iv_col) - pl.col(rv_col)).alias("vol_spread")])
