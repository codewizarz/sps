from abc import ABC, abstractmethod
import polars as pl
import numpy as np
from typing import Optional


class EdgeDefinition(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_pnl_proxies(self, df_history: pl.DataFrame) -> pl.DataFrame:
        """
        Generates a simplified PnL series for the edge proxy trade.
        Input df expected to have: timestamp, spot, iv_atm, rv, skew_metric, etc.
        Output: DataFrame with 'timestamp', 'pnl_proxy'
        """
        pass


class VRPEdge(EdgeDefinition):
    """
    Volatility Risk Premium Edge.
    Proxy: Short Daily Straddle returns.
    Approx Return ~ (IV - RV) * Vega?
    More accurate: Daily Alpha = 0.5 * (ImpliedVariance - RealizedVariance)
    """

    @property
    def name(self) -> str:
        return "VRP_Short_Straddle"

    def generate_pnl_proxies(self, df_history: pl.DataFrame) -> pl.DataFrame:
        # Expected columns: iv_atm, realized_vol (annualized)

        # Daily Variance Risk Premium
        # VRP = IV^2 - RV^2
        # Daily PnL impact ~ VRP / 252

        # Check required columns
        req = ["iv_atm", "realized_vol"]
        if not all(c in df_history.columns for c in req):
            print(f"[VRPEdge] Missing columns {req}")
            return pl.DataFrame()

        return df_history.with_columns(
            [
                (
                    (pl.col("iv_atm").pow(2) - pl.col("realized_vol").pow(2)) / 252.0
                ).alias("pnl_proxy")
            ]
        ).select(["timestamp", "pnl_proxy"])


class SkewEdge(EdgeDefinition):
    """
    Skew Risk Premium.
    Proxy: Risk Reversal (Short Put, Long Call).
    Profit if Skew flattens or if Put IV > Realized Downside Vol.
    Simple Metric: (PutIV - CallIV) - (RealizedDownside - RealizedUpside)?
    Simplification: If Put Skew is high, selling Puts should outperform.
    Proxy PnL = SkewMetric * MeanReversionFactor
    """

    @property
    def name(self) -> str:
        return "Skew_Risk_Reversal"

    def generate_pnl_proxies(self, df_history: pl.DataFrame) -> pl.DataFrame:
        # Expected: iv_skew (25d Put - 25d Call)
        if "iv_skew" not in df_history.columns:
            return pl.DataFrame()

        # If Skew is positive (P > C), we Short Put / Long Call?
        # Actually usually Skew RP means Puts are "expensive". So we SELL Puts.
        # But we need to hedge delta.
        # Let's say Proxy PnL is capturing the 'overpricing' of the Put.
        # Proxy = SkewValue * (if Spot didn't crash) - CrashCost
        # Simplified: Just return the Skew Measure as a 'Carry' metric
        # adjusted by realizing returns.
        # Simulation: constant Short Skew position.

        # PnL = Skew_Carry - Realized_Skew_Vol?
        # Let's use a dummy proxy: 0.1 * iv_skew + Noise - 0.5 * max(0, -ret)
        # This mocks that we collect skew premium, but lose on crashes.

        # Needed: spot returns
        if "spot_return" not in df_history.columns:
            # calc
            if "spot" in df_history.columns:
                df_history = df_history.with_columns(
                    (pl.col("spot").pct_change()).fill_null(0).alias("spot_return")
                )
            else:
                return pl.DataFrame()

        return df_history.with_columns(
            [
                (
                    (pl.col("iv_skew") / 252.0)  # Daily carry
                    - (
                        pl.col("spot_return").clip(upper_bound=0).abs()
                        * 5.0
                        * (pl.col("spot_return") < -0.02)
                    )  # Crash penalty
                ).alias("pnl_proxy")
            ]
        ).select(["timestamp", "pnl_proxy"])
