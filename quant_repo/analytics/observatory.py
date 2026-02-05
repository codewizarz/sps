import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class MarketStatus:
    date: str
    regime: str
    term_structure_slope: float  # >0 Backwardation, <0 Contango
    skew_25d: float
    vvix: float
    vrp_spread: float
    warnings: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)


class VolatilityObservatory:
    """
    Volatility Observatory.
    Provides situational awareness of the options market structure.
    """

    def analyze_market(
        self, market_data: pl.DataFrame, current_date: Optional[str] = None
    ) -> MarketStatus:
        """
        Analyzes the latest market state from the provided DataFrame.
        Expected columns:
        - iv (ATM IV)
        - iv_put_25 (25-Delta Put IV)
        - vix_3m (3-month Vol / Back Month)
        - rv (Realized Vol)
        - date
        """

        # Filter for specific date if provided, else take latest
        if current_date:
            # Ensure we compare string to string (or date to date)
            # Casting col("date") to string is safest if input is YYYY-MM-DD string
            df_day = market_data.filter(
                pl.col("date") == pl.lit(current_date).str.strptime(pl.Date, "%Y-%m-%d")
            )
        else:
            df_day = market_data.tail(1)

        if df_day.height == 0:
            return MarketStatus("N/A", "UNKNOWN", 0.0, 0.0, 0.0, 0.0, ["No Data"])

        row = df_day.to_dict(as_series=False)

        # Extract scalar values safely
        iv_atm = row.get("iv", [0.0])[0]
        iv_put_25 = row.get("iv_put_25", [iv_atm * 1.05])[0]  # Fallback if missing
        iv_back = row.get("vix_3m", [iv_atm * 1.1])[0]  # Fallback to normal contango
        rv = row.get("rv", [iv_atm * 0.8])[0]

        # 1. Term Structure (Slope)
        # (Front - Back) / Front.
        # If Front > Back (Backwardation) -> Positive Slope (Panic).
        # If Front < Back (Contango) -> Negative Slope (Normal).
        term_slope = (iv_atm - iv_back) / iv_atm

        # 2. Skew (25d Put - ATM)
        skew = iv_put_25 - iv_atm

        # 3. VRP (IV - RV)
        vrp = iv_atm - rv

        # 4. VVIX (Vol of Vol) - Requires history
        # Ensure we only look at history relative to the specific date (No Lookahead)
        if current_date:
            df_history = market_data.filter(
                pl.col("date") <= pl.lit(current_date).str.strptime(pl.Date, "%Y-%m-%d")
            )
        else:
            df_history = market_data

        if df_history.height > 20:
            # Simple rolling std of log changes in IV over 20 days
            # This is a scalar proxy for VVIX
            # In a real observatory, we'd use index options VVIX index if available
            recent_iv = df_history.tail(20)["iv"].to_numpy()
            iv_changes = np.diff(np.log(recent_iv))
            vvix_proxy = np.std(iv_changes) * np.sqrt(252)
        else:
            vvix_proxy = 0.50  # Default highish

        # 5. Warnings & Anomalies
        warnings = []
        anomalies = []

        if term_slope > 0.05:
            warnings.append("Backwardation Alert (Panic Structure)")

        if skew < 0.0:
            anomalies.append("Inverted Skew (Call Bid > Put Bid)")

        if vrp < 0.0:
            warnings.append("Negative VRP (Selling is -EV)")

        if vvix_proxy > 1.0:
            warnings.append("Extreme VVIX (Instability)")

        # Regime Proxy
        regime = "NORMAL"
        if term_slope > 0.10:
            regime = "PANIC"
        elif vrp > 0.05 and term_slope < -0.05:
            regime = "HARVEST"

        return MarketStatus(
            date=str(row.get("date", ["N/A"])[0]),
            regime=regime,
            term_structure_slope=term_slope,
            skew_25d=skew,
            vvix=vvix_proxy,
            vrp_spread=vrp,
            warnings=warnings,
            anomalies=anomalies,
        )

    def detect_anomalies(
        self, market_data: pl.DataFrame, window: int = 252
    ) -> pl.DataFrame:
        """
        Scans history for statistical anomalies (Z-Score > 2.0).
        """
        # Calculate Rolling Mean/Std for Skew, IV, VRP
        return market_data.with_columns(
            [
                (
                    (pl.col("iv") - pl.col("iv").rolling_mean(window))
                    / pl.col("iv").rolling_std(window)
                ).alias("iv_zscore")
            ]
        )
