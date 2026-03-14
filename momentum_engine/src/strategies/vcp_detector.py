import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def detect_vcp(df: pd.DataFrame) -> dict:
    """
    Detects Volatility Contraction Pattern (VCP) in a given price dataframe.

    The logic looks for:
    1. Peaks and troughs in the last 20-60 periods.
    2. Three successive pullbacks (P1, P2, P3).
    3. Conditions:
       - P1 > 15%
       - P2 < P1 * 0.6
       - P3 < 5%
       - Volume at the last trough < 60% of its 20-day average.

    Args:
        df: Pandas DataFrame with 'High', 'Low', 'Close', 'Volume' columns.
            Expected to be sorted by date ascending.

    Returns:
        A dictionary with signal, quality score, entry price, and stop loss.
    """
    result = {
        "signal": "NO",
        "quality_score": 0,
        "entry_price": None,
        "stop_loss": None,
    }

    if df is None or len(df) < 60:
        logger.debug("Not enough data to compute VCP.")
        return result

    # We need recent peaks and troughs.
    # For a robust VCP, we look at local highs/lows over a rolling window.
    # To simplify, we find the highest high in the last 60 days as the start of the pattern.
    recent_data = df.tail(60).copy()

    # Calculate 20-day moving average volume
    recent_data["Vol_20MA"] = recent_data["Volume"].rolling(window=20).mean()

    # Identify the major peak (start of P1)
    peak_idx = recent_data["High"].idxmax()
    peak_price = recent_data.loc[peak_idx, "High"]

    # Data after the major peak
    post_peak_data = recent_data.loc[peak_idx:]
    if len(post_peak_data) < 10:
        # P1, P2, P3 need time to develop
        logger.debug("Not enough time after major peak for VCP to develop.")
        return result

    # Sequential peak/trough finding
    # 1. Peak 1 is max high in recent data
    p1_idx = recent_data["High"].idxmax()
    peak1_price = recent_data.loc[p1_idx, "High"]

    # Let's find Peak 3 in the last 15 days
    tail_data = recent_data.tail(15)
    p3_idx = tail_data["High"].idxmax()

    # Make sure Peak 3 happens after Peak 1 with enough space
    if p3_idx <= p1_idx:
        logger.debug("Peak 3 <= Peak 1")
        return result

    # Peak 2 must be between Peak 1 and Peak 3 (excluding them)
    middle_data = recent_data.loc[p1_idx:p3_idx].iloc[1:-1]
    if len(middle_data) < 3:
        logger.debug("Not enough data between Peak 1 and Peak 3")
        return result

    p2_idx = middle_data["High"].idxmax()
    peak2_price = recent_data.loc[p2_idx, "High"]
    peak3_price = recent_data.loc[p3_idx, "High"]

    # 2. Troughs are the lowest points between the peaks (and after peak 3)
    # Trough 1
    t1_data = recent_data.loc[p1_idx:p2_idx]
    if len(t1_data) == 0:
        return result
    trough1_idx = t1_data["Low"].idxmin()
    trough1_price = t1_data.loc[trough1_idx, "Low"]

    # Trough 2
    t2_data = recent_data.loc[p2_idx:p3_idx]
    if len(t2_data) == 0:
        return result
    trough2_idx = t2_data["Low"].idxmin()
    trough2_price = t2_data.loc[trough2_idx, "Low"]

    # Trough 3
    t3_data = recent_data.loc[p3_idx:]
    if len(t3_data) < 2:
        return result
    # Only consider the window after peak 3
    trough3_idx = t3_data.iloc[1:]["Low"].idxmin()
    trough3_price = t3_data.loc[trough3_idx, "Low"]

    # 3. Calculate Pullback Percentages
    # p1_pct is the drop from peak 1 to trough 1
    p1_pct = (peak1_price - trough1_price) / peak1_price * 100
    # p2_pct is the drop from peak 2 to trough 2
    p2_pct = (peak2_price - trough2_price) / peak2_price * 100
    # p3_pct is the drop from peak 3 to trough 3
    p3_pct = (peak3_price - trough3_price) / peak3_price * 100

    # Volume condition: Volume at last trough < 60% of 20MA
    last_trough_vol = t3_data.loc[trough3_idx, "Volume"]
    last_trough_vol_ma = t3_data.loc[trough3_idx, "Vol_20MA"]

    vol_dry_up = False
    if pd.notna(last_trough_vol_ma) and last_trough_vol_ma > 0:
        if last_trough_vol < (0.6 * last_trough_vol_ma):
            vol_dry_up = True

    logger.debug(
        f"P1: {p1_pct:.2f}%, P2: {p2_pct:.2f}%, P3: {p3_pct:.2f}%, Vol Dry: {vol_dry_up}"
    )

    # Check VCP rules
    if p1_pct > 15 and p2_pct < (p1_pct * 0.6) and p3_pct < 5 and vol_dry_up:
        # Calculate quality out of 100 based on how tight P3 is and volume dry up
        tightness = max(0, 5 - p3_pct) / 5 * 50  # Up to 50 pts for tight P3
        vol_score = (
            max(0, 0.6 - (last_trough_vol / last_trough_vol_ma)) / 0.6 * 50
        )  # Up to 50 pts for very dry volume

        result["signal"] = "VCP"
        result["quality_score"] = round(tightness + vol_score)
        result["entry_price"] = df["Close"].iloc[
            -1
        ]  # Entry is typically break of last peak/current price
        result["stop_loss"] = (
            trough3_price  # Stop loss just below the final tight contraction
        )

    return result
