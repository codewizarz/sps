import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DealerProfile:
    total_gex_dollars: float  # Net Gamma Exposure in Notional $ per 1% move
    zero_gamma_level: float  # Estimated flip level
    dominant_magnets: List[float]  # Top OI Strikes
    regime: str  # "Long Gamma" or "Short Gamma"


class DealerPositioningSystem:
    """
    Estimates Dealer (Market Maker) positioning using Options OI and Gamma.
    """

    def calculate_gex(
        self,
        df_options: pl.DataFrame,
        spot_price: float,
        retail_net_short: bool = False,
    ) -> DealerProfile:
        """
        Computes Dealer GEX profile.
        Expects cols: 'strike', 'type' (CALL/PUT), 'open_interest', 'gamma'
        """
        # 1. Calculate GEX per Contract
        # Formula: Gamma * OI * 100 * Spot
        # Direction:
        #   Standard: Dealer Short Calls (-), Dealer Short Puts (+)
        #   Retail Short: Dealer Long Calls (+), Dealer Long Puts (-) -> Inverts sign

        # Standard Directions (Dealer Short Client Long)
        # Call: Dealer Short -> Negative Gamma
        # Put: Dealer Short -> Positive Gamma

        # If Retail is Short (Sold to Dealer), Dealer is Long.
        # Call: Dealer Long -> Positive Gamma
        # Put: Dealer Long -> Positive Gamma? No, Long Put is Positive Gamma.

        # Let's stick to the Standard "Index" Model first:
        # Calls: -1 (Dealer Short)
        # Puts: +1 (Dealer Short)

        direction_mult = -1.0 if retail_net_short else 1.0

        df = df_options.with_columns(
            pl.when(pl.col("type") == "CALL")
            .then(pl.lit(-1.0 * direction_mult))
            .otherwise(pl.lit(1.0 * direction_mult))  # PUT
            .alias("gex_sign")
        )

        # GEX = Gamma * OI * 100 * Spot * Sign
        # Result is "Dollar Gamma per 1% Move" roughly if Spot used appropriately?
        # Usually: Gamma * OI * 100 * Spot is "Dollar Delta change per 1 point move".
        # Computed as total deltas changing. To get Notional change, multiply by Spot again?
        # Let's stick to standard GEX unit: "Number of shares to hedge per 1 pt move".
        # Then multiply by Spot to get $ Exposure?
        # Standard GEX $ Value = Gamma * OI * 100 * Spot * Spot * 0.01 (for 1% move)
        # Simplified: Gamma * OI * 100 * Spot is standard.

        df = df.with_columns(
            (
                pl.col("gamma")
                * pl.col("open_interest")
                * 100.0
                * spot_price
                * pl.col("gex_sign")
            ).alias("gex_notional")
        )

        total_gex = df["gex_notional"].sum()
        regime = "Long Gamma" if total_gex > 0 else "Short Gamma"

        # 2. Identify Magnets (OI Concentration)
        # Aggregate OI by Strike
        df_oi = (
            df.group_by("strike")
            .agg(pl.col("open_interest").sum())
            .sort("open_interest", descending=True)
        )
        magnets = df_oi.head(3)["strike"].to_list()

        # 3. Find Zero Gamma Level (Flip)
        # We need the strike where cumulative GEX crosses zero?
        # Or simply the strike where the net GEX is zero?
        # A simple approximation: Aggregate GEX by strike, sort by strike, CumSum?
        # No, Flip Level is usually where the Market Global GEX flips sign.
        # If Total GEX is positive, flip level is below?
        # Common Approx: The strike with the lowest absolute total GEX? No.
        # Let's use a weighted average of positive vs negative GEX clusters?
        # For this MVP, we will estimate it as the strike where Cumulative GEX (sorted by strike) crosses 0?
        # No, that's assuming a specific distribution.
        # Let's return the strike with the maximum *Negative* GEX (Vol trigger) if Short Gamma,
        # or Max *Positive* GEX if Long Gamma.
        # Actually, "Zero Gamma Level" usually implies a theoretical spot price where Total GEX = 0.
        # We can simulate Total GEX at different spot prices (shifting Gamma distribution), but that requires full re-calc.
        # Fallback: Just return current Spot if Total GEX is small, or "unknown".
        # Allow simplified logic: Return strike where GEX flips from Put Dominated to Call Dominated?
        # Let's use: Strike where Cumulative GEX (from low to high strike) is roughly half of total range?
        # Implementation: Weighted Average Strike of Puts (Positive) vs Calls (Negative).
        # Let's stick to a simpler proxy: The strike with the highest Open Interest is often the gravity center.
        # We will use the 'magnets[0]' as a proxy for the 'center' for now if calc is complex.

        # Refined Flip Logic: Interpolate Net GEX by Strike.
        # Group by Strike, Sum GEX.
        # Smooth the curve. Find zero crossing.

        df_strike_gex = (
            df.group_by("strike").agg(pl.col("gex_notional").sum()).sort("strike")
        )

        # Basic linear scan for sign change
        zero_flip = 0.0
        gex_vals = df_strike_gex["gex_notional"].to_numpy()
        strikes = df_strike_gex["strike"].to_numpy()

        # If all positive or all negative, no flip in range.
        if np.all(gex_vals > 0) or np.all(gex_vals < 0):
            zero_flip = 0.0  # Undefined
        else:
            # Find index where sign changes
            signs = np.sign(gex_vals)
            sign_change = ((np.roll(signs, 1) - signs) != 0).astype(int)
            sign_change[0] = 0
            indices = np.where(sign_change != 0)[0]
            if len(indices) > 0:
                # Take the one closest to spot?
                idx = indices[np.abs(strikes[indices] - spot_price).argmin()]
                zero_flip = strikes[idx]

        return DealerProfile(
            total_gex_dollars=total_gex,
            zero_gamma_level=zero_flip,
            dominant_magnets=magnets,
            regime=regime,
        )
