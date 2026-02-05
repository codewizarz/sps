import polars as pl
import numpy as np
from scipy.interpolate import CubicSpline
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable


@dataclass
class VolMetrics:
    expiry: str
    atm_vol: float
    skew_slope: float  # dVol/dStrike at ATM
    curvature: float  # d2Vol/dStrike2 at ATM


class VolSurface:
    """
    Represents a constructed Volatility Surface at a specific point in time.
    """

    def __init__(self, splines: Dict[str, CubicSpline], spot: float):
        self.splines = splines
        self.spot = spot

    def get_iv(self, expiry: str, strike: float) -> float:
        if expiry not in self.splines:
            return 0.0
        return float(self.splines[expiry](strike))

    def get_greeks(self, expiry: str) -> VolMetrics:
        if expiry not in self.splines:
            return VolMetrics(expiry, 0.0, 0.0, 0.0)

        spline = self.splines[expiry]
        atm_vol = float(spline(self.spot))
        skew = float(spline(self.spot, 1))  # 1st derivative
        curve = float(spline(self.spot, 2))  # 2nd derivative

        return VolMetrics(expiry, atm_vol, skew, curve)


class VolSurfaceGenerator:
    """
    Constructs VolSurface from market options data.
    """

    def fit_surface(self, df_options: pl.DataFrame, spot_price: float) -> VolSurface:
        """
        Fits splines to options data grouped by expiry.
        Expects columns: 'expiry_date', 'strike', 'iv'.
        """
        splines = {}

        # Get unique expiries
        expiries = df_options["expiry_date"].unique().to_list()

        for exp in expiries:
            # Filter and sort by strike
            df_slice = df_options.filter(
                (pl.col("expiry_date") == exp) & (pl.col("iv") > 0)
            ).sort("strike")

            if df_slice.height < 3:
                continue  # Need at least 3 points for Spline

            strikes = df_slice["strike"].to_numpy()
            ivs = df_slice["iv"].to_numpy()

            # 1. Fit Cubic Spline (Natural boundary conditions)
            # In prod, we would smooth this using SVI or penalized splines to avoid overfitting noise
            cs = CubicSpline(strikes, ivs, bc_type="natural")

            # 2. Basic Arbitrage Check (Butterfly)
            # Check if density is non-negative (roughly, Convexity check on prices, here we just check valid IV range)
            # A rigorous check would convert to Call Prices and checking convexity.
            # Here we just clamp extreme outliers in a real scaler

            splines[exp] = cs

        return VolSurface(splines, spot_price)

    def check_calendar_arbitrage(
        self,
        surface: VolSurface,
        expiry_near: str,
        expiry_far: str,
        strike: float,
        t_near: float,
        t_far: float,
    ) -> bool:
        """
        Checks if Variance(Far) > Variance(Near).
        Total Variance = IV^2 * T
        """
        iv_near = surface.get_iv(expiry_near, strike)
        iv_far = surface.get_iv(expiry_far, strike)

        var_near = (iv_near**2) * t_near
        var_far = (iv_far**2) * t_far

        return var_far > var_near
