import polars as pl
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from quant_repo.analytics.dealer_proxy import DealerPositioningSystem, DealerProfile


class GammaRegime(Enum):
    LONG_GAMMA = "LONG_GAMMA"  # Dealers buying dips (Mean Reversion)
    SHORT_GAMMA = "SHORT_GAMMA"  # Dealers selling dips (Trend Following)
    TRANSITION = "TRANSITION"  # Uncertainty zone


@dataclass
class RegimeSignal:
    regime: GammaRegime
    confidence: float
    gex_normalized: float  # GEX relative to some historical baseline/threshold


class GammaRegimeClassifier:
    """
    Classifies market environment based on Dealer Gamma Positioning.
    """

    def __init__(self, dealer_system: DealerPositioningSystem):
        self.dealer_system = dealer_system
        # Configuration
        self.gex_threshold = 0.0  # Standard zero line
        self.transition_zone_pct = 0.005  # 0.5% around flip level assumes transition

    def classify(
        self,
        df_options: pl.DataFrame,
        spot_price: float,
        retail_net_short: bool = False,
    ) -> RegimeSignal:
        """
        Determines the current Gamma Regime.
        """
        # 1. Get Dealer Profile
        profile = self.dealer_system.calculate_gex(
            df_options, spot_price, retail_net_short
        )

        # 2. Determine base regime from Net GEX
        if profile.total_gex_dollars > self.gex_threshold:
            regime = GammaRegime.LONG_GAMMA
        else:
            regime = GammaRegime.SHORT_GAMMA

        # 3. Check for Transition Zone (Proximity to Flip)
        # If absolute difference between spot and flip is small on % basis?
        # Note: Flip level might be 0 if undefined.
        if profile.zero_gamma_level > 0:
            dist_pct = abs(spot_price - profile.zero_gamma_level) / spot_price
            if dist_pct < self.transition_zone_pct:
                regime = GammaRegime.TRANSITION

        # 4. Calculate Confidence/Normalized Strength
        # Rough normalization: Just returning the raw value scaled down for now?
        # Or just return raw dollars.
        # Let's return raw as is, but maybe a simple confidence score?
        # For now, 1.0. Real implementation would Normalize by AVG_DAILY_VOLUME or MARKET_CAP.

        return RegimeSignal(
            regime=regime, confidence=1.0, gex_normalized=profile.total_gex_dollars
        )
