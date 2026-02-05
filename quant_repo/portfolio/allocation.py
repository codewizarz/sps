from dataclasses import dataclass
import numpy as np


@dataclass
class AllocatorConfig:
    target_vol: float = 0.15  # Annualized Target Volatility (e.g. 15%)
    kelly_fraction: float = 0.25  # Fraction of Full Kelly to use
    max_dd_limit: float = 0.20  # Max Drawdown Limit (e.g. 20%)
    use_convex_dd_control: bool = True


@dataclass
class StrategyMetrics:
    win_rate: float
    avg_win: float
    avg_loss: float


@dataclass
class PortfolioState:
    current_equity: float
    current_vol: float  # Annualized
    current_drawdown: float  # e.g. 0.05 for 5% DD


class CapitalAllocator:
    """
    Calculates size scalar based on Volatility, Kelly Edge, and Drawdown.
    Formula: Base * VolScaler * KellyScore * DrawdownScaler
    """

    def calculate_allocation_scaler(
        self, config: AllocatorConfig, metrics: StrategyMetrics, state: PortfolioState
    ) -> float:
        # 1. Volatility Scaler
        # If CurrentVol > TargetVol, reduce size.
        # Cap scaling up to 1.5x or 2.0x to avoid crazy leverage in low vol
        vol_scaler = config.target_vol / max(state.current_vol, 0.01)  # Avoid div/0
        vol_scaler = min(vol_scaler, 2.0)

        # 2. Fractional Kelly
        # Kelly = p - q/b where b = avg_win / avg_loss
        if metrics.avg_loss == 0:
            kelly_full = 0.0  # Should not happen unless no trades
        else:
            b = metrics.avg_win / abs(metrics.avg_loss)
            p = metrics.win_rate
            q = 1 - p
            if b <= 0:
                kelly_full = 0.0
            else:
                kelly_full = p - (q / b)

        # Clip negative kelly (don't trace) and keep positive
        kelly_factor = max(0, kelly_full) * config.kelly_fraction

        # Normalize Kelly Factor?
        # Typically Kelly gives % of Bankroll.
        # Here we treat it as a Scalar for the "Base Unit".
        # But commonly Kelly IS the leverage.
        # Let's assume we return a Leverage Factor.
        # If Kelly says 10% and we are targeting 15% Vol, how do they mix?
        # Strictly, Kelly handles the vol sizing implicitly (high var -> low kelly).
        # But we want to separate Regime (Vol) from Edge (Kelly).

        # Approach: Use VolTarget as the primary sizer, and Scale by Edge Quality (Normalized).
        # OR: Just multiply them.
        # Let's say VolScaler gives us "Neutral Risk Size".
        # Kelly gives us "Edge Conviction".
        # Let's maintain the specific logic:
        # PctCapital = VolScaler * KellyFraction * DrawdownScaler is mixing dimensions.

        # Let's Standardize:
        # Base Allocation = 1.0 (Unit Size).
        # Scaler = VolScaler * (Kelly / ExpectedKelly?)

        # Simpler approach per Design Doc:
        # Allocation Scalar to multiply against "Max Safe Size" or "Unit Size".
        # Let's assume the user has a sizing logic (e.g. Risk 1% per trade).
        # We return a multiplier 0.0 to 1.0 (or >1.0).

        # Specific Implementation strictly following design:
        # Output is a multiplier.
        # VolScaler adjusts for environment.
        # But Kelly is absolute %...

        # Let's treat "Kelly Fraction" as a Scalar for aggressiveness.
        # Check Design Doc: "Allocation = BaseSize * VolScaler * KellyFraction * DDScaler"
        # This implies we take a fraction of the BaseSize.
        # BUT KellyFraction usually refers to "0.3 * Full Kelly".
        # If Full Kelly is 2.0, result is 0.6.
        # If Full Kelly is 0.1, result is 0.03.
        # So "KellyFactor" acts as the Size.

        # Actually, let's assume BaseSize is determined elsewhere (e.g. Risk Manager).
        # We just return the Risk Scalar (0.0 to 1.0).
        # So VolScaler reduces it.
        # DrawdownScaler reduces it.
        # What about Kelly?
        # If Kelly is used, it should probably REPLACE fixed fractional risk strategies.
        # But here let's assume we use Kelly as a Quality Score Scalar?
        # E.g. If Kelly > 0.2, Multiplier = 1.0. If Kelly < 0.05, Multiplier = 0.0.

        # Let's stick to the simplest interpretation of the user request:
        # "fractional Kelly bounds".
        # We will separate "Kelly Sizing" from this calculation or return it as "Recommended Leverage".

        # Let's output "RecommendedLeverage".
        # RecLev = VolScaler * (KellyFull * KellyFraction) * DDScaler

        allocation = vol_scaler * (kelly_full * config.kelly_fraction)

        # 3. Drawdown Control
        # Scaler = (1 - DD/MaxDD)^2
        if config.use_convex_dd_control:
            dd_ratio = min(state.current_drawdown / config.max_dd_limit, 1.0)
            dd_scaler = (1.0 - dd_ratio) ** 2
            allocation *= dd_scaler

        return max(0.0, allocation)
