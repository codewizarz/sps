from typing import Dict, List, Optional
from quant_repo.features.regime import MarketRegime


class StrategyOrchestrator:
    """
    Adapts portfolio weights based on the current market regime.
    """

    def __init__(self, regime_matrix: Dict[MarketRegime, Dict[str, float]]):
        """
        Args:
            regime_matrix: Mapping of Regime -> {StrategyType: Multiplier}
        """
        self.regime_matrix = regime_matrix

    def calculate_final_weights(
        self,
        base_weights: Dict[str, float],
        current_regime: MarketRegime,
        strategy_types: Dict[
            str, str
        ],  # Map StrategyName -> StrategyType (e.g. "VRP_Algo" -> "ShortVol")
    ) -> Dict[str, float]:
        """
        Applies regime-based multipliers to base weights.
        Remainder (if sum < 1.0) is implied CASH.
        """

        final_weights = {}

        # Get multipliers for current regime
        # If regime not in matrix, default to 1.0 (Neutral)
        multipliers = self.regime_matrix.get(current_regime, {})

        for strat_name, weight in base_weights.items():
            strat_type = strategy_types.get(strat_name, "Unknown")

            # Default multiplier 1.0 if not specified
            scalar = multipliers.get(strat_type, 1.0)

            # Apply scalar
            final_weights[strat_name] = weight * scalar

        # Add Cash explicitly
        total_invested = sum(final_weights.values())
        if total_invested < 1.0:
            final_weights["CASH"] = 1.0 - total_invested

        return final_weights
