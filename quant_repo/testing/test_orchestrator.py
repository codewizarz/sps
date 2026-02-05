import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from quant_repo.portfolio.orchestrator import StrategyOrchestrator
from quant_repo.features.regime import MarketRegime


def test_orchestrator():
    print("[TEST] Dynamic Strategy Orchestrator...")

    # 1. Define Regime Matrix
    regime_matrix = {
        MarketRegime.BULL_QUIET: {"ShortVol": 1.0, "Trend": 0.5, "LongVol": 0.2},
        MarketRegime.CRISIS: {
            "ShortVol": 0.0,  # Kill Switch
            "Trend": 0.0,
            "LongVol": 2.0,  # Maximize Hedge
        },
    }

    orchestrator = StrategyOrchestrator(regime_matrix)

    # 2. Define Base Portfolio (from HRP)
    base_weights = {
        "VRP_Strategy_A": 0.40,
        "Trend_Strategy_B": 0.40,
        "Hedge_Strategy_C": 0.20,
    }

    strat_types = {
        "VRP_Strategy_A": "ShortVol",
        "Trend_Strategy_B": "Trend",
        "Hedge_Strategy_C": "LongVol",
    }

    # 3. Test BULL_QUIET
    weights_bull = orchestrator.calculate_final_weights(
        base_weights, MarketRegime.BULL_QUIET, strat_types
    )

    print("\n--- Regime: BULL QUIET ---")
    for k, v in weights_bull.items():
        print(f"{k}: {v:.2f}")

    # Validation
    # ShortVol: 0.4 * 1.0 = 0.4
    # Trend: 0.4 * 0.5 = 0.2
    # LongVol: 0.2 * 0.2 = 0.04
    # Total Invested: 0.64. Cash: 0.36

    assert round(weights_bull["VRP_Strategy_A"], 2) == 0.40
    assert round(weights_bull["Trend_Strategy_B"], 2) == 0.20
    assert round(weights_bull["Hedge_Strategy_C"], 2) == 0.04
    assert round(weights_bull["CASH"], 2) == 0.36

    # 4. Test CRISIS
    weights_crisis = orchestrator.calculate_final_weights(
        base_weights, MarketRegime.CRISIS, strat_types
    )

    print("\n--- Regime: CRISIS ---")
    for k, v in weights_crisis.items():
        print(f"{k}: {v:.2f}")

    # Validation
    # ShortVol: 0.0
    # Trend: 0.0
    # LongVol: 0.2 * 2.0 = 0.4
    # Cash: 0.6

    assert weights_crisis["VRP_Strategy_A"] == 0.0
    assert weights_crisis["Trend_Strategy_B"] == 0.0
    assert weights_crisis["Hedge_Strategy_C"] == 0.40
    assert weights_crisis["CASH"] == 0.60

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_orchestrator()
