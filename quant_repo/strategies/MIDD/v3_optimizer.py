import pandas as pd
import numpy as np
import logging
import itertools
from v3_smart_risk_strategy import BacktestEngine, StrategyConfig

# Suppress standard logging for the loop
logging.getLogger("v3_smart_risk_strategy").setLevel(logging.WARNING)


def run_optimization():
    # Define parameter ranges to test
    param_grid = {
        "stop_loss_multiple": [1.4, 1.5, 1.6, 1.8],
        "profit_target_pct": [0.65, 0.70, 0.75, 0.80],
        "max_lots_cap": [150, 200, 250],
        "trailing_stop_activation": [0.20, 0.25, 0.30],
        "trailing_stop_distance": [0.15, 0.20, 0.25],
        "base_risk_pct": [0.05, 0.06, 0.07],
    }

    # Generate all combinations
    keys = param_grid.keys()
    values = (param_grid[key] for key in keys)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total combinations to test: {len(combinations)}")

    results = []

    # Setup initial engine to get data
    engine = BacktestEngine()
    engine._setup_db()  # Must be called after init to set lake_path and con
    prices = engine.get_spot_data("NIFTY")

    for i, params in enumerate(combinations):
        if i % 10 == 0:
            print(f"Testing combination {i}/{len(combinations)}...")

        config = StrategyConfig(**params)
        engine.config = config
        engine.trade_manager.config = config
        engine.streak_tracker.config = config

        # Reset engine state
        engine.equity = config.initial_capital
        engine.peak_equity = config.initial_capital
        engine.trades = []
        engine.equity_curve = []
        engine.weekly_stop_count = {}
        engine.circuit_breaker_active = False

        try:
            trades_df, equity_df = engine.run(
                prices, start_date="2025-04-01", end_date="2026-03-01"
            )

            if trades_df.empty:
                continue

            print(
                f"  -> Found {len(trades_df)} trades, P&L: {trades_df['Net_PnL'].sum():.0f}"
            )

            total_pnl = trades_df["Net_PnL"].sum()
            ret_pct = total_pnl / config.initial_capital * 100

            eq_values = [config.initial_capital] + equity_df["Equity"].tolist()
            eq_series = pd.Series(eq_values)
            peak = eq_series.cummax()
            dd = (peak - eq_series) / peak
            max_dd = dd.max() * 100

            results.append(
                {
                    **params,
                    "Return_Pct": ret_pct,
                    "Max_DD_Pct": max_dd,
                    "Return_to_DD": ret_pct / max_dd if max_dd > 0 else 0,
                    "Win_Rate": (trades_df["Net_PnL"] > 0).mean() * 100,
                }
            )
        except Exception as e:
            # print(f"Error testing {params}: {e}")
            continue

    # Convert to DataFrame and sort by Return_Pct
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Return_Pct", ascending=False)

    # Save results
    results_df.to_csv("v3_optimization_results.csv", index=False)
    print("\nOptimization complete. Best results:")
    print(results_df.head(10))


if __name__ == "__main__":
    run_optimization()
