import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
from datetime import date, timedelta
from quant_repo.backtest.production_runner import BacktestRunner


# Mock Strategy that behaves like VRP
class SimpleVRPStrategy:
    def __init__(self):
        pass

    def update(self, row):
        pass

    def generate_signals(self, account):
        # Sell 1 contract of Short Vol every day
        return [
            {
                "action": "SELL",
                "quantity": 1,
                "price": 100.0,  # Dummy price, PnL is what matters in this high-level sim
                "type": "SHORT_VOL",
                "pnl": np.random.normal(50, 200),  # Mean $50/day, Std $200
            }
        ]


def run_war_grade_backtest():
    print("Initializing War-Grade Backtest...")

    # 1. Generate 3 Years of Data (approx 750 trading days)
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(1095)]
    # Filter weekends/holidays roughly? No, let's just use all for speed.

    # Regime:
    # Year 1: Bull (Long Gamma)
    # Year 2: Bear (Short Gamma)
    # Year 3: Chop (Transition)

    regimes = []
    for d in dates:
        if d.year == 2023:
            regimes.append("LONG_GAMMA")
        elif d.year == 2024:
            regimes.append("SHORT_GAMMA")
        else:
            regimes.append("TRANSITION")

    history = pl.DataFrame(
        {
            "date": dates,
            "regime": regimes,
            "iv": np.random.uniform(10, 30, len(dates)),
            "price": np.random.uniform(3000, 4000, len(dates)),
        }
    )

    # 2. Setup Engine
    runner = BacktestRunner(
        initial_capital=1_000_000.0,
        commission_per_contract=2.0,  # $2 per contract
    )

    strategy = SimpleVRPStrategy()

    print(f"Running simulation over {len(history)} days...")
    result = runner.run(strategy, history)

    # 3. Report
    print("\n" + "=" * 40)
    print("BACKTEST RESULTS")
    print("=" * 40)
    print(f"Final Equity:   ${result.equity_curve['equity'][-1]:,.2f}")
    print(f"Total Return:   {(result.equity_curve['equity'][-1] / 1_000_000 - 1):.2%}")
    print(f"CAGR:           {result.metrics['cagr']:.2%}")
    print(f"Sharpe Ratio:   {result.metrics['sharpe']:.2f}")
    print(f"Sortino Ratio:  {result.metrics['sortino']:.2f}")
    print(f"Max Drawdown:   {result.metrics['max_drawdown']:.2%}")
    print(f"Tail Loss (5d): {result.metrics['tail_loss_avg_5d']:.2%}")
    print("-" * 40)
    print("Regime Performance:")
    print(result.regime_stats)
    print("=" * 40)


if __name__ == "__main__":
    run_war_grade_backtest()
