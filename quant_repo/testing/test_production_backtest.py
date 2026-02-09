import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
from datetime import date, timedelta
from quant_repo.backtest.production_runner import BacktestRunner


class MockStrategy:
    def __init__(self):
        self.counter = 0

    def update(self, row):
        pass

    def generate_signals(self, account):
        # Trade every day
        # Day 0-9: Win 1000
        # Day 10: Loss 5000 (Tail event)
        self.counter += 1

        signals = []
        if self.counter <= 10:
            signals.append(
                {
                    "action": "SELL",  # Sell Vol
                    "quantity": 10,
                    "price": 10.0,
                    "type": "SHORT_VOL",
                    "pnl": 1000.0,  # Deterministic PnL for sim
                }
            )
        elif self.counter == 11:
            signals.append(
                {
                    "action": "BUY",  # Cover
                    "quantity": 10,
                    "price": 20.0,
                    "type": "SHORT_VOL",
                    "pnl": -5000.0,
                }
            )

        return signals


def test_production_backtest():
    print("[TEST] Production Backtest Engine...")

    # 1. Setup Mock Data
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(20)]
    history = pl.DataFrame(
        {"date": dates, "regime": ["LONG_GAMMA"] * 10 + ["SHORT_GAMMA"] * 10}
    )

    # 2. Setup Engine
    engine = BacktestRunner(initial_capital=100_000.0, commission_per_contract=2.0)
    strategy = MockStrategy()

    print("\n--- Running Simulation ---")
    result = engine.run(strategy, history)

    print(f"Final Equity: ${result.equity_curve['equity'][-1]:,.2f}")
    print(f"Sharpe: {result.metrics['sharpe']:.2f}")
    print(f"Max DD: {result.metrics['max_drawdown']:.2%}")
    print(f"Tail Loss (5d): {result.metrics['tail_loss_avg_5d']:.4%}")

    # Verification
    # 10 wins of 1000 = +10,000
    # 1 Loss of 5000 = -5,000
    # Gross PnL = +5,000

    # Costs:
    # 11 trades * 10 qty * $2 comm = $220
    # 11 trades * 10 qty * $0.05 slippage?
    # Slippage logic: cost = comm + (0.05 * qty of 10 = 0.5)
    # Total cost/trade = (2*10) + 0.5 = 20.5
    # Total cost = 11 * 20.5 = 225.5

    # Net PnL ~ 4774.5
    final_eq = result.equity_curve["equity"][-1]
    expected_eq = 100_000 + 5000 - 225.5

    print(f"Expected Equity: {expected_eq}")

    assert abs(final_eq - expected_eq) < 1.0
    assert result.metrics["max_drawdown"] > 0  # The loss should cause DD

    # Regime Stats check
    print("\nRegime Stats:")
    print(result.regime_stats)

    # Should show PnL for LONG_GAMMA (Wins) and SHORT_GAMMA (Loss)
    assert len(result.regime_stats) > 0

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_production_backtest()
