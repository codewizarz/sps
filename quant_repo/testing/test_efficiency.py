import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pytest
from datetime import date, timedelta
from quant_repo.analytics.efficiency import CapitalEfficiencyAnalyzer


def test_efficiency_analyzer():
    print("[TEST] Capital Efficiency Analyzer...")

    analyzer = CapitalEfficiencyAnalyzer()

    # Generate mock data for 1 year (365 days)
    # 1. Capital Log: Starts 1M, ends 1.2M
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(365)]
    equity = np.linspace(1_000_000, 1_200_000, 365)
    margin = np.full(365, 500_000)  # 50% margin usage constant

    capital_log = pl.DataFrame({"date": dates, "equity": equity})
    margin_log = pl.DataFrame({"date": dates, "margin_used": margin})

    # 2. Trades
    # PnL = 200k total.
    # Hedges cost 20k total.
    trades_data = []

    # Regular Income trades
    trades_data.append(
        {
            "date": date(2023, 6, 1),
            "pnl": 220_000.0,
            "strategy_type": "SHORT_VOL",
            "premium_paid": 0.0,
        }
    )

    # Hedge cost trades
    trades_data.append(
        {
            "date": date(2023, 6, 1),
            "pnl": -20_000.0,
            "strategy_type": "HEDGE",
            "premium_paid": 20_000.0,  # Cost is positive magnitude here for summing?
            # Wait, implementation sums 'premium_paid'.
            # Usually checking implementation: hedge_cost = trades.filter(...)["premium_paid"].sum()
            # So I must ensure 'premium_paid' is logged for hedges.
        }
    )

    trades = pl.DataFrame(trades_data)

    # 3. Market Data (for Convexity)
    # Let's mock uncorrelated (0 convexity)
    iv_change = np.random.normal(0, 0.01, 365)
    # We need daily PnL to correlate.
    # Currently trades are sparse. Let's make trades denser for convexity test.

    # Re-do trades for better stats
    daily_pnl = np.random.normal(500, 1000, 365)  # Mean daily profit
    # Add hedge cost
    daily_premium = np.zeros(365)
    strat_types = ["SHORT_VOL"] * 365

    # Make every 10th day a hedge payment
    for i in range(0, 365, 10):
        daily_pnl[i] -= 500  # PnL hit
        daily_premium[i] = 500  # Explicit cost
        strat_types[i] = "HEDGE"

    trades = pl.DataFrame(
        {
            "date": dates,
            "pnl": daily_pnl,
            "strategy_type": strat_types,
            "premium_paid": daily_premium,
        }
    )

    market_data = pl.DataFrame({"date": dates, "iv_change": iv_change})

    print("\n--- Running Analysis ---")
    report = analyzer.analyze(trades, capital_log, margin_log, market_data)

    print(f"ROM (Ann): {report.rom_annualized:.2%}")
    print(f"Hedge Drag: {report.hedge_drag_bps:.1f} bps")
    print(f"Convexity: {report.convexity_score:.4f}")
    print(f"Tail Risk Ratio: {report.tail_risk_ratio:.4f}")

    # Checks
    # Total PnL approx 500 * 365 - hedge costs...
    # Avg Margin 500k.
    # ROM should be roughly (Total PnL / 500k)

    expected_pnl = daily_pnl.sum()
    expected_rom = expected_pnl / 500_000

    # Allow small float diffs
    assert abs(report.rom_annualized - expected_rom) < 0.01

    # Hedge Drag
    # Total premium = 500 * (365/10) ~ 18000
    # Avg Equity ~ 1.1M
    # Drag ~ 1.6% (160 bps)
    assert report.hedge_drag_bps > 0

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_efficiency_analyzer()
