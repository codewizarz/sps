import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
from typing import Dict, Any
from quant_repo.validation.engine import ValidationEngine
from quant_repo.validation.definitions import BacktestRunner, ValidationStatus


class MockBacktestRunner(BacktestRunner):
    """
    Mocks a backtest strategies behavior under different conditions.
    """

    def run(self, overrides: Dict[str, Any] = None) -> pl.DataFrame:
        overrides = overrides or {}
        spread_mult = overrides.get("execution_spread_multiplier", 1.0)

        # Generate Fake Trades
        # Base: 1000 PnL, 0.1 Slippage
        # If Spread Mult = 2.0 -> Cost increases -> PnL drops

        base_pnl = 100.0
        # If spread doubles, slippage doubles.
        # Say base slippage is 10.0 per trade.
        # Net = 100.
        # Stress Slippage = 20.0. Net = 90. Still profitable.

        rows = []
        for i in range(10):
            slippage = 5.0 * spread_mult  # 5 * 1 = 5, 5 * 2 = 10
            gross_pnl = 105.0
            net_pnl = gross_pnl - slippage  # 100 base, 95 stress

            rows.append(
                {
                    "entry_time": 1000 + i,
                    "exit_time": 2000 + i,
                    "pnl_net": net_pnl,
                    "cost_slippage": slippage,
                }
            )

        return pl.DataFrame(rows)


class FragileBacktestRunner(BacktestRunner):
    """
    A strategy that fails stress tests.
    """

    def run(self, overrides: Dict[str, Any] = None) -> pl.DataFrame:
        overrides = overrides or {}
        spread_mult = overrides.get("execution_spread_multiplier", 1.0)

        rows = []
        for i in range(10):
            # Thin margin strategy
            # Base Slippage = 5. Gap = 6. Net = +1.
            # Stress Slippage = 10. Gap = 6. Net = -4. (Fails)

            slippage = 5.0 * spread_mult
            gross_pnl = 6.0
            net_pnl = gross_pnl - slippage

            rows.append(
                {
                    "entry_time": 1000 + i,
                    "exit_time": 2000 + i,
                    "pnl_net": net_pnl,
                    "cost_slippage": slippage,
                }
            )
        return pl.DataFrame(rows)


def test_validation_framework():
    print("[TEST] Validation Framework...")

    # 1. Test Robust Strategy
    print("\n[TEST] 1. Robust Strategy")
    robust_runner = MockBacktestRunner()
    engine_pass = ValidationEngine(robust_runner)
    results_pass = engine_pass.validate()

    # Should PASS everything
    # Liquidity Illusion check might warn/pass since slippage > 0

    failures = [r for r in results_pass if r.status == ValidationStatus.FAIL]
    assert len(failures) == 0, f"Expected PASS, got failures: {failures}"

    # 2. Test Fragile Strategy
    print("\n[TEST] 2. Fragile Strategy (Spread Stress)")
    fragile_runner = FragileBacktestRunner()
    engine_fail = ValidationEngine(fragile_runner)
    results_fail = engine_fail.validate()

    # Needs to FAIL Spread Stress
    spread_stress = next(r for r in results_fail if "Spread Stress" in r.test_name)
    print(f"  Result: {spread_stress.status} ({spread_stress.details})")
    assert spread_stress.status == ValidationStatus.FAIL

    print("[TEST] SUCCESS")


if __name__ == "__main__":
    test_validation_framework()
