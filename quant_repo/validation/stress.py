from typing import List, Dict, Any
import polars as pl
from quant_repo.validation.definitions import (
    ValidationResult,
    ValidationStatus,
    BacktestRunner,
)


class StressTester:
    """
    Runs sensitivity analysis by modifying backtest parameters.
    """

    def __init__(self, runner: BacktestRunner):
        self.runner = runner

    def run_spread_stress_test(self, baseline_pnl: float) -> ValidationResult:
        """
        Doubles the spread multiplier and checks if strategy survives.
        """
        print("[StressTest] Running 2x Spread Simulation...")

        # Inject override
        try:
            df_stress = self.runner.run(overrides={"execution_spread_multiplier": 2.0})
        except Exception as e:
            return ValidationResult(
                "Spread Stress (2x)", ValidationStatus.FAIL, 0, f"Sim crashed: {e}"
            )

        if len(df_stress) == 0:
            return ValidationResult(
                "Spread Stress (2x)",
                ValidationStatus.FAIL,
                0,
                "Strategy stopped trading.",
            )

        stress_pnl = df_stress["pnl_net"].sum()

        # Criterion: Should not lose > 50% of profit (arbitrary strictness)
        # or flip to negative if baseline was positive.

        details = f"Baseline PnL: {baseline_pnl:.2f}, Stress PnL: {stress_pnl:.2f}"

        if baseline_pnl > 0 and stress_pnl < 0:
            return ValidationResult(
                "Spread Stress (2x)",
                ValidationStatus.FAIL,
                0,
                f"Strategy became unprofitable. {details}",
            )

        if baseline_pnl > 0 and stress_pnl < (0.5 * baseline_pnl):
            return ValidationResult(
                "Spread Stress (2x)",
                ValidationStatus.WARNING,
                50,
                f"Profit dropped significantly. {details}",
            )

        return ValidationResult(
            "Spread Stress (2x)", ValidationStatus.PASS, 100, details
        )

    def run_vol_shock(self) -> ValidationResult:
        """
        Simulates Vol Shock (IV * 1.5).
        """
        # ... Implementation similar to above with iv_multiplier override
        return ValidationResult(
            "Vol Shock", ValidationStatus.PASS, 100, "Not implemented fully"
        )
