from typing import List
import polars as pl
from quant_repo.validation.definitions import (
    ValidationResult,
    ValidationStatus,
    BacktestRunner,
)
from quant_repo.validation.bias import BiasDetector
from quant_repo.validation.stress import StressTester


class ValidationEngine:
    """
    Orchestrates the full validation suite.
    """

    def __init__(self, runner: BacktestRunner):
        self.runner = runner
        self.bias_detector = BiasDetector()
        self.stress_tester = StressTester(runner)

    def validate(self) -> List[ValidationResult]:
        print("[Validation] Starting Baseline Run...")

        # 1. Baseline Run
        df_base = self.runner.run(overrides={})
        baseline_pnl = df_base["pnl_net"].sum() if len(df_base) > 0 else 0.0

        results = []

        # 2. Bias Checks
        print("[Validation] Checking Biases...")
        results.append(self.bias_detector.check_lookahead(df_base))
        results.append(self.bias_detector.check_liquidity_illusion(df_base))

        # 3. Stress Tests
        print("[Validation] Running Stress Tests...")
        results.append(self.stress_tester.run_spread_stress_test(baseline_pnl))

        # 4. Scorecard
        self._print_scorecard(results)

        return results

    def _print_scorecard(self, results: List[ValidationResult]):
        print("\n" + "=" * 40)
        print("VALIDATION SCORECARD")
        print("=" * 40)

        total_score = 0
        n = 0
        failed = False

        for r in results:
            print(f"[{r.status.value}] {r.test_name}: {r.details}")
            total_score += r.score
            n += 1
            if r.status == ValidationStatus.FAIL:
                failed = True

        avg_score = total_score / n if n > 0 else 0
        print("-" * 40)
        print(f"Final Score: {avg_score:.1f} / 100")

        if failed:
            print("VERDICT: **REJECTED**")
        else:
            print("VERDICT: **PASS**")
        print("=" * 40 + "\n")
