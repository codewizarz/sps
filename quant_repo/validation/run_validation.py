#!/usr/bin/env python3
"""
=============================================================================
RUN VALIDATION — MEOW FINAL BOSS
=============================================================================
Entry-point script to run the full validation pipeline.

Usage (from repo root):
    python quant_repo/validation/run_validation.py

    # Custom paths:
    python quant_repo/validation/run_validation.py \
        --strategy quant_repo/strategies/short_vol/meow_final_boss.py \
        --lake data/master_fo_lake \
        --output quant_repo/research_outputs/validation \
        --mc-runs 1000 \
        --seed 42

Output files (in --output directory):
    walkforward_results.csv
    stress_test_results.csv
    monte_carlo_results.csv
    realistic_backtest_results.csv
    final_validation_report.json
=============================================================================
"""

import argparse
import sys
from pathlib import Path

# Allow running from any working directory
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_repo.validation.strategy_validator import ValidationConfig, ValidationRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full validation pipeline for Meow Final Boss strategy"
    )
    parser.add_argument(
        "--strategy",
        default=str(ROOT / "quant_repo" / "strategies" / "short_vol" / "meow_final_boss.py"),
        help="Path to frozen strategy file",
    )
    parser.add_argument(
        "--lake",
        default=str(ROOT / "data" / "master_fo_lake"),
        help="Path to F&O data lake",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "quant_repo" / "research_outputs" / "validation"),
        help="Output directory for validation results",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000_000,
        help="Initial capital (default: 10,000,000)",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=500,
        help="Number of Monte Carlo runs (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = ValidationConfig(
        strategy_path=args.strategy,
        data_lake_path=args.lake,
        output_dir=args.output,
        initial_capital=args.capital,
        monte_carlo_runs=args.mc_runs,
        random_seed=args.seed,
    )

    print(f"\nMEOW FINAL BOSS — VALIDATION PIPELINE")
    print(f"Strategy : {config.strategy_path}")
    print(f"Data Lake: {config.data_lake_path}")
    print(f"Output   : {config.output_dir}")
    print(f"Capital  : Rs {config.initial_capital:,.0f}")
    print(f"MC Runs  : {config.monte_carlo_runs}")
    print()

    runner = ValidationRunner(config)
    results = runner.run_all_validations()

    evaluation = results.get("evaluation", {})
    rec = evaluation.get("recommendation", "unknown").upper()
    is_robust = evaluation.get("is_robust", False)

    print(f"\nFINAL VERDICT: {'✅ ROBUST — ' if is_robust else '🚨 NOT READY — '}{rec}")
    return 0 if rec in ("DEPLOY", "REFINE") else 1


if __name__ == "__main__":
    sys.exit(main())
