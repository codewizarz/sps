import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import shutil
import polars as pl
from quant_repo.nautilus.orchestrator import NautilusOrchestrator


def test_nautilus_orchestrator():
    print("[TEST] Nautilus Orchestrator...")

    # Setup temp dir
    test_dir = "./temp_test_results"
    if Path(test_dir).exists():
        shutil.rmtree(test_dir)

    orchestrator = NautilusOrchestrator(storage_path=test_dir)

    base_config = {"start_date": "2023-01-01", "symbol": "SPX"}

    param_grid = {"lookback": [10, 20], "threshold": [0.1, 0.2]}

    # 2 * 2 = 4 combinations

    print("\n--- Running Grid Search ---")
    results = orchestrator.run_experiment("test_exp_001", base_config, param_grid)

    print("\nResults Summary:")
    print(results)

    # Verification
    assert len(results) == 4
    assert "sharpe" in results.columns

    # Check if files exist
    summary_path = Path(test_dir) / "test_exp_001_summary.parquet"
    assert summary_path.exists()

    # Cleanup
    if Path(test_dir).exists():
        shutil.rmtree(test_dir)

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_nautilus_orchestrator()
