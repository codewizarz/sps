import polars as pl
import itertools
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid
from datetime import datetime


@dataclass
class ExperimentResult:
    experiment_id: str
    params: Dict[str, Any]
    sharpe: float
    max_drawdown: float
    result_path: str


class NautilusOrchestrator:
    """
    Command center for running large-scale NautilusTrader backtests.
    """

    def __init__(self, storage_path: str = "./backtest_results"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def generate_configs(
        self, base_config: Dict[str, Any], parameter_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generates Cartesian product of configuration variations.
        """
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())
        combinations = list(itertools.product(*values))

        configs = []
        for combo in combinations:
            # Deep copy base
            new_config = base_config.copy()

            # Update strategy params
            # Warning: Assumes flat structure or specific 'strategy' key
            # Let's assume params go into 'strategy_params' dict
            if "strategy_params" not in new_config:
                new_config["strategy_params"] = {}

            for k, v in zip(keys, combo):
                new_config["strategy_params"][k] = v

            configs.append(new_config)

        return configs

    def run_experiment(
        self, experiment_name: str, base_config: Dict, parameter_grid: Dict
    ) -> pl.DataFrame:
        """
        Runs a grid of backtests. (In reality, would use Ray or multiprocessing).
        For this mock implementation, we iterate sequentially.
        """
        configs = self.generate_configs(base_config, parameter_grid)
        results = []

        print(f"Starting Experiment: {experiment_name} | Runs: {len(configs)}")

        for i, config in enumerate(configs):
            exp_id = f"{experiment_name}_{uuid.uuid4().hex[:8]}"
            print(
                f"[{i + 1}/{len(configs)}] Running {exp_id} with {config['strategy_params']}"
            )

            # --- MOCK NAUTILUS BRIDGE ---
            # In a real impl, we would:
            # 1. Instantiate Nautilus Node
            # 2. Configure Venue/Data
            # 3. Add Strategy with config['strategy_params']
            # 4. node.run()
            # 5. Extract results

            # Simulating result extraction
            # Let's assume Sharpe is function of some param to verify optimization works
            # e.g., 'lookback' * 0.1
            params = config["strategy_params"]
            mock_sharpe = min(
                (params.get("lookback", 10) / 10.0)
                + (params.get("threshold", 0.0) * 5),
                3.0,
            )
            mock_dd = -0.10

            # Log results to disk (Parquet)
            result_dir = self.storage_path / experiment_name / exp_id
            result_dir.mkdir(parents=True, exist_ok=True)

            # Dummy Trade Log
            trades = pl.DataFrame({"id": range(10), "pnl": [100.0] * 10})
            trades.write_parquet(result_dir / "trades.parquet")

            results.append(
                {
                    "experiment_id": exp_id,
                    "params": str(params),  # Flatten for dataframe
                    "sharpe": mock_sharpe,
                    "max_drawdown": mock_dd,
                    "path": str(result_dir),
                }
            )

        df_results = pl.DataFrame(results)
        summary_path = self.storage_path / f"{experiment_name}_summary.parquet"
        df_results.write_parquet(summary_path)

        print(f"Experiment Complete. Summary saved to {summary_path}")
        return df_results
