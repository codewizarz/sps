import polars as pl
import numpy as np
from typing import List, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from quant_repo.factory.vector_tester import VectorBacktester


@dataclass
class Window:
    train_start: int  # timestamp
    train_end: int
    test_start: int
    test_end: int


class WindowSplitter:
    """
    Generates rolling Train/Test windows.
    """

    def split(
        self, timestamps: List[int], train_size: int, test_size: int
    ) -> List[Window]:
        """
        timestamps: Sorted list of timestamps.
        train_size: Number of indices for training.
        test_size: Number of indices for testing.
        """
        if len(timestamps) < train_size + test_size:
            return []

        windows = []
        # Step size = test_size for contiguous rolling

        current_idx = 0
        while current_idx + train_size + test_size <= len(timestamps):
            train_end_idx = current_idx + train_size
            test_end_idx = train_end_idx + test_size

            # Using indices to get timestamps might be tricky if gaps, but let's assume contiguous simulation steps
            # or just use indices as bounds if accessing via iloc/slice

            w = Window(
                train_start=timestamps[current_idx],
                train_end=timestamps[train_end_idx - 1],
                test_start=timestamps[train_end_idx],
                test_end=timestamps[test_end_idx - 1],
            )
            windows.append(w)

            current_idx += test_size  # Roll forward

        return windows


class WalkForwardOptimizer:
    """
    Orchestrates Train (Optimize) -> Test (OOS) loop.
    """

    def __init__(self):
        self.tester = VectorBacktester()
        self.splitter = WindowSplitter()

    def _optimize(
        self, df_train: pl.DataFrame, param_grid: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Finds best params in Train set.
        MOCK: In real life, uses grid search.
        Here, we assume 'signal' column construction might change?
        Actually, VectorBacktester just takes a dataframe with 'signal'.
        Optimization usually implies changing how 'signal' is generated (e.g. Moving Average Window).

        For this prototype, we will assume 'param_grid' contains different 'signal_col' names
        that have been pre-computed in the dataframe (e.g., 'signal_ma_10', 'signal_ma_50').
        """
        best_sharpe = -999.0
        best_params = None

        for params in param_grid:
            # params = {"signal_col": "signal_A"}
            col = params.get("signal_col", "signal")

            if col not in df_train.columns:
                continue

            res = self.tester.run(df_train, signal_col=col)
            if res["sharpe"] > best_sharpe:
                best_sharpe = res["sharpe"]
                best_params = params

        return best_params

    def run(
        self,
        df: pl.DataFrame,
        param_grid: List[Dict[str, Any]],
        train_size: int = 252,
        test_size: int = 63,
    ) -> pl.DataFrame:
        timestamps = df["timestamp"].to_list()
        windows = self.splitter.split(timestamps, train_size, test_size)

        oos_results = []

        print(f"[WalkForward] Running {len(windows)} segments...")

        for i, w in enumerate(windows):
            # Slice Data
            # Polars filter by range
            df_train = df.filter(
                (pl.col("timestamp") >= w.train_start)
                & (pl.col("timestamp") <= w.train_end)
            )
            df_test = df.filter(
                (pl.col("timestamp") >= w.test_start)
                & (pl.col("timestamp") <= w.test_end)
            )

            # 1. Train
            best_params = self._optimize(df_train, param_grid)
            if not best_params:
                print(f"  Seg {i}: Optimization failed (no params)")
                continue

            sig_col = best_params["signal_col"]

            # 2. Test (OOS)
            # Run simulation on Test Data using Best Params
            # We use VectorBacktester logic but just want the PnL series
            # VectorBacktester.run returns summary. We need the series.
            # Re-implementing simplified pnl calc here or we'd need to modify VectorBacktester to return series.

            # Let's simple-calc
            # Pnl = Signal[prev] * Ret
            test_pnl = df_test.select(
                [
                    "timestamp",
                    "spot_return",
                    pl.col(sig_col).cast(pl.Float64).alias("active_signal"),
                ]
            ).with_columns(
                [
                    (pl.col("active_signal").shift(1) * pl.col("spot_return"))
                    .fill_null(0)
                    .alias("pnl"),
                    pl.lit(sig_col).alias("chosen_param"),
                ]
            )

            oos_results.append(test_pnl)

            print(f"  Seg {i}: Chosen {sig_col} (TrainSharpe: ?)")

        if not oos_results:
            return pl.DataFrame()

        # Stitch
        return pl.concat(oos_results)
