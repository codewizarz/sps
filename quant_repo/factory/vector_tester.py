import polars as pl
import numpy as np
from typing import Dict


class VectorBacktester:
    """
    Stage 1 Filter: High-speed vectorized backtester.
    Assumes simplified execution (mid-price or fixed transaction cost).
    """

    def run(
        self,
        df_history: pl.DataFrame,
        signal_col: str = "signal",
        cost_per_trade: float = 0.0005,
    ) -> Dict[str, float]:
        """
        Calculates PnL based on Signal * NextDayReturn.
        df_history must have: 'timestamp', 'spot_return', and 'signal_col'.

        Returns: {sharpe, win_rate, total_return}
        """
        req = ["timestamp", "spot_return", signal_col]
        if not all(c in df_history.columns for c in req):
            print(f"[VectorTester] Missing columns {req}")
            return {"sharpe": -99.9, "win_rate": 0.0, "total_return": 0.0}

        # PnL = Signal * Return - Cost
        # Signal is usually 1, -1, or 0.

        df_pnl = df_history.with_columns(
            [
                (
                    pl.col(signal_col).shift(
                        1
                    )  # Signal from yesterday acts on today's return
                    * pl.col("spot_return")
                    - (pl.col(signal_col).shift(1).abs() * cost_per_trade)
                )
                .fill_null(0.0)
                .alias("pnl")
            ]
        )

        pnl = df_pnl["pnl"].to_numpy()

        # Stats
        total_pnl = np.sum(pnl)
        mean_ret = np.mean(pnl)
        std_ret = np.std(pnl)

        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

        # Win Rate (of active days)
        active_days = pnl[pnl != 0]
        win_rate = (
            np.sum(active_days > 0) / len(active_days) if len(active_days) > 0 else 0.0
        )

        return {
            "sharpe": float(sharpe),
            "win_rate": float(win_rate),
            "total_return": float(total_pnl),
        }
