from typing import List
import polars as pl
from pathlib import Path
from quant_repo.research.edges import EdgeDefinition
from quant_repo.research.persistence import PersistenceAnalyzer


class EdgeScanner:
    """
    Orchestrates the scanning of multiple edges over historical data.
    """

    def __init__(self, edges: List[EdgeDefinition]):
        self.edges = edges
        self.analyzer = PersistenceAnalyzer()

    def scan(self, df_history: pl.DataFrame) -> pl.DataFrame:
        results = []

        for edge in self.edges:
            print(f"[Scanner] analyzing {edge.name}...")
            pnl_series = edge.generate_pnl_proxies(df_history)

            if pnl_series.is_empty():
                print(f"  -> No PnL generated (missing data?)")
                continue

            metrics = self.analyzer.analyze(pnl_series)

            verdict = "IGNORE"
            if metrics.total_sharpe > 1.0 and metrics.stability_score > 0.5:
                verdict = "GOLD"  # High Sharpe, Stable
            elif metrics.total_sharpe > 0.5:
                verdict = "SILVER"

            # Downgrade if decaying fast
            if metrics.decay_slope < -0.2:
                verdict += " (DECAYING)"

            results.append(
                {
                    "edge_name": edge.name,
                    "sharpe": metrics.total_sharpe,
                    "stability": metrics.stability_score,
                    "decay": metrics.decay_slope,
                    "win_pct": metrics.win_year_pct,
                    "verdict": verdict,
                }
            )

        return pl.DataFrame(results).sort("sharpe", descending=True)
