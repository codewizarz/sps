import polars as pl
from typing import Dict, List
from quant_repo.analytics.metrics import calc_expectancy, calc_tail_risk


class AttributionEngine:
    """
    Groups trades by Signal and Regime to calculate segmented performance.
    """

    def run_attribution(self, df: pl.DataFrame, group_by_col: str) -> pl.DataFrame:
        """
        Returns a summary DataFrame with stats per group.
        """
        # We want a custom struct result for each group
        # Polars approach: group_by -> agg with custom functions is tricky for complex dicts like expectancy
        # Easier: Iterate unique groups if cardinality is low (Signals/Regimes usually < 10)

        groups = df[group_by_col].unique().to_list()
        results = []

        for g in groups:
            sub = df.filter(pl.col(group_by_col) == g)
            stats = calc_expectancy(sub)
            risk = calc_tail_risk(sub)

            row = {
                group_by_col: g,
                "count": len(sub),
                "total_pnl": sub["pnl_net"].sum(),
                **stats,
                **risk,
            }
            results.append(row)

        return pl.DataFrame(results)

    def analyze_by_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.run_attribution(df, "signal_type")

    def analyze_by_regime(self, df: pl.DataFrame) -> pl.DataFrame:
        if "regime" not in df.columns:
            return pl.DataFrame()
        return self.run_attribution(df, "regime")
