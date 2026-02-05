import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from scipy import stats


@dataclass
class DecayReport:
    is_decaying: bool
    p_value_ks: float
    z_score_win_rate: float
    regime_alerts: List[str]
    description: str


class EdgeSentinel:
    """
    Monitors strategy performance for statistical decay.
    """

    def check_decay(
        self, recent_trades: pl.DataFrame, historical_trades: pl.DataFrame
    ) -> DecayReport:
        """
        Compares recent live trades against historical backtest trades.
        Expects 'pnl' column in both.
        """
        alerts = []
        is_decaying = False

        # Data Prep
        if "pnl" not in recent_trades.columns or "pnl" not in historical_trades.columns:
            return DecayReport(False, 1.0, 0.0, ["Missing PnL Data"], "Error")

        recent_pnl = recent_trades["pnl"].to_numpy()
        hist_pnl = historical_trades["pnl"].to_numpy()

        if len(recent_pnl) < 30:
            return DecayReport(
                False, 1.0, 0.0, ["Insufficient Data (<30 trades)"], "Waiting for data"
            )

        # 1. Distribution Shift (KS Test)
        # Tests if two samples are drawn from the same distribution
        ks_stat, p_value_ks = stats.ks_2samp(recent_pnl, hist_pnl)

        if p_value_ks < 0.05:
            # Reject Null Hypothesis (same dist) -> Distributions are different
            alerts.append(f"Distribution Shift Detected (KS p={p_value_ks:.4f})")
            is_decaying = True

        # 2. Win Rate Drift (Z-Score)
        # proportion test
        n1 = len(hist_pnl)
        p1 = np.sum(hist_pnl > 0) / n1

        n2 = len(recent_pnl)
        p2 = np.sum(recent_pnl > 0) / n2

        # Pooled probability
        p_pool = (np.sum(hist_pnl > 0) + np.sum(recent_pnl > 0)) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

        if se == 0:
            z_score = 0.0
        else:
            z_score = (p2 - p1) / se

        # If Z-Score < -1.96 (Sig Level 5%, 1-tailed roughly -1.64, 2-tailed -1.96)
        # We care if Win Rate DROPS.
        if z_score < -1.96:
            alerts.append(f"Win Rate Deterioration (Z={z_score:.2f})")
            is_decaying = True

        # 3. Regime Dependency (Optional Check)
        if "regime" in recent_trades.columns and "regime" in historical_trades.columns:
            # Check if any regime flipped from +Exp to -Exp
            regimes = recent_trades["regime"].unique()
            for r in regimes:
                hist_mean = historical_trades.filter(pl.col("regime") == r)[
                    "pnl"
                ].mean()
                recent_mean = recent_trades.filter(pl.col("regime") == r)["pnl"].mean()

                if hist_mean > 0 and recent_mean < 0:
                    alerts.append(
                        f"Regime Breakdown in {r} (Exp: {hist_mean:.2f} -> {recent_mean:.2f})"
                    )
                    is_decaying = True

        description = "Strategy Healthy"
        if is_decaying:
            description = "EDGE DECAY DETECTED: " + "; ".join(alerts)

        return DecayReport(
            is_decaying=is_decaying,
            p_value_ks=float(p_value_ks),
            z_score_win_rate=float(z_score),
            regime_alerts=alerts,
            description=description,
        )
