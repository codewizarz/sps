import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import polars as pl


@dataclass
class SimulationConfig:
    n_sims: int = 2000
    initial_capital: float = 100_000.0
    prob_skip_fill: float = 0.05  # Probability of skipping a winning trade
    prob_slippage_shock: float = 0.01  # Probability of a large slippage event
    shock_factor: float = 2.0  # Multiplier for loss/reduced win under shock


@dataclass
class SimulationResult:
    max_drawdowns: List[float]
    final_equities: List[float]
    ruin_probability: float  # % of sims where equity < 0
    var_99_equity: float  # 1st percentile of final equity (Capital at Risk)
    median_dd: float  # Median Max Drawdown
    worst_case_dd: float  # 99th percentile Max Drawdown


class MonteCarloEngine:
    """
    Vectorized Monte Carlo Resampler for Strategy Equity Curves.
    """

    def run(self, trade_pnl: np.ndarray, config: SimulationConfig) -> SimulationResult:
        """
        Runs the simulation.
        trade_pnl: Numpy array of trade PnL values (chronological not required for bootstrap,
                   but usually we take the set of historical PnLs).
        """
        n_trades = len(trade_pnl)
        if n_trades == 0:
            return SimulationResult([], [], 0.0, 0.0, 0.0, 0.0)

        # 1. Generate Bootstrap Index Matrix (n_sims, n_trades)
        # Random sampling with replacement
        rng = np.random.default_rng(42)
        indices = rng.integers(0, n_trades, size=(config.n_sims, n_trades))

        # 2. Select PnLs
        sim_pnls = trade_pnl[indices]  # Shape: (n_sims, n_trades)

        # 3. Apply Stress (Vectorized)

        # A. Skipped Fills (Winning trades become 0)
        # Generate mask: 1 if stress event, 0 otherwise
        skip_mask = rng.random((config.n_sims, n_trades)) < config.prob_skip_fill
        # Only apply to winners > 0
        winners_mask = sim_pnls > 0
        final_skip_mask = skip_mask & winners_mask

        # Apply: Zero out
        sim_pnls[final_skip_mask] = 0.0

        # B. Slippage Shock (Losses amplify, Wins shrink)
        shock_mask = rng.random((config.n_sims, n_trades)) < config.prob_slippage_shock
        # If PnL < 0, PnL * Factor (Larger Loss)
        # If PnL > 0, PnL / Factor (Smaller Win) or PnL - Shock?
        # Let's say Shock means "Bad execution".
        # Loss -> 2x Loss. Win -> 0.5x Win.

        # Loss mask
        loss_mask = sim_pnls < 0
        sim_pnls[shock_mask & loss_mask] *= config.shock_factor

        # Win mask
        win_mask = sim_pnls > 0
        sim_pnls[shock_mask & win_mask] /= config.shock_factor

        # 4. Calculate Equity Curves
        # Add initial capital
        # Cumulative Sum along trades
        cumulative_pnl = np.cumsum(sim_pnls, axis=1)
        equity_curves = config.initial_capital + cumulative_pnl

        final_equities = equity_curves[:, -1]

        # 5. Calculate Metrics

        # Risk of Ruin (Any point < 0)
        # Check min equity for each sim
        min_equities = np.min(equity_curves, axis=1)
        ruin_count = np.sum(min_equities <= 0)
        prob_ruin = ruin_count / config.n_sims

        # Max Drawdown per Sim
        # DD = (Peak - Current) / Peak (if % mode) or Peak - Current (cash mode)
        # Let's use Cash Drawdown for simplicity/speed
        running_max = np.maximum.accumulate(equity_curves, axis=1)
        # Ensure running max is at least initial capital (trades could start neg)
        running_max = np.maximum(running_max, config.initial_capital)

        drawdowns = running_max - equity_curves
        max_drawdowns = np.max(drawdowns, axis=1)

        # Stats
        var_99_eq = np.percentile(final_equities, 1)  # 1st percentile outcome
        median_dd = np.median(max_drawdowns)
        worst_dd = np.percentile(max_drawdowns, 99)  # 99th percentile DD

        return SimulationResult(
            max_drawdowns=max_drawdowns.tolist(),
            final_equities=final_equities.tolist(),
            ruin_probability=float(prob_ruin),
            var_99_equity=float(var_99_eq),
            median_dd=float(median_dd),
            worst_case_dd=float(worst_dd),
        )
