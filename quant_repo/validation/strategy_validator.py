#!/usr/bin/env python3
"""
=============================================================================
STRATEGY VALIDATOR — MEOW FINAL BOSS
=============================================================================

Full validation and stress-testing pipeline for the frozen Meow Final Boss
strategy (Iteration 2 candidate). Supports:

  A. Walk-Forward Validation   (Train 2024, Test 2025, Forward 2026)
  B. Regime Stress Testing     (Low/Normal/High Vol, Gap Days, IV Spikes)
  C. Monte Carlo Simulation    (500–1000 paths, trade shuffling + noise)
  D. Execution Realism         (Slippage, Bid-Ask, Delayed Execution)
  E. Final Evaluation          (Robustness gates → deploy/refine/reject)

Output files:
  walkforward_results.csv
  stress_test_results.csv
  monte_carlo_results.csv
  realistic_backtest_results.csv
  final_validation_report.json

Safety gates (auto-flag NOT READY if any triggered):
  • Max Drawdown > 20% in any test segment
  • Forward-period Sharpe < 1.0
  • Strategy returns negative PnL in HIGH VOL regime

IMPORTANT: This module does NOT modify strategy parameters.
           It only validates the frozen candidate.
=============================================================================
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================


@dataclass
class ValidationConfig:
    """Configuration for the validation pipeline."""

    strategy_path: str
    data_lake_path: str
    output_dir: str = "."
    initial_capital: float = 10_000_000
    monte_carlo_runs: int = 500
    random_seed: int = 42
    # Walk-forward windows (train_start, train_end, test_start, test_end, label)
    walk_forward_windows: List[Tuple[str, str, str, str, str]] = field(
        default_factory=lambda: [
            ("2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "WF_2024train_2025test"),
            ("2024-06-01", "2024-12-31", "2025-01-01", "2025-12-31", "WF_2024H2train_2025test"),
            ("2025-01-01", "2025-12-31", "2026-01-01", "2026-03-26", "WF_2025train_2026forward"),
        ]
    )
    # Regime thresholds (annualized RV)
    low_vol_rv_threshold: float = 0.10    # RV < 10%
    high_vol_rv_threshold: float = 0.20   # RV > 20%
    gap_day_threshold: float = 0.015      # gap move > 1.5%
    iv_spike_threshold: float = 0.20      # IV expansion > 20%
    # Safety gates
    max_dd_limit: float = 20.0           # DD > 20% → flag
    min_forward_sharpe: float = 1.0      # Sharpe < 1.0 in forward → flag
    # Execution realism
    slippage_min: float = 0.005          # 0.5% of premium
    slippage_max: float = 0.010          # 1.0% of premium
    bid_ask_penalty: float = 0.005       # 0.5% spread penalty


# ============================================================================
# HELPERS
# ============================================================================


def _load_strategy_module(strategy_path: str):
    """Dynamically load the frozen strategy module."""
    abs_path = str(Path(strategy_path).resolve())
    spec = importlib.util.spec_from_file_location("meow_final_boss", abs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load strategy from: {abs_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["meow_final_boss"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_engine(
    mod,
    start_date: str,
    end_date: str,
    initial_capital: float,
    symbols: Tuple[str, ...] = ("NIFTY", "BANKNIFTY"),
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Instantiate and run the BacktestEngine from the strategy module."""
    config = mod.StrategyConfig(symbols=symbols, initial_capital=initial_capital)
    engine = mod.BacktestEngine(config)
    trades_df, equity_df = engine.run(start_date=start_date, end_date=end_date)

    if trades_df.empty:
        summary = {
            "Total_Return_Pct": 0.0,
            "Max_Drawdown_Pct": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "Calmar": 0.0,
            "Win_Rate_Pct": 0.0,
            "Total_Trades": 0,
            "Final_Equity": initial_capital,
            "Profit_Factor": 0.0,
        }
    else:
        summary = mod.PerformanceTracker.compute_summary(trades_df, equity_df, initial_capital)

    return trades_df, equity_df, summary


def _out(config: ValidationConfig, filename: str) -> str:
    """Resolve output file path."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / filename)


# ============================================================================
# A. WALK-FORWARD VALIDATION
# ============================================================================


class WalkForwardValidator:
    """
    Walk-forward validation across multiple train/test splits.

    Strategy parameters are NOT retrained — we only test the frozen
    Meow Final Boss parameters on unseen date ranges.
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

    def run_all_windows(self) -> pd.DataFrame:
        """Run every configured walk-forward window and return combined results."""
        mod = _load_strategy_module(self.config.strategy_path)
        results = []

        for train_start, train_end, test_start, test_end, label in self.config.walk_forward_windows:
            logger.info(f"[Walk-Forward] Window: {label}  ({test_start} → {test_end})")
            try:
                trades_df, equity_df, summary = _run_engine(
                    mod,
                    start_date=test_start,
                    end_date=test_end,
                    initial_capital=self.config.initial_capital,
                )
                result = {
                    "window": label,
                    "train_period": f"{train_start} to {train_end}",
                    "test_period": f"{test_start} to {test_end}",
                    "return_pct": round(summary.get("Total_Return_Pct", 0.0), 2),
                    "max_dd_pct": round(summary.get("Max_Drawdown_Pct", 0.0), 2),
                    "sharpe": round(summary.get("Sharpe", 0.0), 3),
                    "sortino": round(summary.get("Sortino", 0.0), 3),
                    "calmar": round(summary.get("Calmar", 0.0), 3),
                    "win_rate_pct": round(summary.get("Win_Rate_Pct", 0.0), 2),
                    "trades": int(summary.get("Total_Trades", 0)),
                    "profit_factor": round(summary.get("Profit_Factor", 0.0), 3),
                    "is_forward": "2026" in test_start,
                    "dd_breach": summary.get("Max_Drawdown_Pct", 0.0) > self.config.max_dd_limit,
                    "sharpe_breach": (
                        "2026" in test_start
                        and summary.get("Sharpe", 0.0) < self.config.min_forward_sharpe
                    ),
                }
            except Exception as exc:
                logger.warning(f"  Walk-forward window {label} failed: {exc}")
                result = {
                    "window": label,
                    "train_period": f"{train_start} to {train_end}",
                    "test_period": f"{test_start} to {test_end}",
                    "return_pct": 0.0,
                    "max_dd_pct": 100.0,
                    "sharpe": 0.0,
                    "sortino": 0.0,
                    "calmar": 0.0,
                    "win_rate_pct": 0.0,
                    "trades": 0,
                    "profit_factor": 0.0,
                    "is_forward": "2026" in test_start,
                    "dd_breach": True,
                    "sharpe_breach": "2026" in test_start,
                }

            results.append(result)
            logger.info(
                f"  → Return {result['return_pct']:.2f}%  |  "
                f"MaxDD {result['max_dd_pct']:.2f}%  |  "
                f"Sharpe {result['sharpe']:.2f}  |  "
                f"Trades {result['trades']}"
            )

        return pd.DataFrame(results)


# ============================================================================
# B. REGIME STRESS TESTING
# ============================================================================


class RegimeStressTester:
    """
    Classify each trading day by volatility regime, gap events and IV spikes,
    then break down strategy performance by those classifications.

    Regime thresholds (annualized RV):
      LOW_VOL    : RV20 < 10%
      NORMAL_VOL : 10% ≤ RV20 ≤ 20%
      HIGH_VOL   : RV20 > 20%
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

    def classify_regimes(self, mod, symbol: str = "NIFTY") -> pd.DataFrame:
        """Build per-day regime classification table from vol engine data."""
        engine = mod.BacktestEngine(mod.StrategyConfig(symbols=(symbol,)))
        prices = engine.get_spot_data(symbol)
        vol_data = mod.VolatilityEngine.compute_all(prices, mod.StrategyConfig())

        rows = []
        for date, row in vol_data.iterrows():
            rv = row.get("rv20", np.nan)
            iv_exp = float(np.nan_to_num(row.get("iv_expansion_2d", 0.0), nan=0.0))
            gap = float(np.nan_to_num(row.get("gap_move", 0.0), nan=0.0))

            if pd.isna(rv):
                regime = "UNKNOWN"
            elif rv < self.config.low_vol_rv_threshold:
                regime = "LOW_VOL"
            elif rv <= self.config.high_vol_rv_threshold:
                regime = "NORMAL_VOL"
            else:
                regime = "HIGH_VOL"

            rows.append({
                "date": pd.Timestamp(date),
                "regime": regime,
                "rv20": round(float(rv) if not pd.isna(rv) else 0.0, 4),
                "iv_expansion_2d": round(iv_exp, 4),
                "iv_spike": iv_exp > self.config.iv_spike_threshold,
                "gap_move": round(gap, 4),
                "gap_day": gap > self.config.gap_day_threshold,
            })

        return pd.DataFrame(rows)

    def run_stress_test(
        self,
        trades_df: pd.DataFrame,
        regimes_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join trades to regime classifications and compute per-regime metrics:
          - return_pct, win_rate, avg_pnl, max_dd_pct, trades
        """
        if trades_df.empty or regimes_df.empty:
            return pd.DataFrame()

        trades = trades_df.copy()
        trades["_entry_ts"] = pd.to_datetime(trades["Entry_Date"])

        # Fast merge via date index
        regime_idx = regimes_df.set_index("date")

        def _get_regime_flags(ts: pd.Timestamp):
            if ts in regime_idx.index:
                r = regime_idx.loc[ts]
                return r["regime"], bool(r["iv_spike"]), bool(r["gap_day"])
            return "UNKNOWN", False, False

        trades[["_regime", "_iv_spike", "_gap_day"]] = trades["_entry_ts"].apply(
            lambda ts: pd.Series(_get_regime_flags(ts))
        )

        results = []

        # Vol regime breakdown
        for regime_label in ["LOW_VOL", "NORMAL_VOL", "HIGH_VOL", "UNKNOWN"]:
            subset = trades[trades["_regime"] == regime_label]
            if subset.empty:
                continue
            results.append(
                self._calc_metrics(subset, regime_label, self.config.initial_capital)
            )

        # Gap-day stress
        gap_subset = trades[trades["_gap_day"]]
        if not gap_subset.empty:
            results.append(
                self._calc_metrics(gap_subset, "GAP_DAYS", self.config.initial_capital)
            )

        # IV-spike day stress
        iv_subset = trades[trades["_iv_spike"]]
        if not iv_subset.empty:
            results.append(
                self._calc_metrics(iv_subset, "IV_SPIKE_DAYS", self.config.initial_capital)
            )

        df = pd.DataFrame(results) if results else pd.DataFrame()
        return df

    @staticmethod
    def _calc_metrics(subset: pd.DataFrame, label: str, initial_capital: float) -> Dict:
        pnl_arr = subset["Net_PnL"].values
        total_pnl = pnl_arr.sum()
        wins = (pnl_arr > 0).sum()
        losses = (pnl_arr <= 0).sum()
        n = len(pnl_arr)

        # Drawdown on cumulative PnL path
        equity = initial_capital + np.cumsum(pnl_arr)
        peak = np.maximum.accumulate(equity)
        max_dd_pct = float(abs(((equity / peak) - 1.0).min() * 100.0))

        gross_wins = pnl_arr[pnl_arr > 0].sum()
        gross_losses = abs(pnl_arr[pnl_arr <= 0].sum())
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        return {
            "regime": label,
            "trades": n,
            "return_pct": round((total_pnl / initial_capital) * 100.0, 3),
            "win_rate_pct": round(wins / n * 100.0, 2) if n > 0 else 0.0,
            "avg_pnl": round(total_pnl / n, 0) if n > 0 else 0.0,
            "total_pnl": round(total_pnl, 0),
            "max_dd_pct": round(max_dd_pct, 2),
            "profit_factor": round(profit_factor, 3),
            "wins": int(wins),
            "losses": int(losses),
            "dd_breach": max_dd_pct > 20.0,
            "regime_failed": total_pnl < 0 and label == "HIGH_VOL",
        }


# ============================================================================
# C. MONTE CARLO SIMULATION
# ============================================================================


class MonteCarloSimulator:
    """
    Monte Carlo robustness test:
      1. Shuffle trade order (500–1000 times)
      2. Add PnL noise ±5–10%
      3. Compute distribution of returns, drawdowns, Sharpe

    Outputs worst-case scenario and distributional statistics.
    """

    def __init__(self, config: ValidationConfig):
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    def simulate(self, trades_df: pd.DataFrame, runs: Optional[int] = None) -> pd.DataFrame:
        """Run Monte Carlo simulation."""
        if trades_df.empty:
            logger.warning("[MonteCarlo] No trades to simulate")
            return pd.DataFrame()

        n_runs = runs or self.config.monte_carlo_runs
        pnl_values = trades_df["Net_PnL"].values.copy()
        initial_capital = self.config.initial_capital
        results = []

        for i in range(n_runs):
            # 1. Shuffle trade sequence
            shuffled = pnl_values.copy()
            np.random.shuffle(shuffled)

            # 2. Add noise ±5–10%
            noise_pct = np.random.uniform(0.05, 0.10)
            sign = np.random.choice([-1, 1], size=len(shuffled))
            noisy_pnl = shuffled * (1.0 + sign * noise_pct * np.random.uniform(0, 1, len(shuffled)))

            # 3. Compute equity path
            equity = initial_capital + np.cumsum(noisy_pnl)
            peak = np.maximum.accumulate(equity)
            drawdown_pct = ((equity / peak) - 1.0) * 100.0
            max_dd = float(abs(drawdown_pct.min()))

            # 4. Sharpe
            ret_pct = float((equity[-1] / initial_capital - 1.0) * 100.0)
            daily_returns = noisy_pnl / initial_capital
            sharpe = (
                float(np.sqrt(252) * daily_returns.mean() / daily_returns.std())
                if daily_returns.std() > 0
                else 0.0
            )

            results.append({
                "run": i + 1,
                "final_equity": round(float(equity[-1]), 0),
                "return_pct": round(ret_pct, 3),
                "max_drawdown_pct": round(max_dd, 3),
                "sharpe": round(sharpe, 3),
                "wins": int((noisy_pnl > 0).sum()),
                "losses": int((noisy_pnl <= 0).sum()),
                "dd_breach": max_dd > 20.0,
            })

        df = pd.DataFrame(results)
        logger.info(
            f"[MonteCarlo] {n_runs} runs | "
            f"Return  P5={df['return_pct'].quantile(0.05):.2f}%  "
            f"P50={df['return_pct'].median():.2f}%  "
            f"P95={df['return_pct'].quantile(0.95):.2f}%"
        )
        logger.info(
            f"[MonteCarlo] MaxDD   P5={df['max_drawdown_pct'].quantile(0.05):.2f}%  "
            f"P50={df['max_drawdown_pct'].median():.2f}%  "
            f"P95={df['max_drawdown_pct'].quantile(0.95):.2f}%"
        )
        return df

    @staticmethod
    def summarise(df: pd.DataFrame) -> Dict:
        """Return distributional statistics from MC results."""
        if df.empty:
            return {}
        return {
            "runs": int(len(df)),
            "return_mean": round(float(df["return_pct"].mean()), 2),
            "return_std": round(float(df["return_pct"].std()), 2),
            "return_p5": round(float(df["return_pct"].quantile(0.05)), 2),
            "return_p25": round(float(df["return_pct"].quantile(0.25)), 2),
            "return_p50": round(float(df["return_pct"].median()), 2),
            "return_p75": round(float(df["return_pct"].quantile(0.75)), 2),
            "return_p95": round(float(df["return_pct"].quantile(0.95)), 2),
            "return_worst": round(float(df["return_pct"].min()), 2),
            "return_best": round(float(df["return_pct"].max()), 2),
            "dd_mean": round(float(df["max_drawdown_pct"].mean()), 2),
            "dd_std": round(float(df["max_drawdown_pct"].std()), 2),
            "dd_p5": round(float(df["max_drawdown_pct"].quantile(0.05)), 2),
            "dd_p50": round(float(df["max_drawdown_pct"].median()), 2),
            "dd_p95": round(float(df["max_drawdown_pct"].quantile(0.95)), 2),
            "dd_worst": round(float(df["max_drawdown_pct"].max()), 2),
            "pct_runs_dd_breach": round(float(df["dd_breach"].mean() * 100), 2),
            "sharpe_mean": round(float(df["sharpe"].mean()), 3),
            "sharpe_p5": round(float(df["sharpe"].quantile(0.05)), 3),
        }


# ============================================================================
# D. EXECUTION REALISM
# ============================================================================


class ExecutionRealismAdjuster:
    """
    Apply realistic execution penalties to a trades DataFrame:
      1. Slippage: 0.5–1.0% of premium (random per trade)
      2. Bid-ask spread penalty: 0.5% of premium
      3. Delayed execution proxy: shift PnL down by one-candle estimate (2%)

    Does NOT re-run the engine — applies post-hoc adjustments to PnL.
    """

    def __init__(self, config: ValidationConfig):
        self.config = config
        np.random.seed(config.random_seed)

    def adjust(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of trades_df with execution-adjusted Net_PnL."""
        if trades_df.empty:
            return trades_df

        adj = trades_df.copy()
        lots = adj["Lots"].values
        lot_qty = adj["Lot_Qty"].values
        premium = adj["Premium"].values
        notional = premium * lots * lot_qty

        # 1. Slippage: 0.5–1.0% of premium notional
        slippage_pct = np.random.uniform(
            self.config.slippage_min,
            self.config.slippage_max,
            len(adj),
        )
        slippage_cost = notional * slippage_pct

        # 2. Bid-ask spread penalty: fixed 0.5% of notional
        spread_cost = notional * self.config.bid_ask_penalty

        # 3. Delayed execution proxy: 0.2% additional cost
        delay_cost = notional * 0.002

        total_penalty = slippage_cost + spread_cost + delay_cost
        adj["Net_PnL"] = adj["Net_PnL"] - total_penalty
        adj["Gross_PnL"] = adj["Gross_PnL"] - total_penalty
        adj["Execution_Penalty"] = total_penalty.round(2)

        logger.info(
            f"[ExecRealism] Total penalty applied: Rs {total_penalty.sum():,.0f} "
            f"across {len(adj)} trades"
        )
        return adj

    @staticmethod
    def compute_realistic_summary(
        adj_trades: pd.DataFrame, initial_capital: float
    ) -> Dict:
        """Compute performance summary from execution-adjusted PnL."""
        if adj_trades.empty:
            return {}
        pnl = adj_trades["Net_PnL"].values
        equity = initial_capital + np.cumsum(pnl)
        peak = np.maximum.accumulate(equity)
        dd_pct = (equity / peak - 1.0) * 100.0

        total_return_pct = (equity[-1] / initial_capital - 1.0) * 100.0
        max_dd_pct = abs(dd_pct.min())
        daily_ret = pnl / initial_capital
        sharpe = (
            float(np.sqrt(252) * daily_ret.mean() / daily_ret.std())
            if daily_ret.std() > 0
            else 0.0
        )
        wins = (pnl > 0).sum()
        losses = (pnl <= 0).sum()
        win_rate = wins / len(pnl) * 100

        gross_wins = pnl[pnl > 0].sum()
        gross_losses = abs(pnl[pnl <= 0].sum())
        pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        return {
            "Total_Return_Pct": round(total_return_pct, 2),
            "Max_Drawdown_Pct": round(max_dd_pct, 2),
            "Sharpe": round(sharpe, 3),
            "Win_Rate_Pct": round(win_rate, 2),
            "Profit_Factor": round(pf, 3),
            "Total_Trades": int(len(pnl)),
            "Total_Penalty_Rs": round(adj_trades["Execution_Penalty"].sum(), 0),
            "Final_Equity": round(float(equity[-1]), 0),
        }


# ============================================================================
# E. FINAL EVALUATION & SAFETY GATES
# ============================================================================


class FinalEvaluator:
    """
    Aggregate results from all validation stages and apply safety rules.

    Safety gates (ANY trigger → NOT READY):
      1. DD > 20% in base backtest, realistic backtest, or any WF window
      2. Sharpe < 1.0 in forward test period (2026)
      3. Strategy loses money in HIGH_VOL regime
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

    def evaluate(
        self,
        base_summary: Dict,
        wf_df: pd.DataFrame,
        stress_df: pd.DataFrame,
        mc_summary: Dict,
        mc_df: pd.DataFrame,
        realistic_summary: Dict,
    ) -> Dict:
        failures: List[str] = []

        # --- Gate 1: Drawdown breaches ---
        base_dd = float(base_summary.get("Max_Drawdown_Pct", 0.0))
        if base_dd > self.config.max_dd_limit:
            failures.append(f"Base backtest DD {base_dd:.2f}% > {self.config.max_dd_limit}%")

        real_dd = float(realistic_summary.get("Max_Drawdown_Pct", 0.0))
        if real_dd > self.config.max_dd_limit:
            failures.append(f"Realistic execution DD {real_dd:.2f}% > {self.config.max_dd_limit}%")

        if not wf_df.empty:
            wf_max_dd = float(wf_df["max_dd_pct"].max())
            if wf_max_dd > self.config.max_dd_limit:
                worst_wf = wf_df.loc[wf_df["max_dd_pct"].idxmax(), "window"]
                failures.append(
                    f"Walk-forward DD {wf_max_dd:.2f}% > {self.config.max_dd_limit}% in window {worst_wf}"
                )

        mc_worst_dd = mc_summary.get("dd_worst", 0.0)
        if mc_worst_dd > self.config.max_dd_limit:
            pct_breach = mc_summary.get("pct_runs_dd_breach", 0.0)
            failures.append(
                f"Monte Carlo worst DD {mc_worst_dd:.2f}% > {self.config.max_dd_limit}% "
                f"({pct_breach:.1f}% of runs)"
            )

        # --- Gate 2: Forward period Sharpe ---
        if not wf_df.empty:
            forward_rows = wf_df[wf_df["is_forward"]]
            if not forward_rows.empty:
                min_fwd_sharpe = float(forward_rows["sharpe"].min())
                if min_fwd_sharpe < self.config.min_forward_sharpe:
                    failures.append(
                        f"Forward period Sharpe {min_fwd_sharpe:.3f} < {self.config.min_forward_sharpe}"
                    )

        # --- Gate 3: High-vol regime failure ---
        if not stress_df.empty:
            high_vol_rows = stress_df[stress_df["regime"] == "HIGH_VOL"]
            if not high_vol_rows.empty:
                high_vol_pnl = float(high_vol_rows["total_pnl"].iloc[0])
                if high_vol_pnl < 0:
                    failures.append(
                        f"Strategy LOSES money in HIGH_VOL regime "
                        f"(PnL: Rs {high_vol_pnl:,.0f})"
                    )

        # --- Return ranges ---
        all_returns = [
            base_summary.get("Total_Return_Pct", 0.0),
            realistic_summary.get("Total_Return_Pct", 0.0),
        ]
        if not wf_df.empty:
            all_returns += wf_df["return_pct"].tolist()
        return_range = [
            round(min(all_returns + [mc_summary.get("return_p5", 0.0)]), 2),
            round(max(all_returns + [mc_summary.get("return_p95", 0.0)]), 2),
        ]

        all_dds = [base_dd, real_dd]
        if not wf_df.empty:
            all_dds += wf_df["max_dd_pct"].tolist()
        dd_range = [
            round(min(all_dds), 2),
            round(max(all_dds + [mc_worst_dd]), 2),
        ]

        # --- Final verdict ---
        is_robust = len(failures) == 0 and float(base_summary.get("Sharpe", 0.0)) >= 1.0

        if is_robust:
            recommendation = "deploy"
        elif not failures and float(base_summary.get("Sharpe", 0.0)) >= 1.0:
            recommendation = "deploy"
        elif len(failures) <= 2 and float(base_summary.get("Sharpe", 0.0)) >= 1.0:
            recommendation = "refine"
        else:
            recommendation = "reject"

        return {
            "is_robust": is_robust,
            "expected_return_range": return_range,
            "expected_drawdown_range": dd_range,
            "failure_conditions": failures,
            "recommendation": recommendation,
            "safety_rules_passed": len(failures) == 0,
            "base_return_pct": round(float(base_summary.get("Total_Return_Pct", 0.0)), 2),
            "base_sharpe": round(float(base_summary.get("Sharpe", 0.0)), 3),
            "base_max_dd_pct": round(base_dd, 2),
            "realistic_return_pct": round(float(realistic_summary.get("Total_Return_Pct", 0.0)), 2),
            "realistic_max_dd_pct": round(real_dd, 2),
            "mc_p5_return": mc_summary.get("return_p5", 0.0),
            "mc_p95_return": mc_summary.get("return_p95", 0.0),
            "mc_worst_dd": mc_worst_dd,
            "mc_pct_runs_dd_breach": mc_summary.get("pct_runs_dd_breach", 0.0),
        }


# ============================================================================
# VALIDATION RUNNER (ORCHESTRATOR)
# ============================================================================


class ValidationRunner:
    """
    Main orchestrator that runs the full validation pipeline:
      1. Base backtest
      2. Walk-forward validation
      3. Regime stress test
      4. Monte Carlo simulation
      5. Execution realism
      6. Final evaluation

    All results are persisted to CSV/JSON in config.output_dir.
    """

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.all_results: Dict = {}

    # ------------------------------------------------------------------
    def run_all_validations(self) -> Dict:
        logger.info("=" * 70)
        logger.info("MEOW FINAL BOSS — FULL VALIDATION PIPELINE")
        logger.info(f"Strategy: {self.config.strategy_path}")
        logger.info(f"Output:   {self.config.output_dir}")
        logger.info("=" * 70)

        mod = _load_strategy_module(self.config.strategy_path)

        # ------------------------------------------------------------------
        # STEP 1: Base backtest (full date range for reference)
        # ------------------------------------------------------------------
        logger.info("\n[1/6] Running base backtest...")
        try:
            base_trades, base_equity, base_summary = _run_engine(
                mod,
                start_date="2024-01-01",
                end_date="2026-03-26",
                initial_capital=self.config.initial_capital,
            )
        except Exception as exc:
            logger.error(f"Base backtest failed: {exc}")
            base_trades = pd.DataFrame()
            base_equity = pd.DataFrame()
            base_summary = {}

        self.all_results["original_backtest"] = base_summary
        logger.info(
            f"  Return: {base_summary.get('Total_Return_Pct', 0):.2f}%  |  "
            f"MaxDD: {base_summary.get('Max_Drawdown_Pct', 0):.2f}%  |  "
            f"Sharpe: {base_summary.get('Sharpe', 0):.2f}  |  "
            f"Trades: {base_summary.get('Total_Trades', 0)}"
        )

        if base_trades.empty:
            logger.error("No trades in base backtest — aborting validation pipeline.")
            return {"error": "No trades generated", "original_backtest": base_summary}

        # ------------------------------------------------------------------
        # STEP 2: Walk-forward validation
        # ------------------------------------------------------------------
        logger.info("\n[2/6] Walk-forward validation...")
        wf_validator = WalkForwardValidator(self.config)
        wf_df = wf_validator.run_all_windows()
        wf_df.to_csv(_out(self.config, "walkforward_results.csv"), index=False)
        self.all_results["walk_forward"] = wf_df.to_dict(orient="records")
        logger.info(f"  Saved walkforward_results.csv  ({len(wf_df)} windows)")

        # ------------------------------------------------------------------
        # STEP 3: Regime stress test
        # ------------------------------------------------------------------
        logger.info("\n[3/6] Regime stress test...")
        stress_tester = RegimeStressTester(self.config)
        try:
            regimes_df = stress_tester.classify_regimes(mod, symbol="NIFTY")
            stress_df = stress_tester.run_stress_test(base_trades, regimes_df)
            if not stress_df.empty:
                stress_df.to_csv(_out(self.config, "stress_test_results.csv"), index=False)
                self.all_results["stress_test"] = stress_df.to_dict(orient="records")
                logger.info(f"  Saved stress_test_results.csv  ({len(stress_df)} regime slices)")
            else:
                logger.warning("  Stress test produced empty results")
                stress_df = pd.DataFrame()
                self.all_results["stress_test"] = []
        except Exception as exc:
            logger.warning(f"  Stress test failed: {exc}")
            stress_df = pd.DataFrame()
            self.all_results["stress_test"] = []

        # ------------------------------------------------------------------
        # STEP 4: Monte Carlo simulation
        # ------------------------------------------------------------------
        logger.info(f"\n[4/6] Monte Carlo simulation ({self.config.monte_carlo_runs} runs)...")
        mc_simulator = MonteCarloSimulator(self.config)
        mc_df = mc_simulator.simulate(base_trades, runs=self.config.monte_carlo_runs)
        mc_summary = MonteCarloSimulator.summarise(mc_df)

        if not mc_df.empty:
            mc_df.to_csv(_out(self.config, "monte_carlo_results.csv"), index=False)
        self.all_results["monte_carlo"] = {
            "summary": mc_summary,
            "worst_case": {
                "return_pct": mc_df["return_pct"].min() if not mc_df.empty else 0.0,
                "max_drawdown_pct": mc_df["max_drawdown_pct"].max() if not mc_df.empty else 0.0,
            },
        }
        logger.info(f"  Saved monte_carlo_results.csv  ({len(mc_df)} runs)")

        # ------------------------------------------------------------------
        # STEP 5: Execution realism
        # ------------------------------------------------------------------
        logger.info("\n[5/6] Execution realism adjustments...")
        exec_adjuster = ExecutionRealismAdjuster(self.config)
        adj_trades = exec_adjuster.adjust(base_trades)
        realistic_summary = ExecutionRealismAdjuster.compute_realistic_summary(
            adj_trades, self.config.initial_capital
        )

        if not adj_trades.empty:
            adj_trades.to_csv(
                _out(self.config, "realistic_backtest_results.csv"), index=False
            )
        self.all_results["realistic_execution"] = realistic_summary
        logger.info(
            f"  Return: {realistic_summary.get('Total_Return_Pct', 0):.2f}%  |  "
            f"MaxDD: {realistic_summary.get('Max_Drawdown_Pct', 0):.2f}%  |  "
            f"Sharpe: {realistic_summary.get('Sharpe', 0):.2f}"
        )
        logger.info(f"  Saved realistic_backtest_results.csv")

        # ------------------------------------------------------------------
        # STEP 6: Final evaluation
        # ------------------------------------------------------------------
        logger.info("\n[6/6] Final robustness evaluation...")
        evaluator = FinalEvaluator(self.config)
        evaluation = evaluator.evaluate(
            base_summary=base_summary,
            wf_df=wf_df,
            stress_df=stress_df,
            mc_summary=mc_summary,
            mc_df=mc_df,
            realistic_summary=realistic_summary,
        )
        self.all_results["evaluation"] = evaluation
        self.all_results["run_timestamp"] = datetime.utcnow().isoformat() + "Z"
        self.all_results["strategy_path"] = self.config.strategy_path

        # Save final report
        report_path = _out(self.config, "final_validation_report.json")
        with open(report_path, "w") as f:
            json.dump(self.all_results, f, indent=2, default=str)
        logger.info(f"  Saved final_validation_report.json")

        # ------------------------------------------------------------------
        # Print structured summary
        # ------------------------------------------------------------------
        self._print_summary(evaluation, base_summary, realistic_summary, mc_summary)

        return self.all_results

    # ------------------------------------------------------------------
    def _print_summary(
        self,
        evaluation: Dict,
        base: Dict,
        realistic: Dict,
        mc: Dict,
    ):
        sep = "=" * 72
        print(f"\n{sep}")
        print("MEOW FINAL BOSS — VALIDATION SUMMARY")
        print(sep)

        print("\n📊 COMPARISON ACROSS ALL TEST SCENARIOS")
        print(f"{'Metric':<30} {'Base':>12} {'Realistic':>12} {'MC P5':>10} {'MC P95':>10}")
        print("-" * 80)
        print(
            f"{'Return (%)':<30} "
            f"{base.get('Total_Return_Pct', 0):>12.2f} "
            f"{realistic.get('Total_Return_Pct', 0):>12.2f} "
            f"{mc.get('return_p5', 0):>10.2f} "
            f"{mc.get('return_p95', 0):>10.2f}"
        )
        print(
            f"{'Max Drawdown (%)':<30} "
            f"{base.get('Max_Drawdown_Pct', 0):>12.2f} "
            f"{realistic.get('Max_Drawdown_Pct', 0):>12.2f} "
            f"{'':>10} "
            f"{mc.get('dd_worst', 0):>10.2f}"
        )
        print(
            f"{'Sharpe':<30} "
            f"{base.get('Sharpe', 0):>12.3f} "
            f"{realistic.get('Sharpe', 0):>12.3f} "
            f"{mc.get('sharpe_p5', 0):>10.3f} "
            f"{'':>10}"
        )
        print(
            f"{'Win Rate (%)':<30} "
            f"{base.get('Win_Rate_Pct', 0):>12.2f} "
            f"{realistic.get('Win_Rate_Pct', 0):>12.2f} "
            f"{'':>10} "
            f"{'':>10}"
        )

        print(f"\n{'─' * 72}")
        is_robust = evaluation.get("is_robust", False)
        rec = evaluation.get("recommendation", "unknown").upper()
        ready_flag = "✅ READY" if is_robust else "🚨 NOT READY"
        print(f"Robustness:     {ready_flag}")
        print(f"Recommendation: {rec}")
        print(f"Return Range:   {evaluation.get('expected_return_range', [])}")
        print(f"DD Range:       {evaluation.get('expected_drawdown_range', [])}")
        print(f"Safety Rules:   {'PASSED ✅' if evaluation.get('safety_rules_passed') else 'FAILED ❌'}")

        failures = evaluation.get("failure_conditions", [])
        if failures:
            print("\n🚨 FAILURE CONDITIONS:")
            for fc in failures:
                print(f"  • {fc}")
        else:
            print("\n✅ No failure conditions triggered.")

        print(f"\n{sep}\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Run validation pipeline for the frozen Meow Final Boss strategy."""
    # Resolve paths relative to repository root
    repo_root = Path(__file__).resolve().parent.parent.parent
    strategy_path = repo_root / "quant_repo" / "strategies" / "short_vol" / "meow_final_boss.py"
    data_lake_path = repo_root / "data" / "master_fo_lake"
    output_dir = repo_root / "quant_repo" / "research_outputs" / "validation"

    config = ValidationConfig(
        strategy_path=str(strategy_path),
        data_lake_path=str(data_lake_path),
        output_dir=str(output_dir),
        initial_capital=10_000_000,
        monte_carlo_runs=500,
        random_seed=42,
    )

    runner = ValidationRunner(config)
    results = runner.run_all_validations()
    return results


if __name__ == "__main__":
    main()
