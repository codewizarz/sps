#!/usr/bin/env python3
"""
Iteration runner for V3 Regime Adaptive Short Vol strategy.
Loads the V3 module, runs backtests with custom configs, evaluates metrics.
"""
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

V3_PATH = Path(__file__).with_name("v3_regime_adaptive_short_vol.py")
METRICS_DIR = Path(__file__).parent

V2_BENCHMARK = {
    "Total_Return_Pct": 39.10,
    "Max_Drawdown_Pct": 19.28,
    "Sharpe": 1.509,
    "Sortino": 0.961,
    "Profit_Factor": 1.59,
    "Win_Rate_Pct": 70.73,
    "Total_Trades": 41,
    "Return_DD": 2.028,
    "Final_Equity": 13909618.0,
    "Calmar": 2.028,
}

TARGET = {
    "Return": "25-35%",
    "Max_DD": "<12%",
    "Sharpe": ">1.4",
}


def load_v3():
    spec = importlib.util.spec_from_file_location("v3mod", V3_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_backtest(mod, config_override=None):
    config = mod.StrategyConfig(**(config_override or {}))
    engine = mod.BacktestEngine(config)
    trades_df, equity_df = engine.run(
        start_date="2025-04-01", end_date="2026-03-01"
    )
    if trades_df.empty:
        return trades_df, equity_df, {k: 0 for k in V2_BENCHMARK}
    equity_df_processed = mod.PerformanceTracker.enrich_equity_curve(equity_df)
    summary = mod.PerformanceTracker.compute_summary(
        trades_df, equity_df_processed, config.initial_capital
    )
    return trades_df, equity_df, summary


def compare_metrics(current, previous, v2=None):
    comparison = {}
    for key in current:
        if key in v2 or v2 is None:
            val = current[key]
            prev = previous.get(key, None) if previous else None
            v2_val = v2.get(key, None) if v2 else None
            comparison[key] = {
                "current": val,
                "vs_v2": round(val - v2_val, 4) if v2_val is not None else None,
                "vs_prev": round(val - prev, 4) if prev is not None else None,
            }
    return comparison


def evaluate_decision(current, v2):
    """Evaluate whether current run is improved vs V2."""
    # Weighted score: return is important, but we want to avoid DD > 12%
    ret = current.get("Total_Return_Pct", 0)
    dd = current.get("Max_Drawdown_Pct", 100)
    sharpe = current.get("Sharpe", 0)
    return_dd = current.get("Return_DD", 0)
    win_rate = current.get("Win_Rate_Pct", 0)

    # Penalty for exceeding target DD
    dd_penalty = max(0, dd - 12.0) * 3

    # Score: prioritize return/DD, then Sharpe, then raw return
    score = return_dd * 5 + sharpe * 3 + (ret / 100) * 2 - dd_penalty

    v2_ret = v2.get("Total_Return_Pct", 0)
    v2_dd = v2.get("Max_Drawdown_Pct", 100)
    v2_score = (v2_ret / v2_dd if v2_dd > 0 else 0) * 5 + v2.get("Sharpe", 0) * 3 + (v2_ret / 100) * 2

    # Check target achievement
    target_met = (
        25 <= ret <= 40
        and dd < 12.0
        and sharpe > 1.4
    )

    return {
        "score": round(score, 4),
        "v2_score": round(v2_score, 4),
        "target_met": target_met,
        "improved_vs_v2": score > v2_score,
        "dd_within_target": dd < 12.0,
    }


def save_iteration(trades_df, equity_df, metrics, iteration):
    trades_path = METRICS_DIR / f"trades_v3_iter{iteration}.csv"
    equity_path = METRICS_DIR / f"equity_v3_iter{iteration}.csv"
    metrics_path = METRICS_DIR / f"metrics_iter{iteration}.json"

    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)
    if not equity_df.empty:
        equity_df.to_csv(equity_path, index=False)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return trades_path, equity_path, metrics_path


def print_comparison(iteration, metrics, prev_metrics):
    prev_ret = prev_metrics.get("Total_Return_Pct", 0) if prev_metrics else 0
    prev_dd = prev_metrics.get("Max_Drawdown_Pct", 0) if prev_metrics else 0
    prev_sharpe = prev_metrics.get("Sharpe", 0) if prev_metrics else 0
    v2_ret = V2_BENCHMARK["Total_Return_Pct"]
    v2_dd = V2_BENCHMARK["Max_Drawdown_Pct"]
    v2_sharpe = V2_BENCHMARK["Sharpe"]

    ret = metrics["Total_Return_Pct"]
    dd = metrics["Max_Drawdown_Pct"]
    sharpe = metrics["Sharpe"]
    wr = metrics.get("Win_Rate_Pct", 0)
    pf = metrics.get("Profit_Factor", 0)
    ret_dd = metrics.get("Return_DD", 0)
    trades = metrics.get("Total_Trades", 0)

    ret_change_vs_prev = ret - prev_ret if prev_metrics else None
    dd_change_vs_prev = dd - prev_dd if prev_metrics else None
    sharpe_change_vs_prev = sharpe - prev_sharpe if prev_metrics else None

    decision = evaluate_decision(metrics, V2_BENCHMARK)
    target_met = decision["target_met"]
    score = decision["score"]

    print(f"\n{'='*70}")
    print(f"ITERATION {iteration}")
    print(f"{'='*70}")
    print(f"  Return:        {ret:6.2f}%  (vs V2: {ret - v2_ret:+6.2f}%)")
    print(f"  Max DD:        {dd:6.2f}%  (vs V2: {dd - v2_dd:+6.2f}%)")
    print(f"  Sharpe:        {sharpe:6.3f}   (vs V2: {sharpe - v2_sharpe:+6.3f})")
    print(f"  Return/DD:     {ret_dd:6.3f}x   (vs V2: {ret_dd - 2.028:+6.3f}x)")
    print(f"  Win Rate:      {wr:6.1f}%")
    print(f"  Profit Factor: {pf:6.2f}")
    print(f"  Trades:        {trades:4d}")
    print(f"  Score:         {score:6.4f}")
    print(f"  Target Met:    {target_met}")
    if ret_change_vs_prev is not None:
        print(f"  vs Prev:       Return {ret_change_vs_prev:+.2f}%, DD {dd_change_vs_prev:+.2f}%, Sharpe {sharpe_change_vs_prev:+.3f}")
    print(f"{'='*70}")
    return decision


def run_iteration(iteration, config_override, prev_metrics=None, description=""):
    print(f"\n>>> Running Iteration {iteration}: {description}")
    mod = load_v3()
    trades_df, equity_df, metrics = run_backtest(mod, config_override)
    decision = print_comparison(iteration, metrics, prev_metrics)
    paths = save_iteration(trades_df, equity_df, metrics, iteration)
    return metrics, decision, trades_df, equity_df


if __name__ == "__main__":
    # Quick test of baseline
    mod = load_v3()
    trades_df, equity_df, metrics = run_backtest(mod)
    print(json.dumps(metrics, indent=2))
