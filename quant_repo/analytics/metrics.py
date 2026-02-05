import polars as pl
import numpy as np
from typing import Dict


def calc_expectancy(df: pl.DataFrame, pnl_col: str = "pnl_net") -> Dict[str, float]:
    """
    Calculates Expectancy and Win Rate Stats.
    """
    n_trades = len(df)
    if n_trades == 0:
        return {"expectancy": 0.0, "win_rate": 0.0, "profit_factor": 0.0}

    winners = df.filter(pl.col(pnl_col) > 0)
    losers = df.filter(pl.col(pnl_col) <= 0)

    win_rate = len(winners) / n_trades
    avg_win = winners[pnl_col].mean() if len(winners) > 0 else 0.0
    avg_loss = abs(losers[pnl_col].mean()) if len(losers) > 0 else 0.0

    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Profit Factor
    gross_loss = abs(losers[pnl_col].sum())
    profit_factor = (
        (winners[pnl_col].sum() / gross_loss) if gross_loss > 0 else float("inf")
    )

    return {
        "expectancy": expectancy,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sqn": (expectancy / df[pnl_col].std()) * np.sqrt(n_trades)
        if df[pnl_col].std() > 0
        else 0.0,
    }


def calc_tail_risk(
    df: pl.DataFrame, pnl_col: str = "pnl_net", confidence: float = 0.95
) -> Dict[str, float]:
    """
    Calculates VaR and CVaR (Expected Shortfall).
    """
    if len(df) == 0:
        return {"var_95": 0.0, "cvar_95": 0.0}

    # Historical method
    sorted_pnl = df[pnl_col].sort()

    # Percentile index
    idx = int((1 - confidence) * len(df))
    var = sorted_pnl[idx]

    # CVaR: Mean of losses exceeding VaR
    tail_losses = sorted_pnl.filter(sorted_pnl <= var)
    cvar = tail_losses.mean() if len(tail_losses) > 0 else var

    return {f"var_{int(confidence * 100)}": var, f"cvar_{int(confidence * 100)}": cvar}


def calc_edge_decay(
    df: pl.DataFrame, pnl_col: str = "pnl_net", window: int = 50
) -> pl.DataFrame:
    """
    Calculates rolling average PnL to detect decay.
    """
    # Simply rolling mean for now.
    # Linear regression slope is better but expensive in rolling without generic UDF.
    # Polars has rolling_mean.
    return df.with_columns(
        [pl.col(pnl_col).rolling_mean(window).alias("rolling_avg_pnl")]
    )
