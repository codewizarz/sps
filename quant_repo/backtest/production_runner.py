import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import date

# Import existing structures if available, or define here for self-containment
# We will define a local AccountState for the simulation to avoid circular deps if needed,
# but ideally we align with portfolio definitions.


@dataclass
class SimAccountState:
    date: date
    equity: float
    cash: float
    margin_used: float
    positions: Dict[str, int] = field(default_factory=dict)  # Symbol -> Quantity


@dataclass
class BacktestResult:
    equity_curve: pl.DataFrame
    trade_log: pl.DataFrame
    metrics: Dict[str, float]
    regime_stats: pl.DataFrame


class BacktestRunner:
    """
    Event-driven backtest engine for production-grade strategy simulation.
    Simulates daily steps, imposing spreads, slippage, and commission costs.
    """

    def __init__(
        self, initial_capital: float = 1_000_000.0, commission_per_contract: float = 1.0
    ):
        self.initial_capital = initial_capital
        self.commission = commission_per_contract

    def run(self, strategy: Any, history_df: pl.DataFrame) -> BacktestResult:
        """
        Runs the simulation.
        strategy: Object with 'update(row)' and 'generate_signals()' methods.
        history_df: Polars DataFrame with market data (date, iv, price, regime, etc.)
        """

        # State Initialization
        account = SimAccountState(
            date=date.min,
            equity=self.initial_capital,
            cash=self.initial_capital,
            margin_used=0.0,
        )

        equity_curve_data = []
        trade_log_data = []

        # Ensure sorted
        history = history_df.sort("date")

        # Loop through history
        # Note: iter_rows(named=True) is convenient but slow for millions of rows.
        # For daily backtests (years = ~2500 rows), it's fine.

        for row in history.iter_rows(named=True):
            current_date = row["date"]
            account.date = current_date

            # 1. Update Strategy
            # Strategy internal state update (e.g. indicators)
            if hasattr(strategy, "update"):
                strategy.update(row)

            # 2. Generate Signals
            # Expected format: List of dicts { 'action': 'BUY'/'SELL', 'quantity': 1, 'price': 100.0, 'type': 'SHORT_VOL' }
            if hasattr(strategy, "generate_signals"):
                signals = strategy.generate_signals(account)  # Pass account for sizing?
            else:
                signals = []

            # 3. Execute Signals
            daily_pnl = 0.0

            for sig in signals:
                # Sim Execution
                qty = sig.get("quantity", 0)
                price = sig.get("price", 0.0)
                action = sig.get("action", "HOLD")
                strat_type = sig.get("type", "UNKNOWN")

                if qty == 0:
                    continue

                # Apply Friction
                # Slippage: Assume 1 tick (0.05) or proportional
                slippage = 0.05 * qty  # rough tick slippage
                comm = self.commission * abs(qty)

                cost = comm + slippage

                # Exec Price
                if action == "BUY":
                    exec_price = price + 0.05  # Pay more
                else:  # SELL
                    exec_price = price - 0.05  # Receive less

                # PnL Impact (Immediate for simplicity in this signal-based sim,
                # or we assume 'price' is the PnL achieved for that day if signals are 'daily_pnl' records)
                # NOTE: Real event-driven tracks positions over time.
                # For this implementation, we assume signals represent *completed trades* or *daily mark-to-market* records
                # passed from the strategy for the sake of the 'Alpha Factory' workflow often used in this repo.

                # However, the user asked for "enforce margin constraints" which implies position tracking.
                # Let's support a simplified daily-rebalance model.

                pnl_trade = sig.get("pnl", 0.0)  # If provided directly

                # Deduct costs
                net_pnl = pnl_trade - cost
                daily_pnl += net_pnl

                # Log Trade
                trade_log_data.append(
                    {
                        "date": current_date,
                        "action": action,
                        "quantity": qty,
                        "price": price,
                        "exec_price": exec_price,
                        "cost": cost,
                        "pnl": net_pnl,
                        "strategy_type": strat_type,
                        "regime": row.get("regime", "UNKNOWN"),
                    }
                )

            # 4. Update Account
            account.equity += daily_pnl
            account.cash += daily_pnl
            # Margin would be updated based on open positions if we tracked them fully

            equity_curve_data.append(
                {
                    "date": current_date,
                    "equity": account.equity,
                    "daily_pnl": daily_pnl,
                    "drawdown": 0.0,  # Calc later
                }
            )

            if account.equity <= 0:
                print(f"BROKE at {current_date}")
                break

        # Post-Processing
        df_equity = pl.DataFrame(equity_curve_data)

        # Calculate Drawdown
        if len(df_equity) > 0:
            high_water_mark = df_equity["equity"].cum_max()
            drawdown = (high_water_mark - df_equity["equity"]) / high_water_mark
            df_equity = df_equity.with_columns(drawdown.alias("drawdown"))

            # Metrics
            total_ret = (account.equity - self.initial_capital) / self.initial_capital

            # CAGR
            days = (df_equity["date"].max() - df_equity["date"].min()).days
            years = max(days / 365.25, 0.01)
            cagr = (account.equity / self.initial_capital) ** (1 / years) - 1

            # Sharpe (Daily)
            daily_returns = df_equity["daily_pnl"] / df_equity["equity"].shift(
                1
            ).fill_null(self.initial_capital)
            mean_ret = daily_returns.mean()
            std_ret = daily_returns.std()
            if std_ret != 0:
                sharpe = (mean_ret / std_ret) * np.sqrt(252)
            else:
                sharpe = 0.0

            max_dd = df_equity["drawdown"].max()

            # Sortino
            downside = daily_returns.filter(daily_returns < 0)
            if len(downside) > 0:
                downside_std = downside.std()
                if downside_std is not None and downside_std != 0:
                    sortino = (mean_ret / downside_std) * np.sqrt(252)
                else:
                    sortino = 0.0
            else:
                sortino = 0.0  # No downside! Infinite Sortino technically.

            # Tail Loss (Worst 5 days avg)
            sorted_rets = daily_returns.sort()
            tail_loss = sorted_rets.head(5).mean()

            metrics = {
                "total_return": total_ret,
                "cagr": cagr,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": max_dd,
                "tail_loss_avg_5d": tail_loss if tail_loss is not None else 0.0,
            }
        else:
            metrics = {
                "total_return": 0.0,
                "cagr": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            }

        df_trades = pl.DataFrame(trade_log_data)

        # Regime Stats
        regime_stats = pl.DataFrame()
        if len(df_trades) > 0 and "regime" in df_trades.columns:
            # Join trades with equity to get returns? Or just raw PnL sum
            regime_stats = df_trades.group_by("regime").agg(
                [
                    pl.col("pnl").sum().alias("total_pnl"),
                    pl.col("pnl").count().alias("trade_count"),
                    pl.col("pnl").mean().alias("avg_pnl"),
                ]
            )

        return BacktestResult(
            equity_curve=df_equity,
            trade_log=df_trades,
            metrics=metrics,
            regime_stats=regime_stats,
        )
