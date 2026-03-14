import pandas as pd
import numpy as np
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    input_file = repo_root / "research_outputs" / "vrp_nifty.parquet"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Please run validate_vrp_india.py first.")
        sys.exit(1)

    print(f"Loading research outputs from: {input_file}")
    df = pd.read_parquet(input_file)

    initial_rows = len(df)

    # Clean data (remove rows where IV > 1.0 OR IV < 0.05)
    df = df[(df["IV"] <= 1.0) & (df["IV"] >= 0.05)].copy()

    cleaned_rows = len(df)
    print(f"Removed {initial_rows - cleaned_rows} rows with bad IV estimates.")
    print(f"Data points available for backtest: {cleaned_rows}")

    if len(df) == 0:
        print("Error: No data left after cleaning.")
        sys.exit(1)

    df = df.sort_index()

    # Pre-calculate signal
    df["Signal"] = df["IV"] > (df["RV"] * 1.2)

    # Convert the index to a list for iterating over time series easily
    timestamps = df.index.tolist()
    trades_list = []

    for i, ts in enumerate(timestamps):
        if not df.loc[ts, "Signal"]:
            continue

        entry_price = df.loc[ts, "SPOT"]
        entry_iv = df.loc[ts, "IV"]
        premium = entry_iv * entry_price * 0.01
        strike = entry_price

        exit_date = None
        days_held = 0
        exit_reason = ""
        trade_pnl = 0

        # Check daily price for up to 5 days (weekly expiry)
        for day_offset in range(1, 6):
            if i + day_offset >= len(timestamps):
                # Not enough data to reach expiry, force exit on last available day
                exit_idx = len(timestamps) - 1
                curr_ts = timestamps[exit_idx]
                curr_price = df.loc[curr_ts, "SPOT"]

                daily_loss = abs(curr_price - strike)
                current_value = daily_loss
                profit = premium - current_value

                exit_date = curr_ts
                days_held = exit_idx - i
                exit_reason = "END_OF_DATA"
                trade_pnl = profit
                break

            curr_ts = timestamps[i + day_offset]
            curr_price = df.loc[curr_ts, "SPOT"]

            daily_loss = abs(curr_price - strike)
            current_value = daily_loss
            profit = premium - current_value

            # 1) Take profit if Profit >= premium * 0.5
            if profit >= premium * 0.5:
                exit_date = curr_ts
                days_held = day_offset
                exit_reason = "TP"
                trade_pnl = profit
                break

            # 2) Stop loss if current_value >= premium * 2
            if current_value >= premium * 2:
                exit_date = curr_ts
                days_held = day_offset
                exit_reason = "SL"
                trade_pnl = profit
                break

            # 4) If neither triggered and expiry reached (day 5), exit at final intrinsic value
            if day_offset == 5:
                exit_date = curr_ts
                days_held = day_offset
                exit_reason = "EXPIRY"
                trade_pnl = profit
                break

        # Only log trades that actually transacted
        if exit_date is not None and days_held > 0:
            trades_list.append(
                {
                    "entry_date": ts,
                    "exit_date": exit_date,
                    "days_held": days_held,
                    "exit_reason": exit_reason,
                    "trade_pnl": trade_pnl,
                }
            )

    trades_df = pd.DataFrame(trades_list)

    if len(trades_df) == 0:
        print("Warning: No trades triggered or resolved. IV was never > RV * 1.2")
        sys.exit(0)

    # -----------------------
    # Evaluate Metrics
    # -----------------------
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df["trade_pnl"] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    worst_trade = trades_df["trade_pnl"].min()
    avg_hold_days = trades_df["days_held"].mean()

    # Track equity curve assuming a starting capital of $10,000
    # Group profit by exit_date for daily returns distribution
    daily_profits = trades_df.groupby("exit_date")["trade_pnl"].sum()

    # Complete daily equity series over trading days that had an exit
    current_equity = 10000.0
    equity_series = []

    for _, pnl in daily_profits.items():
        current_equity += pnl
        equity_series.append(current_equity)

    daily_profits_df = pd.DataFrame(
        {"Equity": equity_series}, index=daily_profits.index
    )

    peak = daily_profits_df["Equity"].cummax()
    drawdown = (daily_profits_df["Equity"] - peak) / peak
    max_dd = drawdown.min()

    final_equity = current_equity

    # Compute Sharpe Ratio (Annualized proxy based on daily exiting capital)
    daily_returns = daily_profits / 10000.0  # Return on $10k base
    if len(daily_returns) > 1 and daily_returns.std() != 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # -----------------------
    # Output and Logging
    # -----------------------
    output_dir = repo_root / "research_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "vrp_backtest_with_exits.parquet"

    try:
        trades_df.to_parquet(out_file, index=False)
        print(f"\nSaved trade log output to: {out_file}")
    except Exception as e:
        print(f"Error saving output dataframe: {e}")

    # Print metrics
    print("\n====== BACKTEST RESULTS ======")
    print(f"Total trades      : {total_trades}")
    print(f"Win %             : {win_rate:.2%}")
    print(f"Final equity      : ${final_equity:,.2f} (starting: $10,000.00)")
    print(f"Sharpe Ratio      : {sharpe:.2f}")
    print(f"Max DD            : {max_dd:.2%}")
    print(f"Average hold days : {avg_hold_days:.2f}")
    print(f"Worst trade       : ${worst_trade:,.2f}")


if __name__ == "__main__":
    main()
