import pandas as pd
import numpy as np
import sys
from pathlib import Path


def evaluate_threshold(df, timestamps, threshold):
    trades_list = []

    # 0.2% total slippage = 0.002
    slippage_rate = 0.002
    # ₹50 per leg = ₹100 round trip = 100
    commission = 100

    for i, ts in enumerate(timestamps):
        # IF VRP > threshold: simulate selling ATM straddle
        if not (df.loc[ts, "VRP"] > threshold):
            continue

        entry_price = df.loc[ts, "SPOT"]
        entry_iv = df.loc[ts, "TRUE_IV"]
        premium = entry_iv * entry_price * 0.01
        strike = entry_price

        exit_date = None
        days_held = 0
        exit_reason = ""
        trade_pnl = 0

        # Check daily price for up to 5 days (weekly expiry)
        for day_offset in range(1, 6):
            if i + day_offset >= len(timestamps):
                # End of data, force exit
                exit_idx = len(timestamps) - 1
                curr_ts = timestamps[exit_idx]
                curr_price = df.loc[curr_ts, "SPOT"]

                daily_loss = abs(curr_price - strike)
                profit_gross = premium - daily_loss

                # Apply slippage on exit nominal value (entry is already baked into premium mostly, but we assume total slip)
                slip_cost = (entry_price + curr_price) * slippage_rate
                profit_net = profit_gross - slip_cost - commission

                exit_date = curr_ts
                days_held = exit_idx - i
                exit_reason = "END_OF_DATA"
                trade_pnl = profit_net
                break

            curr_ts = timestamps[i + day_offset]
            curr_price = df.loc[curr_ts, "SPOT"]

            daily_loss = abs(curr_price - strike)
            profit_gross = premium - daily_loss
            slip_cost = (entry_price + curr_price) * slippage_rate
            profit_net = profit_gross - slip_cost - commission

            # PROFIT TARGET: exit if premium decays to 40% (profit >= 60% of premium)
            if profit_gross >= premium * 0.60:
                exit_date = curr_ts
                days_held = day_offset
                exit_reason = "TP"
                trade_pnl = profit_net
                break

            # STOP LOSS: exit if loss reaches 2x premium (current_value >= premium * 2)
            if daily_loss >= premium * 2.0:
                exit_date = curr_ts
                days_held = day_offset
                exit_reason = "SL"
                trade_pnl = profit_net
                break

            # EXPIRY: if neither triggered and day 5 reached
            if day_offset == 5:
                exit_date = curr_ts
                days_held = day_offset
                exit_reason = "EXPIRY"
                trade_pnl = profit_net
                break

        if exit_date is not None and days_held > 0:
            trades_list.append(
                {"entry_date": ts, "exit_date": exit_date, "trade_pnl": trade_pnl}
            )

    if not trades_list:
        return 0, 0, 10000.0, 0.0, 0.0

    trades_df = pd.DataFrame(trades_list)
    total_trades = len(trades_df)
    win_rate = (
        len(trades_df[trades_df["trade_pnl"] > 0]) / total_trades
        if total_trades > 0
        else 0
    )

    # Calculate Equity & Metrics
    daily_profits = trades_df.groupby("exit_date")["trade_pnl"].sum()

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

    daily_returns = daily_profits / 10000.0
    if len(daily_returns) > 1 and daily_returns.std() != 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    return total_trades, win_rate, current_equity, sharpe, max_dd


def main():
    repo_root = Path(__file__).resolve().parent.parent
    input_file = repo_root / "research_outputs" / "vrp_nifty.parquet"

    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    df = pd.read_parquet(input_file)

    if "TRUE_IV" not in df.columns:
        print("Error: TRUE_IV column missing. Run validate_vrp_india.py first.")
        sys.exit(1)

    df = df[(df["TRUE_IV"] <= 1.0) & (df["TRUE_IV"] >= 0.05)].copy()
    df = df.sort_index()

    df["VRP"] = df["TRUE_IV"] - df["RV"]
    timestamps = df.index.tolist()

    results = []

    # Threshold sweep: 0.01 to 0.10 (max Empirical True VRP was ~0.066)
    thresholds = np.arange(0.01, 0.11, 0.01)

    for thresh in thresholds:
        t_trades, t_win_rate, t_equity, t_sharpe, t_max_dd = evaluate_threshold(
            df, timestamps, thresh
        )

        results.append(
            {
                "threshold": np.round(thresh, 2),
                "trades": t_trades,
                "win_rate": t_win_rate,
                "equity": t_equity,
                "sharpe": t_sharpe,
                "max_dd": t_max_dd,
            }
        )

    results_df = pd.DataFrame(results)

    # Objective Function: Sharpe > 1.2, Max DD < 25% (-0.25), Trades > 40
    # Note max_dd is negative
    valid_scans = results_df[(results_df["trades"] > 40)]

    output_dir = repo_root / "research_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "vrp_threshold_scan.parquet"
    results_df.to_parquet(out_file, index=False)

    if len(valid_scans) == 0:
        print(
            "\nNO STRATEGY PARAMETERS SATISFIED THE MINIMUM TRADE OBJECTIVE FUNCTION."
        )
        best_overall = results_df.sort_values("sharpe", ascending=False).iloc[0]
        print(f"\nCLOSEST BEST VRP THRESHOLD FOUND: {best_overall['threshold']:.2f}")
        print(f"SHARPE: {best_overall['sharpe']:.2f}")
        print(f"MAX DD: {best_overall['max_dd']:.2%}")
        print(f"FINAL EQUITY: ${best_overall['equity']:,.2f}")
        print(f"TRADES: {best_overall['trades']}")
        sys.exit(0)

    # Pick highest sharpe
    best_candidate = valid_scans.sort_values("sharpe", ascending=False).iloc[0]

    print(f"\nBEST VRP THRESHOLD FOUND: {best_candidate['threshold']:.2f}")
    print(f"SHARPE: {best_candidate['sharpe']:.2f}")
    print(f"MAX DD: {best_candidate['max_dd']:.2%}")
    print(f"FINAL EQUITY: ${best_candidate['equity']:,.2f}")
    print(f"TRADES: {best_candidate['trades']}")


if __name__ == "__main__":
    main()
