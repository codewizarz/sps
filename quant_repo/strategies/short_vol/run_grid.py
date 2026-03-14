import subprocess
import re
import sys
import os

with open("backtest_short_vol.py", "r") as f:
    content = f.read()

results = []

entry_windows = [2, 3, 4, 5]
profit_targets = [0.15, 0.20, 0.30, 0.40]  # 85%, 80%, 70%, 60% decay
gamma_exits = [0, 1]  # T-0 (expiry day), T-1

count = 1
total = len(entry_windows) * len(profit_targets) * len(gamma_exits)

for t in entry_windows:
    for pt in profit_targets:
        for gamma in gamma_exits:
            print(f"Running {count}/{total}: Entry T-{t}, Target {pt}, Gamma T-{gamma}")
            count += 1

            # 1. Modify Entry
            # entry_idx = exp_idx - X
            new_content = re.sub(
                r"entry_idx = exp_idx - \d+", f"entry_idx = exp_idx - {t}", content
            )

            if gamma == 1:
                # Add gamma avoidance logic
                if "days_to_expiry <= 1" not in new_content:
                    replacement = f"""                    # 2. Profit Take
                    best_val = ce_low + pe_low
                    target_val = pos["Premium"] * {pt}

                    # 3. Gamma Risk Avoidance
                    days_to_expiry = (expiry.date() - curr_date.date()).days

                    if worst_val >= stop_val:
                        exit_signal = True
                        exit_reason = "Stop Loss (2x)"
                        buyback_cost = stop_val
                    elif best_val <= target_val:
                        exit_signal = True
                        exit_reason = f"Profit Take ({int((1 - pt) * 100)}%)"
                        buyback_cost = target_val
                    elif days_to_expiry <= 1:
                        exit_signal = True
                        exit_reason = "Gamma Avoidance (T-1)"
                        buyback_cost = ce_close + pe_close
                    elif is_expiry:"""

                    new_content = re.sub(
                        r"                    # 2\. Profit Take \(Intraday Best\)\s+best_val = ce_low \+ pe_low\s+target_val = pos\[\"Premium\"\] \* [\d\.]+\s+if worst_val >= stop_val:\s+exit_signal = True\s+exit_reason = \"Stop Loss \(2x\)\"\s+buyback_cost = stop_val\s+elif best_val <= target_val:\s+exit_signal = True\s+exit_reason = \"Profit Take \(70%\)\"\s+buyback_cost = target_val\s+elif is_expiry:",
                        replacement,
                        new_content,
                    )
            else:
                # Standard expiry logic
                replacement = f"""                    # 2. Profit Take
                    best_val = ce_low + pe_low
                    target_val = pos["Premium"] * {pt}

                    if worst_val >= stop_val:
                        exit_signal = True
                        exit_reason = "Stop Loss (2x)"
                        buyback_cost = stop_val
                    elif best_val <= target_val:
                        exit_signal = True
                        exit_reason = f"Profit Take ({int((1 - pt) * 100)}%)"
                        buyback_cost = target_val
                    elif is_expiry:"""

                new_content = re.sub(
                    r"                    # 2\. Profit Take \(Intraday Best\)\s+best_val = ce_low \+ pe_low\s+target_val = pos\[\"Premium\"\] \* [\d\.]+\s+if worst_val >= stop_val:\s+exit_signal = True\s+exit_reason = \"Stop Loss \(2x\)\"\s+buyback_cost = stop_val\s+elif best_val <= target_val:\s+exit_signal = True\s+exit_reason = \"Profit Take \(70%\)\"\s+buyback_cost = target_val\s+elif is_expiry:",
                    replacement,
                    new_content,
                )

            with open("backtest_short_vol_temp.py", "w") as f:
                f.write(new_content)

            res = subprocess.run(
                ["python", "backtest_short_vol_temp.py"], capture_output=True, text=True
            )

            # Parse output
            ret_match = re.search(r"Total Return:\s+([\d\.]+)%", res.stdout)
            sharpe_match = re.search(r"Sharpe:\s+([\d\.]+)", res.stdout)

            if ret_match and sharpe_match:
                ret = float(ret_match.group(1))
                sharpe = float(sharpe_match.group(1))
                results.append(
                    {
                        "entry": t,
                        "target": (1 - pt) * 100,
                        "gamma": gamma,
                        "return": ret,
                        "sharpe": sharpe,
                    }
                )
                print(f"  -> Ret: {ret}%, Sharpe: {sharpe}")
            else:
                print("  -> Failed to parse output")


# Sort by return
results.sort(key=lambda x: x["return"], reverse=True)

print("\n=== TOP 5 CONFIGURATIONS BY TOTAL RETURN ===")
for r in results[:5]:
    print(
        f"Entry: T-{r['entry']} | TP: {int(r['target'])}% | Gamma Exit: T-{r['gamma']} | Return: {r['return']}% | Sharpe: {r['sharpe']}"
    )

if os.path.exists("backtest_short_vol_temp.py"):
    os.remove("backtest_short_vol_temp.py")
