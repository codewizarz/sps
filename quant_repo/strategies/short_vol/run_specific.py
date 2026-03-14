import subprocess
import re

with open("backtest_short_vol.py", "r") as f:
    content = f.read()

# Modify Entry
new_content = re.sub(r"entry_idx = exp_idx - \d+", f"entry_idx = exp_idx - 4", content)

replacement = f"""                    # 2. Profit Take
                    best_val = ce_low + pe_low
                    target_val = pos["Premium"] * 0.15

                    # 3. Gamma Risk Avoidance
                    days_to_expiry = (expiry.date() - curr_date.date()).days

                    if worst_val >= stop_val:
                        exit_signal = True
                        exit_reason = "Stop Loss (2x)"
                        buyback_cost = stop_val
                    elif best_val <= target_val:
                        exit_signal = True
                        exit_reason = "Profit Take (85%)"
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
    print(f"Entry: T-4 | TP: 85% | Gamma Exit: T-1 | Return: {ret}% | Sharpe: {sharpe}")
else:
    print(res.stdout[-1000:])
