import subprocess
import re

with open("backtest_short_vol.py", "r") as f:
    content = f.read()

best_ret = -100
best_config = ""

for t in [2, 3]:
    for pt in [0.20, 0.30]:
        for gamma in [0]:
            target_str = "80%" if pt == 0.20 else "70%"

            new_content = re.sub(
                r"entry_idx = exp_idx - \d+", f"entry_idx = exp_idx - {t}", content
            )

            replacement = f"""                    # 2. Profit Take
                    best_val = ce_low + pe_low
                    target_val = pos["Premium"] * {pt}

                    if worst_val >= stop_val:
                        exit_signal = True
                        exit_reason = "Stop Loss (2x)"
                        buyback_cost = stop_val
                    elif best_val <= target_val:
                        exit_signal = True
                        exit_reason = "Profit Take ({target_str})"
                        buyback_cost = target_val
                    elif is_expiry:"""

            new_content = re.sub(
                r"                    # 2\. Profit Take \(Intraday Best\)\s+best_val = ce_low \+ pe_low\s+target_val = pos\[\"Premium\"\] \* [\d\.]+\s+if worst_val >= stop_val:\s+exit_signal = True\s+exit_reason = \"Stop Loss \(2x\)\"\s+buyback_cost = stop_val\s+elif best_val <= target_val:\s+exit_signal = True\s+exit_reason = \"Profit Take \(70%\)\"\s+buyback_cost = target_val\s+elif is_expiry:",
                replacement,
                new_content,
            )

            with open("bks2.py", "w") as f:
                f.write(new_content)

            res = subprocess.run(["python", "bks2.py"], capture_output=True, text=True)

            ret_match = re.search(r"Total Return:\s+([\d\.]+)%", res.stdout)
            if ret_match:
                ret = float(ret_match.group(1))
                print(f"T-{t} | TP: {target_str} -> {ret}%")
                if ret > best_ret:
                    best_ret = ret
                    best_config = f"Entry: T-{t} | TP: {target_str} | Return: {ret}%"

print(f"\nWINNER: {best_config}")
