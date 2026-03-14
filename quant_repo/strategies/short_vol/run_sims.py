import subprocess
import re
import sys

with open("backtest_short_vol.py", "r") as f:
    content = f.read()

results = {}
for i in [2, 3, 4, 5]:
    new_content = re.sub(
        r"entry_idx = exp_idx - \d+", f"entry_idx = exp_idx - {i}", content
    )
    with open("backtest_short_vol_temp.py", "w") as f:
        f.write(new_content)

    res = subprocess.run(
        ["python", "backtest_short_vol_temp.py"], capture_output=True, text=True
    )

    match = re.search(r"Sharpe Ratio\s+\|\s+([\d\.]+)", res.stdout)
    dd_match = re.search(r"Max Drawdown\s+\|\s+(-\d+\.\d+%)", res.stdout)
    if match and dd_match:
        sharpe = float(match.group(1))
        dd = dd_match.group(1)
        results[f"T-{i}"] = {"sharpe": sharpe, "dd": dd}
        print(f"T-{i}: Sharpe {sharpe} | DD {dd}")
    else:
        print(f"T-{i}: Failed to parse")
        # print(res.stdout[-1000:])

print("\nBest Configuration:")
best = max(results.keys(), key=lambda k: results[k]["sharpe"])
print(f"Best: {best} -> {results[best]}")
