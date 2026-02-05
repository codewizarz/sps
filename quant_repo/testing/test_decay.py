import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
from quant_repo.analytics.decay import EdgeSentinel


def test_decay():
    print("[TEST] Edge Decay Monitor...")

    sentinel = EdgeSentinel()

    # 1. Historical Data (Baseline)
    # Profitable: Mean 50, Std 100
    np.random.seed(42)
    hist_pnl = np.random.normal(50, 100, 1000)

    df_hist = pl.DataFrame({"pnl": hist_pnl, "regime": ["NORMAL"] * 1000})

    # 2. Check Healthy Recent Data
    # Similar distribution
    # Using same mean/std, different seed
    recent_healthy = np.random.normal(
        50, 100, 100
    )  # Changed mean to 50 to match hist exactly
    df_healthy = pl.DataFrame({"pnl": recent_healthy, "regime": ["NORMAL"] * 100})

    report_healthy = sentinel.check_decay(df_healthy, df_hist)
    print(f"\nScenario: Healthy")
    print(f"Decaying: {report_healthy.is_decaying}")
    print(f"KS p-value: {report_healthy.p_value_ks:.4f}")

    assert not report_healthy.is_decaying
    assert report_healthy.p_value_ks > 0.05

    # 3. Check Broken Strategy (Win Rate Drop)
    # Mean -20, Std 100
    recent_broken = np.random.normal(-20, 100, 100)
    df_broken = pl.DataFrame({"pnl": recent_broken, "regime": ["NORMAL"] * 100})

    report_broken = sentinel.check_decay(df_broken, df_hist)
    print(f"\nScenario: Broken (Win Rate Drop)")
    print(f"Decaying: {report_broken.is_decaying}")
    print(f"Z-Score: {report_broken.z_score_win_rate:.2f}")
    print(f"Desc: {report_broken.description}")

    assert report_broken.is_decaying
    assert "Win Rate Deterioration" in report_broken.description

    # 4. Check Distribution Shift (Fat Left Tail)
    # Mixed Gaussian: 80% Normal, 20% Crash (-1000)
    recent_crash = np.concatenate(
        [np.random.normal(50, 100, 80), np.random.normal(-1000, 50, 20)]
    )
    df_crash = pl.DataFrame({"pnl": recent_crash, "regime": ["NORMAL"] * 100})

    report_crash = sentinel.check_decay(df_crash, df_hist)
    print(f"\nScenario: Fat Tail Shift")
    print(f"Decaying: {report_crash.is_decaying}")
    print(f"KS p-value: {report_crash.p_value_ks:.4f}")

    assert report_crash.is_decaying
    assert "Distribution Shift" in report_crash.description

    print("[TEST] SUCCESS")


if __name__ == "__main__":
    test_decay()
