import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import shutil
from quant_repo.analytics.engine import AnalyticsEngine


def test_analytics():
    print("[TEST] Analytics Layer...")

    # 1. Mock Trade Log
    # 100 Trades
    # Signal A: Consistent Winner (Vol Arb)
    # Signal B: Consistent Loser (Random)

    rows = []

    # Signal A: 60% win rate, Avg Win 200, Avg Loss 100
    for i in range(50):
        # Regime Toggle
        regime = "HIGH_VOL" if i < 25 else "LOW_VOL"

        if i % 10 < 6:  # Win
            pnl = 200.0 + np.random.normal(0, 10)
        else:
            pnl = -100.0 + np.random.normal(0, 10)

        rows.append(
            {
                "trade_id": f"A-{i}",
                "signal_type": "VOL_ARB",
                "regime": regime,
                "pnl_net": pnl,
                "exit_time": i,  # Mock time
            }
        )

    # Signal B: 40% win rate, Avg Win 100, Avg Loss 100
    for i in range(50):
        if i % 10 < 4:  # Win
            pnl = 100.0
        else:
            pnl = -100.0

        rows.append(
            {
                "trade_id": f"B-{i}",
                "signal_type": "NOISE",
                "regime": "LOW_VOL",
                "pnl_net": pnl,
                "exit_time": 100 + i,
            }
        )

    df = pl.DataFrame(rows)

    temp_dir = Path.cwd() / "temp_analytics"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    log_path = temp_dir / "trade_log.parquet"
    df.write_parquet(log_path)

    # 2. Run Engine
    engine = AnalyticsEngine()
    engine.analyze(str(log_path), str(temp_dir))

    # 3. Verify Outputs
    attr_sig_path = temp_dir / "attribution_by_signal.parquet"
    assert attr_sig_path.exists()

    stats_sig = pl.read_parquet(attr_sig_path)

    # Verify Vol Arb is better
    vol_arb = stats_sig.filter(pl.col("signal_type") == "VOL_ARB")
    noise = stats_sig.filter(pl.col("signal_type") == "NOISE")

    print("\n[TEST] Validation...")
    print(f"  Vol Arb SQN: {vol_arb['sqn'][0]:.2f} (Exp: High positive)")
    print(f"  Noise SQN: {noise['sqn'][0]:.2f} (Exp: Negative or Low)")

    assert vol_arb["expectancy"][0] > 0
    assert vol_arb["expectancy"][0] > noise["expectancy"][0]

    # Verify Regime
    attr_reg_path = temp_dir / "attribution_by_regime.parquet"
    assert attr_reg_path.exists()
    stats_reg = pl.read_parquet(attr_reg_path)
    assert len(stats_reg) >= 2  # HIGH_VOL, LOW_VOL

    print("[TEST] SUCCESS")
    # Cleanup
    # shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_analytics()
