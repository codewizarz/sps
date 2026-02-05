import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
import numpy as np
import pandas as pd
from quant_repo.factory.pipeline import (
    PipelineOrchestrator,
    StrategyCandidate,
    PipelineStage,
)


def test_alpha_factory():
    print("[TEST] Alpha Factory Pipeline...")

    # 1. Generate Mock Data
    # 252 days
    dates = pd.date_range("2021-01-01", periods=252, freq="B").view(np.int64)
    np.random.seed(42)

    # Random returns
    returns = np.random.normal(0, 0.01, 252)

    # Good Signal: Correlated with returns
    # If Return > 0, Signal = 1. Simple look-ahead proxy to force high sharpe.
    # Shifted logic in tester handles the PnL calc (Signal[t-1] * Ret[t])
    # So we need Signal[t] to predict Ret[t+1]

    # To make a winner: Signal[i] = sign(Ret[i+1])
    # But let's just make 'signal' column = sign(returns) shifted back -1?
    # Actually, let's just create a 'signal' column that aligns with returns for testing.

    # Using 'perfect' signal
    good_signal = np.sign(returns)
    # But backtester shifts signal by 1.
    # PnL[t] = Signal[t-1] * Ret[t].
    # So Signal[t-1] needs to match Ret[t].
    # So Signal must be Returns shifted by -1.
    good_signal = np.roll(np.sign(returns), -1)
    good_signal[-1] = 0

    # Bad Signal: Random
    bad_signal = np.random.choice([-1, 1], size=252)

    df = pl.DataFrame(
        {
            "timestamp": dates,
            "spot_return": returns,
            "signal": good_signal,  # For good candidate
            "bad_signal": bad_signal,  # For bad candidate
        }
    )

    pipeline = PipelineOrchestrator()

    # 2. Test Good Candidate
    cand_good = StrategyCandidate(name="Alpha_Strategy_1", params={})
    # Copy 'signal' to expected Col
    res_good = pipeline.run_pipeline(cand_good, df)  # Uses 'signal' col

    print(f"\n[Result] {res_good.name}: {res_good.current_stage}")
    if res_good.rejection_reason:
        print(f"  Reason: {res_good.rejection_reason}")
    print(f"  Metrics: {res_good.metrics}")

    assert (
        res_good.current_stage == PipelineStage.EVENT_SIM
    )  # Passed Stage 1 and Mock Stage 2 logic (Sharpe > 1.0)
    assert res_good.metrics["stage1_sharpe"] > 2.0

    # 3. Test Bad Candidate
    cand_bad = StrategyCandidate(name="Random_Strategy_Fail", params={})
    # Use bad signal
    df_bad = df.drop("signal").rename({"bad_signal": "signal"})
    res_bad = pipeline.run_pipeline(cand_bad, df_bad)

    print(f"\n[Result] {res_bad.name}: {res_bad.current_stage}")
    if res_bad.rejection_reason:
        print(f"  Reason: {res_bad.rejection_reason}")

    assert res_bad.current_stage == PipelineStage.REJECTED
    # assert "Sharpe too low" in res_bad.rejection_reason

    print("[TEST] SUCCESS")


if __name__ == "__main__":
    test_alpha_factory()
