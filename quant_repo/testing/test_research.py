import sys
from pathlib import Path
import os

sys.path.append(str(Path.cwd()))

from quant_repo.research.lab import ResearchLab, ExperimentResult


def test_research_lab():
    print("[TEST] Research Operating System...")

    # Use a mock DB file
    db_path = "test_research.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    lab = ResearchLab(db_path)

    # 1. Propose Hypothesis
    hyp_id = lab.propose_hypothesis(
        title="VRP in Gold Options",
        description="Selling 20 Delta Strangles on Gold Futures captures risk premium.",
    )
    assert hyp_id.startswith("HYP-")

    # 2. Run Experiment (Mock)
    params = {"lookback": 50, "stop_loss": 2.0}
    exp_id = lab.create_experiment(hyp_id, params, commit_hash="abc1234")
    assert exp_id.startswith("EXP-")

    # 3. Log Result
    # Simulate a bad result
    result = ExperimentResult(
        experiment_id=exp_id,
        sharpe=0.45,
        max_dd=0.25,
        win_rate=0.55,
        artifact_path="/tmp/equity_curve.parquet",
        notes="Failed due to correlation breakdown in 2022.",
    )
    lab.log_result(result)

    # 4. Check Leaderboard
    df = lab.get_leaderboard()
    print("\n[Leaderboard]")
    print(df.to_string())

    assert len(df) == 1
    assert df.iloc[0]["Hypothesis"] == "VRP in Gold Options"
    assert df.iloc[0]["Sharpe"] == 0.45

    # 5. Conclude
    lab.conclude_hypothesis(hyp_id, "REJECTED")

    # Verify DB update (manual check logic or via query)
    import sqlite3

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT status FROM hypotheses WHERE id = ?", (hyp_id,))
    status = c.fetchone()[0]
    conn.close()

    print(f"\nFinal Hypothesis Status: {status}")
    assert status == "REJECTED"

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

    print("[TEST] SUCCESS")


if __name__ == "__main__":
    test_research_lab()
