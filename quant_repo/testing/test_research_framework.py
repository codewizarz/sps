import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from quant_repo.research.hypothesis import (
    Hypothesis,
    ResearchOrchestrator,
    HypothesisRegistry,
)


def test_research_framework():
    print("[TEST] Research Acceleration Framework...")

    registry = HypothesisRegistry()
    orchestrator = ResearchOrchestrator(registry)

    # 1. Test ID Stability (Key Order Independence)
    print("\n--- Testing Deduplication Logic ---")
    p1 = {"dte": 45, "stop": 0.2}
    p2 = {"stop": 0.2, "dte": 45}  # Different order

    h1 = Hypothesis("STRAT_A", p1, "User1", "Test")
    h2 = Hypothesis("STRAT_A", p2, "User1", "Test")

    print(f"ID 1: {h1.id}")
    print(f"ID 2: {h2.id}")
    assert h1.id == h2.id

    # 2. Test Execution & Caching
    print("\n--- Testing Execution flow ---")
    res1 = orchestrator.run_research(h1)
    print(f"Run 1 Status: {res1.status} | Sharpe: {res1.metrics['sharpe']}")
    assert res1.status == "COMPLETED"

    res2 = orchestrator.run_research(h2)  # Same ID
    print(f"Run 2 Status: {res2.status}")
    assert res2.status == "CACHED"
    assert res2.metrics["sharpe"] == res1.metrics["sharpe"]

    # 3. Test Leaderboard
    print("\n--- Testing Leaderboard ---")
    # Add a different strategy
    h3 = Hypothesis("STRAT_B", {"dte": 30}, "User2", "Other")
    res3 = orchestrator.run_research(h3)

    leaderboard = orchestrator.get_leaderboard()
    print(leaderboard)
    assert len(leaderboard) == 2
    assert "metrics_sharpe" in leaderboard.columns

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_research_framework()
