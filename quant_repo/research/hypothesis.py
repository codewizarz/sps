from dataclasses import dataclass, field, asdict, replace
from typing import Dict, Any, Optional, List
import hashlib
import json
import datetime
import polars as pl


@dataclass
class Hypothesis:
    strategy_type: str  # e.g., "VRP_STRADDLE"
    params: Dict[str, Any]  # { "entry_time": "09:30", "dte": 0, "stop_loss": 0.2 }
    author: str  # "Trader1"
    rationale: str  # "Capture overnight decay"

    @property
    def id(self) -> str:
        """
        Canonical Hash of sorted params + strategy_type.
        Ensures 100% deduplication of identical setups.
        """
        # Sort params to ensure {"a": 1, "b": 2} == {"b": 2, "a": 1}
        # JSON dump needed to handle nested dicts roughly, though for robust PROD use standard serialization
        try:
            params_str = json.dumps(self.params, sort_keys=True)
            raw = f"{self.strategy_type}|{params_str}"
            return hashlib.sha256(raw.encode()).hexdigest()
        except TypeError:
            # Fallback for non-serializable objects (should avoid in params)
            raw = f"{self.strategy_type}|{str(self.params)}"
            return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class ResearchResult:
    hypothesis_id: str
    status: str  # "COMPLETED", "FAILED", "CACHED"
    metrics: Dict[str, float]  # Sharpe, Sortino etc.
    run_timestamp: str
    artifacts_path: Optional[str] = None


class HypothesisRegistry:
    """
    Simulates a persistent database (SQLite/Parquet) for research logs.
    """

    def __init__(self):
        self._store: Dict[str, ResearchResult] = {}

    def exists(self, hyp_id: str) -> bool:
        return hyp_id in self._store

    def get_result(self, hyp_id: str) -> Optional[ResearchResult]:
        return self._store.get(hyp_id)

    def save_result(self, result: ResearchResult):
        self._store[result.hypothesis_id] = result
        # In PROD: append to Parquet or SQL insert


class ResearchOrchestrator:
    """
    Workflow Engine.
    """

    def __init__(self, registry: HypothesisRegistry):
        self.registry = registry

    def run_research(
        self, hypothesis: Hypothesis, force_rerun: bool = False
    ) -> ResearchResult:
        hyp_id = hypothesis.id

        # 1. Deduplication
        if not force_rerun and self.registry.exists(hyp_id):
            print(f"[Research] CACHE HIT: {hyp_id[:8]} ({hypothesis.strategy_type})")
            cached_res = self.registry.get_result(hyp_id)
            # Create a copy with status CACHED for caller awareness, or just return original
            return replace(cached_res, status="CACHED")

        print(f"[Research] RUNNING: {hyp_id[:8]} ({hypothesis.strategy_type})")

        # 2. Execution (Simulated Routing)
        # Here we would instantiate the Strategy Class based on `strategy_type`
        # and run it through `VectorBacktester` or `NautilusRunner`.
        # For this prototype, we simulate a result.

        try:
            metrics = self._simulate_backtest(hypothesis)
            status = "COMPLETED"
        except Exception as e:
            print(f"[Research] FAILED: {e}")
            metrics = {}
            status = "FAILED"

        # 3. Logging
        result = ResearchResult(
            hypothesis_id=hyp_id,
            status=status,
            metrics=metrics,
            run_timestamp=datetime.datetime.now().isoformat(),
            artifacts_path=f"/data/research/{hyp_id}",
        )

        self.registry.save_result(result)
        return result

    def _simulate_backtest(self, hypothesis: Hypothesis) -> Dict[str, float]:
        """
        Mock Backtest Execution.
        In real impl, this calls `VectorBacktester.run(config)`.
        """
        # Deterministic mock metric based on hash to show "consistency"
        h_val = int(hypothesis.id, 16)
        mock_sharpe = (h_val % 300) / 100.0  # 0.0 to 3.0
        return {"sharpe": mock_sharpe, "win_rate": 0.55}

    def get_leaderboard(self) -> pl.DataFrame:
        if not self.registry._store:
            return pl.DataFrame()

        records = []
        for res in self.registry._store.values():
            rec = asdict(res)
            rec["metrics_sharpe"] = res.metrics.get("sharpe", 0.0)  # Flatten for DF
            records.append(rec)

        return pl.DataFrame(records).sort("metrics_sharpe", descending=True)
