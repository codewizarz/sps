import sqlite3
import json
import uuid
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path


@dataclass
class ExperimentResult:
    experiment_id: str
    sharpe: float
    max_dd: float
    win_rate: float
    artifact_path: str
    notes: str


class ResearchLab:
    """
    Manages the Research Operating System: Hypotheses, Experiments, and Results.
    """

    def __init__(self, db_path: str = "research.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Hypotheses
        c.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                status TEXT, -- PROPOSED, ACTIVE, REJECTED, GRADUATED
                created_at TIMESTAMP
            )
        """)

        # Experiments
        c.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                hypothesis_id TEXT,
                params_json TEXT,
                commit_hash TEXT,
                run_date TIMESTAMP,
                FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(id)
            )
        """)

        # Results
        c.execute("""
            CREATE TABLE IF NOT EXISTS results (
                experiment_id TEXT PRIMARY KEY,
                sharpe REAL,
                max_dd REAL,
                win_rate REAL,
                artifact_path TEXT,
                notes TEXT,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            )
        """)

        conn.commit()
        conn.close()

    def propose_hypothesis(self, title: str, description: str) -> str:
        """
        Registers a new research idea.
        """
        hyp_id = f"HYP-{uuid.uuid4().hex[:8].upper()}"
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO hypotheses (id, title, description, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (hyp_id, title, description, "PROPOSED", datetime.datetime.now()),
        )
        conn.commit()
        conn.close()
        print(f"[ResearchLab] Hypothesis Proposed: {hyp_id} - {title}")
        return hyp_id

    def create_experiment(
        self, hypothesis_id: str, params: Dict[str, Any], commit_hash: str = "HEAD"
    ) -> str:
        """
        Logs an experiment run.
        """
        exp_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Verify hypothesis exists
        c.execute("SELECT id FROM hypotheses WHERE id = ?", (hypothesis_id,))
        if not c.fetchone():
            conn.close()
            raise ValueError(f"Hypothesis {hypothesis_id} not found.")

        c.execute(
            "INSERT INTO experiments (id, hypothesis_id, params_json, commit_hash, run_date) VALUES (?, ?, ?, ?, ?)",
            (
                exp_id,
                hypothesis_id,
                json.dumps(params),
                commit_hash,
                datetime.datetime.now(),
            ),
        )

        # Update Hypothesis status to ACTIVE
        c.execute(
            "UPDATE hypotheses SET status = 'ACTIVE' WHERE id = ?", (hypothesis_id,)
        )

        conn.commit()
        conn.close()
        return exp_id

    def log_result(self, result: ExperimentResult):
        """
        Logs the outcome of an experiment.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute(
            """
            INSERT INTO results (experiment_id, sharpe, max_dd, win_rate, artifact_path, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result.experiment_id,
                result.sharpe,
                result.max_dd,
                result.win_rate,
                result.artifact_path,
                result.notes,
            ),
        )
        conn.commit()
        conn.close()
        print(
            f"[ResearchLab] Result Logged for {result.experiment_id}: Sharpe={result.sharpe:.2f}"
        )

    def get_leaderboard(self) -> pd.DataFrame:
        """
        Returns a DataFrame of all experiments ranked by Sharpe.
        """
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT 
                h.title as Hypothesis,
                e.id as ExperimentID,
                r.sharpe as Sharpe,
                r.max_dd as MaxDD,
                r.win_rate as WinRate,
                r.notes as Conclusion,
                e.run_date as RunDate
            FROM results r
            JOIN experiments e ON r.experiment_id = e.id
            JOIN hypotheses h ON e.hypothesis_id = h.id
            ORDER BY r.sharpe DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def conclude_hypothesis(self, hypothesis_id: str, status: str):
        """
        Marks a hypothesis as GRADUATED or REJECTED.
        """
        if status not in ["GRADUATED", "REJECTED"]:
            raise ValueError("Status must be GRADUATED or REJECTED")

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "UPDATE hypotheses SET status = ? WHERE id = ?", (status, hypothesis_id)
        )
        conn.commit()
        conn.close()
        print(f"[ResearchLab] Hypothesis {hypothesis_id} marked as {status}")
