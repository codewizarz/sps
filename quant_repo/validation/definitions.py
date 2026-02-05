from dataclasses import dataclass, field
from enum import Enum
from typing import List, Protocol, Dict, Any, Optional
import polars as pl


class ValidationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass
class ValidationResult:
    test_name: str
    status: ValidationStatus
    score: float  # 0.0 to 100.0
    details: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BacktestRunner(Protocol):
    """
    Protocol for a component that can run a backtest and return a trade log.
    """

    def run(self, overrides: Dict[str, Any] = None) -> pl.DataFrame:
        """
        Runs the backtest simulation.
        overrides: Dictionary of parameters to modify (e.g., {'spread_multiplier': 2.0})
        Returns: Trade Log DataFrame
        """
        ...
