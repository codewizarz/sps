from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional


class SignalType(Enum):
    VOL_ARBITRAGE = "VOL_ARBITRAGE"
    PARITY_VIOLATION = "PARITY_VIOLATION"
    SKEW_STEEPENING = "SKEW_STEEPENING"


class Direction(Enum):
    LONG = 1  # Buy Undervalued / Sell Overvalued (net position context dependent)
    SHORT = -1
    FLAT = 0


@dataclass
class Signal:
    timestamp: int  # Unix Nanoseconds
    instrument_id: str
    signal_type: SignalType
    direction: Direction
    strength: float  # Z-Score or dimensionless magnitude
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "instrument_id": self.instrument_id,
            "signal_type": self.signal_type.value,
            "direction": self.direction.name,
            "strength": self.strength,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
