#!/usr/bin/env python3
"""
=============================================================================
FEATURE ENGINE — Real-Time Rolling Feature Computation
=============================================================================
Maintains a rolling price buffer and computes volatility features (returns,
RV20) in real time from WebSocket ticks.

Used as a DIAGNOSTIC companion to StrategyWrapper's own vol pipeline —
provides independent RV20 logging and "ready" detection without replacing
strategy logic.
=============================================================================
"""

from __future__ import annotations

import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, Optional


class FeatureEngine:
    """
    Rolling feature computer for live tick data.

    Accumulates prices in a fixed-length deque and computes:
      - log returns
      - RV20 (20-period realised volatility, annualised)

    Does NOT replace StrategyWrapper.get_vol_features() — this is a
    lightweight diagnostic layer for logging and Telegram alerts.
    """

    def __init__(self, maxlen: int = 500):
        self.prices: deque = deque(maxlen=maxlen)
        self.timestamps: deque = deque(maxlen=maxlen)

    def update(self, price: float, timestamp: datetime):
        """Append a new price observation."""
        self.prices.append(price)
        self.timestamps.append(timestamp)

    def is_ready(self, min_points: int = 30) -> bool:
        """Check if enough data is accumulated for feature computation."""
        return len(self.prices) >= min_points

    def compute_features(self) -> Optional[Dict]:
        """
        Compute rolling features from the price buffer.

        Returns None if not enough data, otherwise a dict with:
          - price_series: np.ndarray of prices
          - returns: np.ndarray of log returns
          - rv20: annualised 20-period realised volatility
          - timestamp: most recent timestamp
        """
        if not self.is_ready():
            return None

        prices = np.array(self.prices)
        log_returns = np.diff(np.log(prices))

        if len(log_returns) < 20:
            return None

        rv20 = float(np.std(log_returns[-20:]) * np.sqrt(252))

        return {
            "price_series": prices,
            "returns": log_returns,
            "rv20": rv20,
            "timestamp": self.timestamps[-1],
        }
