#!/usr/bin/env python3
"""
V4 REGIME GENERALIZED STRATEGY
Iteration 3 — Robustness-focused evolution of Meow Final Boss (V3/Iter2)

Key improvements over V3:
  A. Adaptive vol thresholds  — rolling percentile vs static 10/20%
  B. Soft quality filter      — size penalty instead of hard block
  C. Trade frequency guard    — relax cooldown when entry-starved
  D. Continuous regime scale  — sigmoid curve vs discrete buckets
  E. Drawdown-aware sizing    — progressive de-risking under drawdown

Target:
  Forward Sharpe ≥ 1.2 | Max DD < 12% | Min forward trades ≥ 8

=============================================================================
V3 REGIME ADAPTIVE SHORT VOL - Robust Short Volatility Engine
=============================================================================

Extends the V2 smart-risk short straddle framework with a stronger focus on
downside control and portfolio robustness rather than higher trade frequency.

Key upgrades over V2:
  1. Explicit volatility regime classification (low / normal / high vol)
  2. Tail-risk entry vetoes for IV spikes, RV acceleration, and gap risk
  3. Adaptive position sizing linked to realized volatility
  4. Convex stop structure with breakeven and profit-lock trails
  5. Directional delta exit for straddles that stop behaving market-neutral
  6. ATM straddle quality filter using liquidity and spread proxies
  7. Event-day proxy filter and one-cycle cooldown after hard stops
  8. Portfolio overlays for concurrent exposure and correlated index scaling
  9. Daily equity curve with rolling Sharpe and drawdown tracking
  10. Automated comparison against the V2 benchmark

This file intentionally preserves the V2 architecture shape:
  - centralized config
  - volatility engine
  - position sizing
  - trade manager
  - event-driven backtest engine
  - CSV outputs and summary logging
============================================================================="""

from __future__ import annotations

import importlib.util
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


class LiveStrategy:
    """Minimal live tick strategy hook for paper-trader wiring verification."""

    def on_tick(self, symbol=None, price=None, features=None, timestamp=None, **kwargs):
        logger.info("[DEBUG] Strategy on_tick called")
        logger.info(f"[DEBUG] {symbol} | Features: {features}")

        if price is None or features is None:
            logger.info(f"[BLOCKED] {symbol} missing price/features")
            return

        # Extract features
        rv20 = features.get("rv20")
        if rv20 is None:
            logger.info("[BLOCKED] rv20 missing")
            return None

        # Determine regime (simple fallback)
        if rv20 < 0.01:
            regime = "LOW"
        elif rv20 < 0.02:
            regime = "NORMAL"
        else:
            regime = "HIGH"

        logger.info(f"[REGIME] {symbol} | RV20={rv20:.4f} | Regime={regime}")

        # Basic entry condition (temporary debug)
        if regime in ["LOW", "NORMAL"]:
            logger.info("[SIGNAL] Entry condition satisfied")
            return "ENTRY"
        else:
            logger.info("[BLOCKED] Regime too high")
            return "EXIT"


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class StrategyConfig:
    """All tunable parameters for the V3 regime-adaptive engine."""

    # --- Universe / Capital ---
    symbols: Tuple[str, ...] = ("NIFTY", "BANKNIFTY")
    initial_capital: float = 10_000_000
    default_lot_size: int = 25
    margin_per_lot_pct: float = 0.0065
    base_risk_pct: float = 0.05
    max_portfolio_margin_pct: float = 0.65
    max_symbol_margin_pct: float = 0.35
    max_concurrent_positions: int = 2

    # --- Volatility Windows ---
    rv_window_short: int = 5
    rv_window_long: int = 20
    iv_lookback: int = 20
    vov_lookback: int = 10

    # --- Regime Detection ---
    low_vol_threshold: float = 0.10
    high_vol_threshold: float = 0.20
    target_realized_vol: float = 0.12
    realized_vol_floor: float = 0.08
    min_position_fraction: float = 0.40
    max_position_fraction: float = 1.15  # reduced from 1.25 to control DD
    normal_regime_size: float = 0.85
    high_regime_size: float = 0.50
    regime_acceleration_warning: float = 0.15
    regime_iv_expansion_warning: float = 0.10
    regime_signal_penalty: float = 0.90
    high_vol_block_acceleration: float = 0.30
    high_vol_block_iv_expansion: float = 0.25
    vov_threshold: float = 1.25
    vov_penalty: float = 0.90

    # --- Tail Risk Filters (now soft scaling) ---
    iv_spike_limit_2d: float = 0.35
    rv_acceleration_limit_3d: float = 0.40
    gap_move_limit: float = 0.020

    # --- Entry / Timing Filters ---
    entry_dte_primary: int = 3
    event_day_iv_jump_threshold: float = 0.18
    cooldown_cycles_after_stop: int = 1

    # --- Straddle Quality Filters ---
    min_option_oi: int = 75_000
    min_option_volume: int = 75_000
    max_spread_proxy_pct: float = 0.20
    min_premium_absolute: float = 40.0
    min_premium_yield_pct: float = 0.004

    # --- Exit Rules / Convex Risk Controls ---
    stop_loss_multiple: float = 1.35  # slightly tighter for DD control
    breakeven_activation: float = 0.25  # earlier breakeven
    profit_lock_activation: float = 0.45
    profit_lock_stop_pct: float = 0.20
    profit_target_pct: float = 0.65
    partial_profit_decay: float = 0.45
    partial_close_ratio: float = 0.40
    time_exit_pct: float = 0.60  # slightly tighter
    time_exit_min_decay: float = 0.15  # exit earlier if not profitable
    delta_exit_threshold: float = 1.20
    risk_free_rate: float = 0.065

    # --- Streak Memory ---
    streak_boost_after: int = 3
    streak_boost_mult: float = 1.05
    streak_reduce_after: int = 3
    streak_reduce_mult: float = 0.90

    # --- Portfolio Overlays ---
    correlation_pair_scale: float = 0.70
    correlation_pairs: Tuple[Tuple[str, str], ...] = (("NIFTY", "BANKNIFTY"),)

    # --- Transaction Costs ---
    brokerage_per_lot: float = 25.0
    stt_pct: float = 0.0005
    slippage_pct: float = 0.001

    # =========================================================
    # V4 ADDITIONS
    # =========================================================

    # A. Adaptive vol thresholds (rolling percentile of RV20)
    use_adaptive_thresholds: bool = True
    rv_percentile_window: int = 252      # lookback for percentile calc
    rv_low_percentile: float = 0.30      # 30th pct → LOW boundary
    rv_high_percentile: float = 0.70     # 70th pct → HIGH boundary

    # B. Soft quality filter — size penalty instead of hard-blocking
    soft_quality_filter: bool = True
    soft_oi_penalty: float = 0.55        # size × 0.55 when OI < min
    soft_volume_penalty: float = 0.65    # size × 0.65 when vol < min
    soft_premium_penalty: float = 0.70   # size × 0.70 when premium thin

    # C. Trade frequency stabilizer
    frequency_lookback_days: int = 28    # rolling window to count trades
    min_trigger_trades: int = 1          # if fewer, enter frequency mode
    relaxed_cooldown_cycles: int = 0     # cooldown drop when starved
    relaxed_spread_multiplier: float = 1.4  # 40% looser spread gate

    # D. Continuous regime scaling (sigmoid)
    use_continuous_regime_scale: bool = True
    regime_scale_steepness: float = 4.0  # sigmoid steepness
    regime_scale_floor: float = 0.45     # minimum size in HIGH vol
    regime_scale_ceiling: float = 1.05   # maximum size in LOW vol

    # E. Drawdown-aware sizing
    dd_derisking_threshold: float = 5.0  # start de-risking at 5% DD
    dd_derisking_rate: float = 0.12      # reduce lots 12% per % DD over threshold
    dd_derisking_floor: float = 0.45     # minimum remaining fraction
    low_vol_max_position_fraction: float = 0.95  # cap in LOW vol (was 1.15)


class VolatilityRegime(Enum):
    LOW = "LOW VOL"
    NORMAL = "NORMAL VOL"
    HIGH = "HIGH VOL"


class ExitReason(Enum):
    STOP_LOSS = "Stop Loss (1.5x)"
    TRAIL_BREAKEVEN = "Trail to Breakeven"
    TRAIL_PROFIT_LOCK = "Profit Lock Trail"
    DIRECTIONAL_DELTA = "Directional Delta Exit"
    PROFIT_TARGET = "Profit Target (70%)"
    TIME_EXIT = "Time Exit"
    EXPIRY_CLOSE = "Expiry Close"
    EXPIRY_PARTIAL = "Expiry Close + Partial"


@dataclass(frozen=True)
class SymbolContext:
    symbol: str
    prices: pd.Series
    vol_data: pd.DataFrame
    trading_dates: List[pd.Timestamp]
    entry_schedule: Dict[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class OptionPairSnapshot:
    symbol: str
    trade_date: datetime
    expiry_date: datetime
    strike: float
    spot: float
    premium: float
    lot_quantity: int
    ce_close: float
    pe_close: float
    ce_high: float
    pe_high: float
    ce_low: float
    pe_low: float
    ce_last: float
    pe_last: float
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int

    @property
    def min_oi(self) -> int:
        return int(min(self.ce_oi, self.pe_oi))

    @property
    def min_volume(self) -> int:
        return int(min(self.ce_volume, self.pe_volume))

    @property
    def premium_yield(self) -> float:
        return self.premium / self.spot if self.spot > 0 else 0.0

    @property
    def spread_proxy_pct(self) -> float:
        ce_proxy = (
            abs(self.ce_close - self.ce_last) / self.ce_close
            if self.ce_close > 0 and self.ce_last > 0
            else 0.0
        )
        pe_proxy = (
            abs(self.pe_close - self.pe_last) / self.pe_close
            if self.pe_close > 0 and self.pe_last > 0
            else 0.0
        )
        return float(max(ce_proxy, pe_proxy))


@dataclass(frozen=True)
class RegimeDecision:
    regime: VolatilityRegime
    size_multiplier: float
    signal_multiplier: float
    allowed: bool
    reason: str


@dataclass(frozen=True)
class EntryCandidate:
    symbol: str
    entry_date: datetime
    expiry_date: datetime
    strike: float
    spot: float
    premium: float
    lot_quantity: int
    rv20: float
    rv_acceleration_3d: float
    iv: float
    iv_expansion_1d: float
    iv_expansion_2d: float
    gap_move: float
    vov: float
    regime: str
    regime_multiplier: float
    signal_multiplier: float
    tail_scale: float  # NEW: soft tail risk multiplier
    min_oi: int
    min_volume: int
    spread_proxy_pct: float
    premium_yield: float
    quality_score: float
    risk_scaled: bool
    entry_window: str
    rationale: str


@dataclass
class OpenPosition:
    symbol: str
    entry_date: datetime
    expiry_date: datetime
    strike: float
    spot_at_entry: float
    premium: float
    lots: int
    lot_quantity: int
    margin_required: float
    rv_at_entry: float
    rv_acceleration_at_entry: float
    iv_at_entry: float
    iv_expansion_at_entry: float
    gap_move_at_entry: float
    regime: str
    position_scale: float
    risk_scaled: bool
    entry_window: str
    quality_score: float
    spread_proxy_pct: float
    streak_mult: float
    stop_price: float
    stop_stage: int = 0
    best_pnl_pct: float = 0.0
    partial_taken: bool = False
    partial_pnl: float = 0.0
    original_lots: int = 0
    last_delta: float = 0.0
    last_mark_price: float = 0.0

    def __post_init__(self):
        self.original_lots = self.lots
        self.last_mark_price = self.premium


@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    symbol: str
    trade_type: str
    strike: float
    spot: float
    premium: float
    exit_price: float
    lots: int
    lot_quantity: int
    gross_pnl: float
    net_pnl: float
    exit_reason: str
    rv: float
    rv_acceleration: float
    iv: float
    iv_expansion_2d: float
    gap_move: float
    regime: str
    position_scale: float
    risk_scaled: bool
    entry_window: str
    partial_pnl: float
    streak_mult: float
    quality_score: float
    spread_proxy_pct: float
    last_delta: float


# =============================================================================
# VOLATILITY / OPTION UTILITIES
# =============================================================================


class VolatilityEngine:
    """Computes realized vol, IV proxy, and regime features."""

    @staticmethod
    def realized_volatility(prices: pd.Series, window: int) -> pd.Series:
        log_returns = np.log(prices / prices.shift(1))
        return log_returns.rolling(window).std() * np.sqrt(252)

    @staticmethod
    def implied_volatility_proxy(prices: pd.Series, window: int = 20) -> pd.Series:
        log_returns = np.log(prices / prices.shift(1))
        rv = log_returns.rolling(window).std() * np.sqrt(252)
        return rv * 1.25 + 0.01

    @staticmethod
    def vol_of_vol(rv_series: pd.Series, lookback: int = 10) -> pd.Series:
        denom = rv_series.rolling(lookback).mean().replace(0, np.nan)
        return rv_series.rolling(lookback).std() / denom

    @staticmethod
    def compute_all(prices: pd.Series, config: StrategyConfig) -> pd.DataFrame:
        df = pd.DataFrame({"close": prices})
        df["rv5"] = VolatilityEngine.realized_volatility(prices, config.rv_window_short)
        df["rv20"] = VolatilityEngine.realized_volatility(prices, config.rv_window_long)
        df["iv"] = VolatilityEngine.implied_volatility_proxy(prices, config.iv_lookback)
        df["vov"] = VolatilityEngine.vol_of_vol(df["rv20"], config.vov_lookback)
        df["rv_acceleration_3d"] = df["rv20"].pct_change(3)
        df["iv_expansion_1d"] = df["iv"].pct_change(1)
        df["iv_expansion_2d"] = df["iv"].pct_change(2)
        df["iv_jump_prev_day"] = df["iv_expansion_1d"].shift(1)
        df["gap_move"] = prices.pct_change().abs()

        # V4-A: Adaptive rolling percentile thresholds
        if config.use_adaptive_thresholds:
            w = config.rv_percentile_window
            df["rv_low_thresh"] = (
                df["rv20"].rolling(w, min_periods=max(30, w // 4))
                .quantile(config.rv_low_percentile)
                .fillna(config.low_vol_threshold)
            )
            df["rv_high_thresh"] = (
                df["rv20"].rolling(w, min_periods=max(30, w // 4))
                .quantile(config.rv_high_percentile)
                .fillna(config.high_vol_threshold)
            )
        else:
            df["rv_low_thresh"] = config.low_vol_threshold
            df["rv_high_thresh"] = config.high_vol_threshold

        return df


class BSPricer:
    """Black-Scholes helper used as fallback when daily option data is sparse."""

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(K - S, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def straddle_premium(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return BSPricer.call_price(S, K, T, r, sigma) + BSPricer.put_price(
            S, K, T, r, sigma
        )


class GreeksEngine:
    """Delta proxy for ATM straddle directional drift checks."""

    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        return float(norm.cdf(GreeksEngine._d1(S, K, T, r, sigma)))

    @staticmethod
    def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return -1.0 if S < K else 0.0
        return float(norm.cdf(GreeksEngine._d1(S, K, T, r, sigma)) - 1.0)

    @staticmethod
    def short_straddle_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        long_straddle_delta = GreeksEngine.call_delta(S, K, T, r, sigma) + GreeksEngine.put_delta(S, K, T, r, sigma)
        return -float(long_straddle_delta)


# =============================================================================
# ENTRY FILTERS / SIZING
# =============================================================================


class RegimeFilter:
    """V4: Classifies vol regime using adaptive thresholds + continuous sigmoid scaling."""

    @staticmethod
    def _sigmoid_scale(
        rv20: float,
        rv_low: float,
        rv_high: float,
        config: StrategyConfig,
    ) -> float:
        """Smooth sigmoid sizing: 1.05 at low vol, 0.45 at high vol."""
        rv_mid = (rv_low + rv_high) / 2.0
        rv_range = max(rv_high - rv_low, 1e-4)
        z = (rv20 - rv_mid) / (rv_range * 0.4)
        sigmoid = 1.0 / (1.0 + np.exp(config.regime_scale_steepness * z))
        scale = config.regime_scale_floor + (config.regime_scale_ceiling - config.regime_scale_floor) * sigmoid
        return float(np.clip(scale, config.regime_scale_floor, config.regime_scale_ceiling))

    @staticmethod
    def assess(
        rv20: float,
        rv_acceleration_3d: float,
        iv_expansion_2d: float,
        config: StrategyConfig,
        rv_low_thresh: float = None,
        rv_high_thresh: float = None,
    ) -> RegimeDecision:
        rv20 = float(rv20)
        rv_acceleration_3d = float(np.nan_to_num(rv_acceleration_3d, nan=0.0))
        iv_expansion_2d = float(np.nan_to_num(iv_expansion_2d, nan=0.0))

        rv_low = rv_low_thresh if rv_low_thresh is not None else config.low_vol_threshold
        rv_high = rv_high_thresh if rv_high_thresh is not None else config.high_vol_threshold

        if rv20 < rv_low:
            regime = VolatilityRegime.LOW
        elif rv20 <= rv_high:
            regime = VolatilityRegime.NORMAL
        else:
            regime = VolatilityRegime.HIGH

        # V4-D: Continuous sigmoid size multiplier
        if config.use_continuous_regime_scale:
            size_multiplier = RegimeFilter._sigmoid_scale(rv20, rv_low, rv_high, config)
        else:
            if regime == VolatilityRegime.LOW:
                size_multiplier = 1.0
            elif regime == VolatilityRegime.NORMAL:
                size_multiplier = config.normal_regime_size
            else:
                size_multiplier = config.high_regime_size

        signal_multiplier = 1.0
        notes: List[str] = [f"RV={rv20 * 100:.1f} lo={rv_low*100:.1f} hi={rv_high*100:.1f}"]

        if rv_acceleration_3d > config.regime_acceleration_warning:
            signal_multiplier *= config.regime_signal_penalty
            notes.append(f"RV accel={rv_acceleration_3d:.0%}")

        if iv_expansion_2d > config.regime_iv_expansion_warning:
            signal_multiplier *= config.regime_signal_penalty
            notes.append(f"IV expand={iv_expansion_2d:.0%}")

        allowed = not (
            regime == VolatilityRegime.HIGH
            and (
                rv_acceleration_3d > config.high_vol_block_acceleration
                or iv_expansion_2d > config.high_vol_block_iv_expansion
            )
        )

        if not allowed:
            notes.append("blocked high-vol escalation")

        return RegimeDecision(
            regime=regime,
            size_multiplier=size_multiplier,
            signal_multiplier=signal_multiplier,
            allowed=allowed,
            reason=" | ".join(notes),
        )


class TailRiskFilter:
    """Applies soft scaling when short-vol convexity risk is visibly rising."""

    @staticmethod
    def compute_multiplier(
        iv_expansion_2d: float,
        rv_acceleration_3d: float,
        gap_move: float,
        config: StrategyConfig,
    ) -> Tuple[float, str]:
        iv_exp = float(np.nan_to_num(iv_expansion_2d, nan=0.0))
        rv_acc = float(np.nan_to_num(rv_acceleration_3d, nan=0.0))
        gap = float(np.nan_to_num(gap_move, nan=0.0))

        scale = 1.0
        notes: List[str] = []

        # IV spike: scale down proportionally
        if iv_exp > config.iv_spike_limit_2d:
            excess = (iv_exp - config.iv_spike_limit_2d) / config.iv_spike_limit_2d
            penalty = min(0.20 + excess * 0.5, 0.50)
            scale *= (1.0 - penalty)
            notes.append(f"IV spike {iv_exp:.0%} -> -{penalty*100:.0f}%")
        elif iv_exp > config.iv_spike_limit_2d * 0.5:
            notes.append(f"IV up {iv_exp:.0%}")

        # RV acceleration: scale down proportionally
        if rv_acc > config.rv_acceleration_limit_3d:
            excess = (rv_acc - config.rv_acceleration_limit_3d) / config.rv_acceleration_limit_3d
            penalty = min(0.15 + excess * 0.4, 0.40)
            scale *= (1.0 - penalty)
            notes.append(f"RV accel {rv_acc:.0%} -> -{penalty*100:.0f}%")

        # Gap move: minor scaling
        if gap > config.gap_move_limit:
            excess = (gap - config.gap_move_limit) / config.gap_move_limit
            penalty = min(0.10 + excess * 0.3, 0.25)
            scale *= (1.0 - penalty)
            notes.append(f"Gap {gap:.1%} -> -{penalty*100:.0f}%")

        reason = "; ".join(notes) if notes else "tail risk neutral"
        return max(scale, 0.30), reason  # floor at 30%


class QualityFilter:
    """V4-B: Soft filter — returns size penalty instead of hard-blocking on OI/volume."""

    @staticmethod
    def check(
        snapshot: OptionPairSnapshot,
        config: StrategyConfig,
        relaxed_spread: bool = False,
    ) -> Tuple[bool, float, float, str]:
        """Returns (quality_ok, quality_score, size_penalty, reason).
        Hard-blocks only when no option data or premium=0.
        Soft-penalises OI/volume/spread shortfalls.
        """
        notes: List[str] = []
        size_penalty = 1.0  # multiplicative sizing penalty

        # Hard block: no usable premium at all
        if snapshot.premium <= 0:
            return False, 0.0, 0.0, "Zero premium"

        max_spread = (
            config.max_spread_proxy_pct * config.relaxed_spread_multiplier
            if relaxed_spread
            else config.max_spread_proxy_pct
        )

        # V4-B: Soft penalties (size scaling) instead of hard reject
        if config.soft_quality_filter:
            if snapshot.min_oi < config.min_option_oi:
                size_penalty *= config.soft_oi_penalty
                notes.append(f"OI soft-penalty {snapshot.min_oi:,}")
            if snapshot.min_volume < config.min_option_volume:
                size_penalty *= config.soft_volume_penalty
                notes.append(f"Vol soft-penalty {snapshot.min_volume:,}")
            if snapshot.spread_proxy_pct > max_spread:
                # Still hard-block extreme spreads (>3x threshold)
                if snapshot.spread_proxy_pct > max_spread * 3:
                    return False, 0.0, 0.0, f"Spread too wide {snapshot.spread_proxy_pct:.1%}"
                size_penalty *= 0.75
                notes.append(f"Spread soft-penalty {snapshot.spread_proxy_pct:.1%}")

            premium_floor = max(
                config.min_premium_absolute * 0.6,  # 40% softer floor
                snapshot.spot * config.min_premium_yield_pct * 0.6,
            )
            if snapshot.premium < premium_floor:
                size_penalty *= config.soft_premium_penalty
                notes.append(f"Premium soft-penalty {snapshot.premium:.1f}")
        else:
            # Original hard-block logic
            blockers: List[str] = []
            if snapshot.min_oi < config.min_option_oi:
                blockers.append(f"OI {snapshot.min_oi:,}")
            if snapshot.min_volume < config.min_option_volume:
                blockers.append(f"Vol {snapshot.min_volume:,}")
            if snapshot.spread_proxy_pct > max_spread:
                blockers.append(f"Spread {snapshot.spread_proxy_pct:.1%}")
            premium_floor = max(config.min_premium_absolute, snapshot.spot * config.min_premium_yield_pct)
            if snapshot.premium < premium_floor:
                blockers.append(f"Premium {snapshot.premium:.1f} < {premium_floor:.1f}")
            if blockers:
                return False, 0.0, 0.0, "; ".join(blockers)

        quality_score = (
            min(snapshot.min_oi / config.min_option_oi, 3.0)
            + min(snapshot.min_volume / config.min_option_volume, 3.0)
            + min(snapshot.premium_yield / config.min_premium_yield_pct, 3.0)
            - min(snapshot.spread_proxy_pct / max(config.max_spread_proxy_pct, 1e-6), 2.0)
        )

        reason = "; ".join(notes) if notes else "ATM quality passed"
        return True, quality_score, size_penalty, reason


class PositionSizer:
    """V4-E: Adaptive lot sizing with drawdown-aware de-risking and LOW VOL cap."""

    @staticmethod
    def compute_lots(
        equity: float,
        spot: float,
        lot_quantity: int,
        rv20: float,
        vov: float,
        regime_multiplier: float,
        signal_multiplier: float,
        tail_scale: float,
        streak_mult: float,
        corr_scale: float,
        used_margin: float,
        config: StrategyConfig,
        current_drawdown_pct: float = 0.0,
        quality_size_penalty: float = 1.0,
        regime: Optional[str] = None,
    ) -> Tuple[int, float, float]:
        margin_per_lot = spot * config.margin_per_lot_pct * lot_quantity
        if margin_per_lot <= 0 or equity <= 0:
            return 0, 0.0, 0.0

        base_capital = equity * config.base_risk_pct
        base_lots = int(base_capital / margin_per_lot)
        if base_lots <= 0:
            return 0, 0.0, 0.0

        vol_scale = config.target_realized_vol / max(rv20, config.realized_vol_floor)

        # V4-E: Cap LOW VOL position fraction to prevent over-concentration
        max_frac = (
            config.low_vol_max_position_fraction
            if regime == VolatilityRegime.LOW.value
            else config.max_position_fraction
        )
        vol_scale = float(np.clip(vol_scale, config.min_position_fraction, max_frac))

        adjustment_scale = regime_multiplier * signal_multiplier * tail_scale * streak_mult
        if float(np.nan_to_num(vov, nan=0.0)) > config.vov_threshold:
            adjustment_scale *= config.vov_penalty

        raw_scale = min(vol_scale * adjustment_scale, max_frac)
        final_scale = max(config.min_position_fraction, raw_scale)
        final_scale *= corr_scale
        final_scale = min(final_scale, max_frac)

        # V4-B: Apply quality soft penalty
        final_scale *= quality_size_penalty

        # V4-E: Progressive drawdown de-risking
        dd = float(np.nan_to_num(current_drawdown_pct, nan=0.0))
        if dd > config.dd_derisking_threshold:
            dd_factor = max(
                config.dd_derisking_floor,
                1.0 - config.dd_derisking_rate * (dd - config.dd_derisking_threshold),
            )
            final_scale *= dd_factor

        lots = int(np.floor(base_lots * final_scale))
        if lots < 1:
            lots = 1

        available_portfolio_margin = max(
            equity * config.max_portfolio_margin_pct - used_margin,
            0.0,
        )
        max_lots_portfolio = int(available_portfolio_margin / margin_per_lot)
        max_lots_symbol = int((equity * config.max_symbol_margin_pct) / margin_per_lot)
        lots = min(lots, max_lots_portfolio, max_lots_symbol)

        if lots <= 0:
            return 0, 0.0, 0.0

        return lots, final_scale, lots * margin_per_lot


class PortfolioRiskManager:
    """Portfolio-level exposure overlays for concurrent index risk."""

    @staticmethod
    def current_margin_usage(open_positions: List[OpenPosition]) -> float:
        return float(sum(p.margin_required for p in open_positions))

    @staticmethod
    def correlation_scale(
        symbol: str,
        active_symbols: List[str],
        pending_symbols: List[str],
        config: StrategyConfig,
    ) -> float:
        active_set = set(active_symbols) | set(pending_symbols) | {symbol}
        for sym_a, sym_b in config.correlation_pairs:
            if {sym_a, sym_b}.issubset(active_set) and symbol in {sym_a, sym_b}:
                return config.correlation_pair_scale
        return 1.0


class CostCalculator:
    """Realistic transaction cost model for options round trips."""

    @staticmethod
    def compute(
        premium: float,
        lots: int,
        lot_quantity: int,
        config: StrategyConfig,
    ) -> float:
        notional = premium * lots * lot_quantity
        brokerage = config.brokerage_per_lot * lots * 2
        stt = notional * config.stt_pct
        slippage = notional * config.slippage_pct
        return brokerage + stt + slippage


class StreakTracker:
    """Preserves V2's risk memory, but with a more conservative default tilt."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.consecutive_wins = 0
        self.consecutive_losses = 0

    def record_trade(self, is_win: bool):
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    @property
    def multiplier(self) -> float:
        if self.consecutive_losses >= self.config.streak_reduce_after:
            return self.config.streak_reduce_mult
        if self.consecutive_wins >= self.config.streak_boost_after:
            return self.config.streak_boost_mult
        return 1.0


# =============================================================================
# TRADE MANAGER
# =============================================================================


class TradeManager:
    """Applies convex exit logic to live short straddles."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def check_exits(
        self,
        position: OpenPosition,
        current_date: datetime,
        current_spot: float,
        current_iv: float,
        current_price: float,
        worst_price: float,
    ) -> Tuple[Optional[ExitReason], float]:
        cfg = self.config
        dte = max((position.expiry_date - current_date).days, 0)
        T = dte / 365.0

        pnl_pct = (
            (position.premium - current_price) / position.premium
            if position.premium > 0
            else 0.0
        )
        position.best_pnl_pct = max(position.best_pnl_pct, pnl_pct)
        position.last_mark_price = current_price

        if pnl_pct >= cfg.breakeven_activation and position.stop_stage < 1:
            position.stop_stage = 1
            position.stop_price = min(position.stop_price, position.premium)

        if pnl_pct >= cfg.profit_lock_activation and position.stop_stage < 2:
            position.stop_stage = 2
            position.stop_price = min(
                position.stop_price,
                position.premium * (1.0 - cfg.profit_lock_stop_pct),
            )

        if worst_price >= position.stop_price:
            if position.stop_stage == 0:
                return ExitReason.STOP_LOSS, position.stop_price
            if position.stop_stage == 1:
                return ExitReason.TRAIL_BREAKEVEN, position.stop_price
            return ExitReason.TRAIL_PROFIT_LOCK, position.stop_price

        net_delta = GreeksEngine.short_straddle_delta(
            current_spot,
            position.strike,
            T,
            cfg.risk_free_rate,
            max(current_iv, 0.01),
        )
        position.last_delta = net_delta
        if abs(net_delta) > cfg.delta_exit_threshold:
            return ExitReason.DIRECTIONAL_DELTA, current_price

        if pnl_pct >= cfg.profit_target_pct:
            return ExitReason.PROFIT_TARGET, current_price

        if (
            not position.partial_taken
            and pnl_pct >= cfg.partial_profit_decay
            and position.lots > 1
        ):
            lots_to_close = int(position.lots * cfg.partial_close_ratio)
            if lots_to_close > 0:
                pnl_per_lot = (position.premium - current_price) * position.lot_quantity
                position.partial_pnl += lots_to_close * pnl_per_lot
                position.lots -= lots_to_close
                position.partial_taken = True

        total_days = max((position.expiry_date - position.entry_date).days, 1)
        elapsed_days = max((current_date - position.entry_date).days, 0)
        time_elapsed_pct = elapsed_days / total_days
        if time_elapsed_pct >= cfg.time_exit_pct and pnl_pct < cfg.time_exit_min_decay:
            return ExitReason.TIME_EXIT, current_price

        if dte <= 0:
            exit_reason = (
                ExitReason.EXPIRY_PARTIAL
                if position.partial_taken
                else ExitReason.EXPIRY_CLOSE
            )
            return exit_reason, current_price

        return None, current_price


# =============================================================================
# PERFORMANCE ANALYTICS
# =============================================================================


class PerformanceTracker:
    """Builds daily equity metrics and summary stats."""

    @staticmethod
    def enrich_equity_curve(equity_df: pd.DataFrame) -> pd.DataFrame:
        if equity_df.empty:
            return equity_df

        df = equity_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df["Peak_Equity"] = df["Equity"].cummax()
        df["Drawdown_Pct"] = (df["Equity"] / df["Peak_Equity"] - 1.0) * 100.0
        df["Daily_Return"] = df["Equity"].pct_change().fillna(0.0)
        rolling_mean = df["Daily_Return"].rolling(30).mean()
        rolling_std = df["Daily_Return"].rolling(30).std(ddof=0).replace(0, np.nan)
        df["Rolling_Sharpe_30D"] = np.sqrt(252) * rolling_mean / rolling_std
        return df

    @staticmethod
    def compute_summary(
        trades_df: pd.DataFrame,
        equity_df: pd.DataFrame,
        initial_capital: float,
    ) -> Dict[str, float]:
        if equity_df.empty:
            return {
                "Final_Equity": initial_capital,
                "Total_Return_Pct": 0.0,
                "Sharpe": 0.0,
                "Sortino": 0.0,
                "Max_Drawdown_Pct": 0.0,
                "Calmar": 0.0,
                "Profit_Factor": 0.0,
                "Win_Rate_Pct": 0.0,
                "Total_Trades": 0,
                "Rolling_Sharpe_30D_Last": 0.0,
            }

        final_equity = float(equity_df["Equity"].iloc[-1])
        total_return_pct = (final_equity / initial_capital - 1.0) * 100.0
        daily_returns = equity_df["Daily_Return"]
        daily_std = daily_returns.std(ddof=0)
        sharpe = (
            float(np.sqrt(252) * daily_returns.mean() / daily_std)
            if daily_std and not np.isnan(daily_std)
            else 0.0
        )

        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std(ddof=0)
        sortino = (
            float(np.sqrt(252) * daily_returns.mean() / downside_std)
            if downside_std and not np.isnan(downside_std)
            else 0.0
        )

        max_drawdown_pct = abs(float(equity_df["Drawdown_Pct"].min()))
        calmar = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0.0
        rolling_sharpe_last = float(
            equity_df["Rolling_Sharpe_30D"].dropna().iloc[-1]
            if equity_df["Rolling_Sharpe_30D"].notna().any()
            else 0.0
        )

        if trades_df.empty:
            gross_wins = 0.0
            gross_losses = 0.0
            win_rate = 0.0
            total_trades = 0
        else:
            gross_wins = float(trades_df.loc[trades_df["Net_PnL"] > 0, "Net_PnL"].sum())
            gross_losses = abs(
                float(trades_df.loc[trades_df["Net_PnL"] <= 0, "Net_PnL"].sum())
            )
            total_trades = int(len(trades_df))
            win_rate = float((trades_df["Net_PnL"] > 0).mean() * 100.0)

        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0
        return_dd = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0.0

        return {
            "Final_Equity": round(final_equity, 2),
            "Total_Return_Pct": round(total_return_pct, 2),
            "Sharpe": round(sharpe, 3),
            "Sortino": round(sortino, 3),
            "Max_Drawdown_Pct": round(max_drawdown_pct, 2),
            "Calmar": round(calmar, 3),
            "Return_DD": round(return_dd, 3),
            "Profit_Factor": round(profit_factor, 3),
            "Win_Rate_Pct": round(win_rate, 2),
            "Total_Trades": total_trades,
            "Rolling_Sharpe_30D_Last": round(rolling_sharpe_last, 3),
        }

    @staticmethod
    def align_realized_equity_to_calendar(
        equity_df: pd.DataFrame,
        calendar: pd.Index,
        initial_capital: float,
    ) -> pd.DataFrame:
        if calendar.empty:
            return pd.DataFrame()

        df = equity_df.copy()
        if df.empty or "Date" not in df.columns or "Equity" not in df.columns:
            aligned = pd.DataFrame(index=pd.to_datetime(calendar))
            aligned["Equity"] = initial_capital
        else:
            df["Date"] = pd.to_datetime(df["Date"])
            aligned = (
                df[["Date", "Equity"]]
                .drop_duplicates(subset=["Date"], keep="last")
                .set_index("Date")
                .reindex(pd.to_datetime(calendar))
                .ffill()
            )
            aligned["Equity"] = aligned["Equity"].fillna(initial_capital)

        aligned = aligned.reset_index()
        aligned = aligned.rename(columns={aligned.columns[0]: "Date"})
        return PerformanceTracker.enrich_equity_curve(aligned)

    @staticmethod
    def build_comparison(v2_summary: Dict[str, float], v3_summary: Dict[str, float]) -> pd.DataFrame:
        metrics = [
            "Final_Equity",
            "Total_Return_Pct",
            "Sharpe",
            "Sortino",
            "Max_Drawdown_Pct",
            "Calmar",
            "Return_DD",
            "Profit_Factor",
            "Win_Rate_Pct",
            "Total_Trades",
            "Rolling_Sharpe_30D_Last",
        ]

        rows = []
        for metric in metrics:
            v2_value = float(v2_summary.get(metric, 0.0))
            v3_value = float(v3_summary.get(metric, 0.0))
            rows.append(
                {
                    "Metric": metric,
                    "V2": v2_value,
                    "V3": v3_value,
                    "Delta": round(v3_value - v2_value, 4),
                }
            )
        return pd.DataFrame(rows)


# =============================================================================
# BACKTEST ENGINE
# =============================================================================


class BacktestEngine:
    """Daily event-driven engine that keeps V2's data-lake and logging model."""

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.trade_manager = TradeManager(self.config)
        self.streak_tracker = StreakTracker(self.config)
        self.cost_calculator = CostCalculator()

        self.lake_path = str(
            Path(
                Path(__file__).resolve().parent.parent.parent.parent
                / "data"
                / "master_fo_lake"
            ).absolute()
        )
        self.con = duckdb.connect(":memory:")
        self._setup_db()

        self.closed_equity = self.config.initial_capital
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Dict[str, float]] = []
        self.open_positions: List[OpenPosition] = []
        self.cooldown_cycles_remaining = 0
        self.option_cache: Dict[Tuple[str, str, str, Optional[float]], pd.DataFrame] = {}

    def _setup_db(self):
        logger.info(f"Registering Lake: {self.lake_path}")
        try:
            pattern = str(Path(self.lake_path) / "**" / "*.parquet")
            self.con.execute("INSTALL parquet;")
            self.con.execute("LOAD parquet;")
            self.con.execute(
                f"CREATE OR REPLACE VIEW fo_data AS SELECT * FROM read_parquet('{pattern}', hive_partitioning=true);"
            )
        except Exception as exc:
            logger.error(f"Failed to setup DuckDB: {exc}")
            raise

    def get_trading_dates_and_expiries(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
        d_query = f"SELECT DISTINCT TradDt FROM fo_data WHERE TckrSymb = '{symbol}'"
        e_query = f"SELECT DISTINCT XpryDt FROM fo_data WHERE TckrSymb = '{symbol}'"

        if start_date:
            d_query += f" AND TradDt >= '{start_date}'"
            e_query += f" AND XpryDt >= '{start_date}'"
        if end_date:
            d_query += f" AND TradDt <= '{end_date}'"
            e_query += f" AND XpryDt <= '{end_date}'"

        d_query += " ORDER BY TradDt ASC"
        e_query += " ORDER BY XpryDt ASC"

        trading_dates = pd.to_datetime(self.con.execute(d_query).df()["TradDt"]).tolist()
        expiries = pd.to_datetime(self.con.execute(e_query).df()["XpryDt"]).tolist()
        return trading_dates, expiries

    def get_spot_data(self, symbol: str) -> pd.Series:
        query = (
            "SELECT TradDt, AVG(UndrlygPric) AS Price "
            f"FROM fo_data WHERE TckrSymb = '{symbol}' "
            "GROUP BY TradDt ORDER BY TradDt ASC"
        )
        df = self.con.execute(query).df()
        df["TradDt"] = pd.to_datetime(df["TradDt"])
        df.set_index("TradDt", inplace=True)
        return df["Price"]

    def _validate_symbols(self) -> List[str]:
        active_symbols: List[str] = []
        for symbol in self.config.symbols:
            count = self.con.execute(
                f"SELECT COUNT(*) FROM fo_data WHERE TckrSymb = '{symbol}'"
            ).fetchone()[0]
            if count > 0:
                active_symbols.append(symbol)
                logger.info(f"{symbol} rows in lake: {count}")
            else:
                logger.warning(f"Skipping {symbol}: no rows found in lake")

        if not active_symbols:
            raise RuntimeError("No configured symbols found in the FO lake.")
        return active_symbols

    def _prepare_symbol_contexts(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, SymbolContext]:
        contexts: Dict[str, SymbolContext] = {}
        for symbol in self._validate_symbols():
            prices = self.get_spot_data(symbol)
            vol_data = VolatilityEngine.compute_all(prices, self.config)
            trading_dates, expiries = self.get_trading_dates_and_expiries(
                symbol, start_date, end_date
            )
            date_map = {pd.Timestamp(d): idx for idx, d in enumerate(trading_dates)}
            entry_schedule: Dict[pd.Timestamp, pd.Timestamp] = {}

            for expiry in expiries:
                expiry_ts = pd.Timestamp(expiry)
                if expiry_ts not in date_map:
                    continue
                entry_idx = date_map[expiry_ts] - self.config.entry_dte_primary
                if entry_idx < 0:
                    continue
                entry_date = pd.Timestamp(trading_dates[entry_idx])
                existing = entry_schedule.get(entry_date)
                if existing is None or expiry_ts < existing:
                    entry_schedule[entry_date] = expiry_ts

            contexts[symbol] = SymbolContext(
                symbol=symbol,
                prices=prices,
                vol_data=vol_data,
                trading_dates=[pd.Timestamp(d) for d in trading_dates],
                entry_schedule=entry_schedule,
            )

        return contexts

    def _fetch_option_slice(
        self,
        symbol: str,
        trade_date: pd.Timestamp,
        expiry_date: pd.Timestamp,
        strike: Optional[float] = None,
    ) -> pd.DataFrame:
        key = (
            symbol,
            trade_date.strftime("%Y-%m-%d"),
            expiry_date.strftime("%Y-%m-%d"),
            None if strike is None else float(strike),
        )
        if key in self.option_cache:
            return self.option_cache[key]

        strike_clause = f" AND StrkPric = {float(strike)}" if strike is not None else ""
        query = f"""
        SELECT
            OptnTp, StrkPric, OpnPric, HghPric, LwPric, ClsPric,
            LastPric, UndrlygPric, OpnIntrst, TtlTradgVol, NewBrdLotQty
        FROM fo_data
        WHERE TckrSymb = '{symbol}'
          AND TradDt = '{trade_date.strftime("%Y-%m-%d")}'
          AND XpryDt = '{expiry_date.strftime("%Y-%m-%d")}'
          {strike_clause}
        """
        df = self.con.execute(query).df()
        self.option_cache[key] = df
        return df

    def _select_atm_snapshot(
        self,
        symbol: str,
        trade_date: pd.Timestamp,
        expiry_date: pd.Timestamp,
        spot: float,
    ) -> Optional[OptionPairSnapshot]:
        options = self._fetch_option_slice(symbol, trade_date, expiry_date)
        if options.empty:
            return None

        calls = options[options["OptnTp"] == "CE"]
        puts = options[options["OptnTp"] == "PE"]
        if calls.empty or puts.empty:
            return None

        common_strikes = sorted(set(calls["StrkPric"].tolist()) & set(puts["StrkPric"].tolist()))
        if not common_strikes:
            return None

        strike = min(common_strikes, key=lambda value: abs(float(value) - float(spot)))
        ce_row = calls[calls["StrkPric"] == strike].iloc[0]
        pe_row = puts[puts["StrkPric"] == strike].iloc[0]
        lot_quantity = int(
            ce_row["NewBrdLotQty"]
            if pd.notna(ce_row["NewBrdLotQty"])
            else self.config.default_lot_size
        )

        return OptionPairSnapshot(
            symbol=symbol,
            trade_date=trade_date.to_pydatetime(),
            expiry_date=expiry_date.to_pydatetime(),
            strike=float(strike),
            spot=float(spot),
            premium=float(ce_row["ClsPric"] + pe_row["ClsPric"]),
            lot_quantity=lot_quantity,
            ce_close=float(ce_row["ClsPric"]),
            pe_close=float(pe_row["ClsPric"]),
            ce_high=float(ce_row["HghPric"]),
            pe_high=float(pe_row["HghPric"]),
            ce_low=float(ce_row["LwPric"]),
            pe_low=float(pe_row["LwPric"]),
            ce_last=float(ce_row["LastPric"]),
            pe_last=float(pe_row["LastPric"]),
            ce_oi=int(ce_row["OpnIntrst"] or 0),
            pe_oi=int(pe_row["OpnIntrst"] or 0),
            ce_volume=int(ce_row["TtlTradgVol"] or 0),
            pe_volume=int(pe_row["TtlTradgVol"] or 0),
        )

    def _build_candidate(
        self,
        context: SymbolContext,
        entry_date: pd.Timestamp,
        expiry_date: pd.Timestamp,
        entry_window: str,
        frequency_mode: bool = False,
    ) -> Optional[EntryCandidate]:
        row = context.vol_data.loc[entry_date]
        spot = float(context.prices.loc[entry_date])

        rv20 = float(row["rv20"])
        rv_acceleration_3d = float(np.nan_to_num(row["rv_acceleration_3d"], nan=0.0))
        iv = float(row["iv"])
        iv_expansion_1d = float(np.nan_to_num(row["iv_expansion_1d"], nan=0.0))
        iv_expansion_2d = float(np.nan_to_num(row["iv_expansion_2d"], nan=0.0))
        iv_jump_prev_day = float(np.nan_to_num(row["iv_jump_prev_day"], nan=0.0))
        gap_move = float(np.nan_to_num(row["gap_move"], nan=0.0))
        vov = float(np.nan_to_num(row["vov"], nan=0.0))

        # V4-A: Read adaptive thresholds computed by VolatilityEngine
        rv_low_thresh = float(np.nan_to_num(row.get("rv_low_thresh", self.config.low_vol_threshold), nan=self.config.low_vol_threshold))
        rv_high_thresh = float(np.nan_to_num(row.get("rv_high_thresh", self.config.high_vol_threshold), nan=self.config.high_vol_threshold))

        if any(pd.isna([rv20, iv])):
            return None

        # In frequency mode, relax the IV jump event-day filter
        iv_jump_limit = (
            self.config.event_day_iv_jump_threshold * 1.5
            if frequency_mode
            else self.config.event_day_iv_jump_threshold
        )
        if iv_jump_prev_day > iv_jump_limit:
            logger.debug(
                f"  SKIP {entry_date.date()} {context.symbol}: event-day proxy IV jump {iv_jump_prev_day:.0%}"
            )
            return None

        snapshot = self._select_atm_snapshot(context.symbol, entry_date, expiry_date, spot)
        if snapshot is None:
            logger.debug(
                f"  SKIP {entry_date.date()} {context.symbol}: ATM option pair not available"
            )
            return None

        # V4-A: Pass adaptive thresholds to regime filter
        regime = RegimeFilter.assess(
            rv20,
            rv_acceleration_3d,
            iv_expansion_2d,
            self.config,
            rv_low_thresh=rv_low_thresh,
            rv_high_thresh=rv_high_thresh,
        )
        if not regime.allowed:
            logger.debug(
                f"  SKIP {entry_date.date()} {context.symbol}: regime blocked | {regime.reason}"
            )
            return None

        tail_scale, tail_reason = TailRiskFilter.compute_multiplier(
            iv_expansion_2d,
            rv_acceleration_3d,
            gap_move,
            self.config,
        )

        # V4-B: Soft quality filter — returns 4-tuple with size_penalty
        quality_ok, quality_score, quality_size_penalty, quality_reason = QualityFilter.check(
            snapshot,
            self.config,
            relaxed_spread=frequency_mode,
        )
        if not quality_ok:
            logger.debug(
                f"  SKIP {entry_date.date()} {context.symbol}: quality hard-block | {quality_reason}"
            )
            return None

        rationale = f"{regime.reason} | {quality_reason} | tail {tail_reason}"
        return EntryCandidate(
            symbol=context.symbol,
            entry_date=entry_date.to_pydatetime(),
            expiry_date=expiry_date.to_pydatetime(),
            strike=snapshot.strike,
            spot=snapshot.spot,
            premium=snapshot.premium,
            lot_quantity=snapshot.lot_quantity,
            rv20=rv20,
            rv_acceleration_3d=rv_acceleration_3d,
            iv=iv,
            iv_expansion_1d=iv_expansion_1d,
            iv_expansion_2d=iv_expansion_2d,
            gap_move=gap_move,
            vov=vov,
            regime=regime.regime.value,
            regime_multiplier=regime.size_multiplier,
            signal_multiplier=regime.signal_multiplier,
            tail_scale=tail_scale * quality_size_penalty,  # V4-B: bake penalty into tail_scale
            min_oi=snapshot.min_oi,
            min_volume=snapshot.min_volume,
            spread_proxy_pct=snapshot.spread_proxy_pct,
            premium_yield=snapshot.premium_yield,
            quality_score=quality_score,
            risk_scaled=regime.regime != VolatilityRegime.LOW,
            entry_window=entry_window,
            rationale=rationale,
        )

    def _mark_position(
        self,
        position: OpenPosition,
        current_date: pd.Timestamp,
        context: SymbolContext,
    ) -> Tuple[float, float, float]:
        spot = float(context.prices.loc[current_date])
        current_iv = float(
            context.vol_data.loc[current_date, "iv"]
            if current_date in context.vol_data.index
            else position.iv_at_entry
        )
        option_rows = self._fetch_option_slice(
            position.symbol,
            current_date,
            pd.Timestamp(position.expiry_date),
            position.strike,
        )

        if option_rows.empty:
            dte = max((position.expiry_date - current_date.to_pydatetime()).days, 0)
            T = dte / 365.0
            current_price = BSPricer.straddle_premium(
                spot,
                position.strike,
                T,
                self.config.risk_free_rate,
                max(current_iv, 0.01),
            )
            return current_price, current_price, current_iv

        ce = option_rows[option_rows["OptnTp"] == "CE"]
        pe = option_rows[option_rows["OptnTp"] == "PE"]
        if ce.empty or pe.empty:
            dte = max((position.expiry_date - current_date.to_pydatetime()).days, 0)
            T = dte / 365.0
            current_price = BSPricer.straddle_premium(
                spot,
                position.strike,
                T,
                self.config.risk_free_rate,
                max(current_iv, 0.01),
            )
            return current_price, current_price, current_iv

        ce_row = ce.iloc[0]
        pe_row = pe.iloc[0]
        ce_close = float(ce_row["ClsPric"])
        pe_close = float(pe_row["ClsPric"])
        ce_high = float(ce_row["HghPric"])
        pe_high = float(pe_row["HghPric"])
        ce_low = float(ce_row["LwPric"])
        pe_low = float(pe_row["LwPric"])

        current_price = ce_close + pe_close
        worst_candidates = [current_price, ce_high + pe_low, ce_low + pe_high]
        worst_price = float(max(x for x in worst_candidates if not np.isnan(x)))
        return current_price, worst_price, current_iv

    def _portfolio_equity(
        self,
        current_date: pd.Timestamp,
        contexts: Dict[str, SymbolContext],
    ) -> float:
        open_pnl = 0.0
        for position in self.open_positions:
            context = contexts[position.symbol]
            if current_date not in context.prices.index:
                continue
            current_price, _, _ = self._mark_position(position, current_date, context)
            remaining_pnl = (
                (position.premium - current_price)
                * position.lots
                * position.lot_quantity
            )
            open_pnl += remaining_pnl + position.partial_pnl
        return self.closed_equity + open_pnl

    def _record_equity_point(
        self,
        current_date: pd.Timestamp,
        contexts: Dict[str, SymbolContext],
    ):
        equity = self._portfolio_equity(current_date, contexts)
        self.equity_curve.append(
            {
                "Date": current_date.strftime("%Y-%m-%d"),
                "Closed_Equity": round(self.closed_equity, 2),
                "Open_Positions": len(self.open_positions),
                "Margin_Used": round(
                    PortfolioRiskManager.current_margin_usage(self.open_positions), 2
                ),
                "Equity": round(equity, 2),
            }
        )

    def _record_exit(
        self,
        position: OpenPosition,
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_reason: ExitReason,
    ):
        remaining_pnl = (
            (position.premium - exit_price) * position.lots * position.lot_quantity
        )
        gross_pnl = remaining_pnl + position.partial_pnl
        costs = self.cost_calculator.compute(
            position.premium,
            position.original_lots,
            position.lot_quantity,
            self.config,
        )
        net_pnl = gross_pnl - costs

        self.closed_equity += net_pnl
        self.streak_tracker.record_trade(net_pnl > 0)
        if exit_reason == ExitReason.STOP_LOSS:
            self.cooldown_cycles_remaining = max(
                self.cooldown_cycles_remaining,
                self.config.cooldown_cycles_after_stop,
            )

        self.trades.append(
            TradeRecord(
                entry_date=position.entry_date.strftime("%Y-%m-%d"),
                exit_date=exit_date.strftime("%Y-%m-%d"),
                symbol=position.symbol,
                trade_type="STRADDLE",
                strike=position.strike,
                spot=position.spot_at_entry,
                premium=round(position.premium, 2),
                exit_price=round(exit_price, 2),
                lots=position.original_lots,
                lot_quantity=position.lot_quantity,
                gross_pnl=round(gross_pnl, 2),
                net_pnl=round(net_pnl, 2),
                exit_reason=exit_reason.value,
                rv=round(position.rv_at_entry, 4),
                rv_acceleration=round(position.rv_acceleration_at_entry, 4),
                iv=round(position.iv_at_entry, 4),
                iv_expansion_2d=round(position.iv_expansion_at_entry, 4),
                gap_move=round(position.gap_move_at_entry, 4),
                regime=position.regime,
                position_scale=round(position.position_scale, 4),
                risk_scaled=position.risk_scaled,
                entry_window=position.entry_window,
                partial_pnl=round(position.partial_pnl, 2),
                streak_mult=round(position.streak_mult, 4),
                quality_score=round(position.quality_score, 4),
                spread_proxy_pct=round(position.spread_proxy_pct, 4),
                last_delta=round(position.last_delta, 4),
            )
        )

        logger.info(
            f"  EXIT  {position.symbol} {exit_date.date()} | {exit_reason.value} | "
            f"P&L Rs {net_pnl:+,.0f} | Closed Equity Rs {self.closed_equity:,.0f}"
        )

    def _process_exits(
        self,
        current_date: pd.Timestamp,
        contexts: Dict[str, SymbolContext],
    ):
        for position in list(self.open_positions):
            context = contexts[position.symbol]
            if current_date not in context.prices.index:
                continue
            if current_date < pd.Timestamp(position.entry_date):
                continue

            current_spot = float(context.prices.loc[current_date])
            current_price, worst_price, current_iv = self._mark_position(
                position,
                current_date,
                context,
            )

            exit_reason, exit_price = self.trade_manager.check_exits(
                position=position,
                current_date=current_date.to_pydatetime(),
                current_spot=current_spot,
                current_iv=current_iv,
                current_price=current_price,
                worst_price=worst_price,
            )

            if exit_reason is not None:
                self._record_exit(position, current_date, exit_price, exit_reason)
                self.open_positions.remove(position)

    def _process_entries(
        self,
        current_date: pd.Timestamp,
        contexts: Dict[str, SymbolContext],
    ):
        # V4-C: Detect frequency starvation mode
        lookback_days = self.config.frequency_lookback_days
        cutoff = current_date - pd.Timedelta(days=lookback_days)
        recent_trade_count = sum(
            1 for t in self.trades
            if pd.Timestamp(t.entry_date) >= cutoff
        )
        frequency_mode = recent_trade_count < self.config.min_trigger_trades
        if frequency_mode:
            logger.debug(f"  FREQ-MODE {current_date.date()}: {recent_trade_count} trades in last {lookback_days}d")

        # V4-C: In frequency mode, allow entry even if cooldown is active (drop to 0)
        effective_cooldown = (
            self.config.relaxed_cooldown_cycles
            if frequency_mode and self.cooldown_cycles_remaining > 0
            else self.cooldown_cycles_remaining
        )

        candidates: List[EntryCandidate] = []
        active_symbols = [position.symbol for position in self.open_positions]

        for symbol, context in contexts.items():
            if current_date not in context.entry_schedule:
                continue
            if symbol in active_symbols:
                continue
            candidate = self._build_candidate(
                context,
                current_date,
                context.entry_schedule[current_date],
                "T-3",
                frequency_mode=frequency_mode,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return

        if effective_cooldown > 0:
            logger.info(
                f"  SKIP {current_date.date()}: cooldown active ({self.cooldown_cycles_remaining} remaining)"
            )
            self.cooldown_cycles_remaining -= 1
            return

        # Sync actual cooldown counter (only decrement if we didn't use frequency bypass)
        if self.cooldown_cycles_remaining > 0 and not frequency_mode:
            self.cooldown_cycles_remaining -= 1
            return
        elif self.cooldown_cycles_remaining > 0 and frequency_mode:
            # Bypass accepted: reset cooldown
            logger.info(f"  FREQ-MODE bypass: overriding cooldown on {current_date.date()}")
            self.cooldown_cycles_remaining = 0

        available_slots = max(
            self.config.max_concurrent_positions - len(self.open_positions),
            0,
        )
        if available_slots <= 0:
            logger.debug(
                f"  SKIP {current_date.date()}: concurrent exposure cap reached"
            )
            return

        candidates = sorted(candidates, key=lambda item: item.quality_score, reverse=True)
        selected = candidates[:available_slots]
        selected_symbols = [candidate.symbol for candidate in selected]

        # V4-E: Compute current portfolio drawdown for position sizer
        reference_equity = self._portfolio_equity(current_date, contexts)
        peak_equity = max(
            self.config.initial_capital,
            max((eq["Equity"] for eq in self.equity_curve), default=self.config.initial_capital),
        )
        current_drawdown_pct = max(0.0, (1.0 - reference_equity / peak_equity) * 100.0)

        for candidate in selected:
            used_margin = PortfolioRiskManager.current_margin_usage(self.open_positions)
            corr_scale = PortfolioRiskManager.correlation_scale(
                candidate.symbol,
                [position.symbol for position in self.open_positions],
                selected_symbols,
                self.config,
            )

            lots, position_scale, margin_required = PositionSizer.compute_lots(
                equity=reference_equity,
                spot=candidate.spot,
                lot_quantity=candidate.lot_quantity,
                rv20=candidate.rv20,
                vov=candidate.vov,
                regime_multiplier=candidate.regime_multiplier,
                signal_multiplier=candidate.signal_multiplier,
                tail_scale=candidate.tail_scale,
                streak_mult=self.streak_tracker.multiplier,
                corr_scale=corr_scale,
                used_margin=used_margin,
                config=self.config,
                current_drawdown_pct=current_drawdown_pct,
                regime=candidate.regime,
            )
            if lots <= 0:
                logger.debug(
                    f"  SKIP {current_date.date()} {candidate.symbol}: insufficient margin"
                )
                continue

            position = OpenPosition(
                symbol=candidate.symbol,
                entry_date=candidate.entry_date,
                expiry_date=candidate.expiry_date,
                strike=candidate.strike,
                spot_at_entry=candidate.spot,
                premium=candidate.premium,
                lots=lots,
                lot_quantity=candidate.lot_quantity,
                margin_required=margin_required,
                rv_at_entry=candidate.rv20,
                rv_acceleration_at_entry=candidate.rv_acceleration_3d,
                iv_at_entry=candidate.iv,
                iv_expansion_at_entry=candidate.iv_expansion_2d,
                gap_move_at_entry=candidate.gap_move,
                regime=candidate.regime,
                position_scale=position_scale,
                risk_scaled=candidate.risk_scaled or corr_scale < 1.0,
                entry_window=candidate.entry_window,
                quality_score=candidate.quality_score,
                spread_proxy_pct=candidate.spread_proxy_pct,
                streak_mult=self.streak_tracker.multiplier,
                stop_price=candidate.premium * self.config.stop_loss_multiple,
            )
            self.open_positions.append(position)

            logger.info(
                f"  ENTRY {candidate.symbol} {current_date.date()} | "
                f"Strike {candidate.strike:.0f} | Premium {candidate.premium:.2f} | "
                f"Lots {lots} x {candidate.lot_quantity} | Regime {candidate.regime} | "
                f"Scale {position_scale:.2f}"
            )

    def _build_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([trade.__dict__ for trade in self.trades]).rename(
            columns={
                "entry_date": "Entry_Date",
                "exit_date": "Exit_Date",
                "symbol": "Symbol",
                "trade_type": "Type",
                "strike": "Strike",
                "spot": "Spot",
                "premium": "Premium",
                "exit_price": "Exit_Price",
                "lots": "Lots",
                "lot_quantity": "Lot_Qty",
                "gross_pnl": "Gross_PnL",
                "net_pnl": "Net_PnL",
                "exit_reason": "Exit_Reason",
                "rv": "RV",
                "rv_acceleration": "RV_Acceleration_3D",
                "iv": "IV",
                "iv_expansion_2d": "IV_Expansion_2D",
                "gap_move": "Gap_Move",
                "regime": "Regime",
                "position_scale": "Position_Scale",
                "risk_scaled": "Risk_Scaled",
                "entry_window": "Entry_Window",
                "partial_pnl": "Partial_PnL",
                "streak_mult": "Streak_Mult",
                "quality_score": "Quality_Score",
                "spread_proxy_pct": "Spread_Proxy_Pct",
                "last_delta": "Last_Delta",
            }
        )

    def _log_summary(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame):
        summary = PerformanceTracker.compute_summary(
            trades_df,
            equity_df,
            self.config.initial_capital,
        )

        logger.info("\n" + "=" * 72)
        logger.info("V3 REGIME ADAPTIVE SHORT VOL SUMMARY")
        logger.info("=" * 72)
        logger.info(f"Configured Symbols: {', '.join(self.config.symbols)}")
        logger.info(f"Final Equity:       Rs {summary['Final_Equity']:,.0f}")
        logger.info(f"Return:             {summary['Total_Return_Pct']:.2f}%")
        logger.info(f"Sharpe:             {summary['Sharpe']:.2f}")
        logger.info(f"Sortino:            {summary['Sortino']:.2f}")
        logger.info(f"Max Drawdown:       {summary['Max_Drawdown_Pct']:.2f}%")
        logger.info(f"Calmar:             {summary['Calmar']:.2f}")
        logger.info(f"Return / DD:        {summary['Return_DD']:.2f}x")
        logger.info(f"Profit Factor:      {summary['Profit_Factor']:.2f}")
        logger.info(f"Win Rate:           {summary['Win_Rate_Pct']:.1f}%")
        logger.info(f"Total Trades:       {summary['Total_Trades']}")
        logger.info(
            f"Rolling Sharpe 30D: {summary['Rolling_Sharpe_30D_Last']:.2f}"
        )
        if not trades_df.empty:
            stop_losses = int(trades_df["Exit_Reason"].str.contains("Stop Loss").sum())
            delta_exits = int(
                trades_df["Exit_Reason"].str.contains("Directional Delta").sum()
            )
            logger.info(f"Hard Stops:         {stop_losses}")
            logger.info(f"Delta Exits:        {delta_exits}")
        logger.info("=" * 72)

    def run(
        self,
        start_date: str = "2025-04-01",
        end_date: str = "2026-03-01",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Starting V3 Regime Adaptive Short Vol Backtest")
        logger.info(
            f"Capital: Rs {self.config.initial_capital:,.0f} | Period: {start_date} to {end_date}"
        )

        contexts = self._prepare_symbol_contexts(start_date, end_date)
        all_dates = sorted(
            set(
                dt
                for context in contexts.values()
                for dt in context.trading_dates
                if pd.Timestamp(start_date) <= dt <= pd.Timestamp(end_date)
            )
        )

        for current_date in all_dates:
            self._process_exits(current_date, contexts)
            self._process_entries(current_date, contexts)
            self._record_equity_point(current_date, contexts)

        trades_df = self._build_trades_df()
        equity_df = PerformanceTracker.enrich_equity_curve(pd.DataFrame(self.equity_curve))
        self._log_summary(trades_df, equity_df)
        return trades_df, equity_df


# =============================================================================
# BENCHMARK COMPARISON
# =============================================================================


def run_v2_benchmark(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    v2_path = Path(__file__).with_name("v2_smart_risk_strategy.py")
    spec = importlib.util.spec_from_file_location("v2_smart_risk_strategy", v2_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load V2 benchmark strategy.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    config = module.StrategyConfig()
    engine = module.BacktestEngine(config)
    prices = engine.get_spot_data("NIFTY")
    trades_df, equity_df = engine.run(prices, start_date=start_date, end_date=end_date)

    calendar = prices.loc[start_date:end_date].index
    benchmark_equity = PerformanceTracker.align_realized_equity_to_calendar(
        pd.DataFrame(equity_df),
        calendar,
        config.initial_capital,
    )

    summary = PerformanceTracker.compute_summary(
        trades_df,
        benchmark_equity,
        config.initial_capital,
    )
    return trades_df, benchmark_equity, summary


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    start_date = "2025-04-01"
    end_date = "2026-03-01"

    config = StrategyConfig()
    engine = BacktestEngine(config)
    trades_df, equity_df = engine.run(start_date=start_date, end_date=end_date)

    if trades_df.empty:
        logger.error("No trades generated. Check filters and data availability.")
        return None, None, None

    trades_df.to_csv("trades_v3.csv", index=False)
    equity_df.to_csv("equity_v3.csv", index=False)
    logger.info("Saved: trades_v3.csv, equity_v3.csv")

    v3_summary = PerformanceTracker.compute_summary(
        trades_df,
        equity_df,
        config.initial_capital,
    )

    try:
        _, _, v2_summary = run_v2_benchmark(start_date, end_date)
        comparison_df = PerformanceTracker.build_comparison(v2_summary, v3_summary)
        comparison_df.to_csv("comparison_v2_vs_v3.csv", index=False)
        logger.info("Saved: comparison_v2_vs_v3.csv")
    except Exception as exc:
        logger.warning(f"V2 comparison skipped: {exc}")
        comparison_df = pd.DataFrame()

    return trades_df, equity_df, comparison_df


if __name__ == "__main__":
    main()
