#!/usr/bin/env python3
"""
=============================================================================
PAPER TRADING — MAIN ORCHESTRATOR
=============================================================================
Runs the paper trading loop for the V4 Regime Generalized strategy.

Usage:
    python3 quant_repo/live/paper_trader.py
    python3 quant_repo/live/paper_trader.py --mode live   # with Angel One API
    python3 quant_repo/live/paper_trader.py --interval 5  # 5s tick interval

NO real trades are placed. All execution is simulated.
=============================================================================
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import signal
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_repo.live.logger import PaperLogger
from quant_repo.live.market_feed import (
    MarketSnapshot, SimulatedFeed, Tick, create_feed,
)
from quant_repo.live.execution_engine import ExecutionEngine
from quant_repo.live.feature_engine import FeatureEngine
from quant_repo.live.market_hours import is_market_open
from quant_repo.live.position_manager import (
    PaperPosition, PositionManager, RiskLimits,
)


logger = logging.getLogger(__name__)


# ── V4 Strategy Wrapper ────────────────────────────────────────────────

class StrategyWrapper:
    """
    Wraps V4 strategy components for live signal generation.
    Imports the strategy module dynamically and uses its:
      - StrategyConfig
      - VolatilityEngine
      - RegimeFilter
      - TailRiskFilter
      - PositionSizer
    WITHOUT modifying any strategy logic.
    """

    def __init__(self, strategy_path: str):
        self.strategy_path = strategy_path
        self._mod = self._load_module(strategy_path)

        # Import strategy classes
        self.StrategyConfig = self._mod.StrategyConfig
        self.VolatilityEngine = self._mod.VolatilityEngine
        self.RegimeFilter = self._mod.RegimeFilter
        self.TailRiskFilter = self._mod.TailRiskFilter
        self.PositionSizer = self._mod.PositionSizer
        self.QualityFilter = self._mod.QualityFilter
        self.VolatilityRegime = self._mod.VolatilityRegime

        self.config = self.StrategyConfig()
        self._live_strategy = None

        # Optional live strategy implementation for direct tick callbacks.
        live_strategy_cls = getattr(self._mod, "LiveStrategy", None)
        if live_strategy_cls is not None:
            try:
                self._live_strategy = live_strategy_cls()
            except Exception as exc:
                logger.error(f"[FATAL] Failed to initialize LiveStrategy: {exc}")

        # Rolling price buffer for vol computation
        self._price_buffers: Dict[str, List[float]] = {}
        self._vol_cache: Dict[str, Dict] = {}

    def on_tick(self, symbol: str, price: float, features: Dict, timestamp: datetime, **kwargs):
        """Delegate live tick handling to strategy module implementation when available."""
        if self._live_strategy is not None and hasattr(self._live_strategy, "on_tick"):
            return self._live_strategy.on_tick(
                symbol=symbol,
                price=price,
                features=features,
                timestamp=timestamp,
                **kwargs,
            )

        mod_on_tick = getattr(self._mod, "on_tick", None)
        if callable(mod_on_tick):
            return mod_on_tick(
                symbol=symbol,
                price=price,
                features=features,
                timestamp=timestamp,
                **kwargs,
            )

        logger.error("[FATAL] Strategy has no on_tick method")
        return None

    @staticmethod
    def _load_module(path: str):
        """Dynamically import the frozen strategy module."""
        mod_name = "v4_strategy"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        # Register in sys.modules BEFORE exec so dataclass introspection works
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    def update_price(self, symbol: str, price: float):
        """Feed a price observation for rolling vol computation."""
        if symbol not in self._price_buffers:
            self._price_buffers[symbol] = []
        self._price_buffers[symbol].append(price)

        # Keep last 300 observations (more than enough for RV252)
        if len(self._price_buffers[symbol]) > 300:
            self._price_buffers[symbol] = self._price_buffers[symbol][-300:]

    def get_vol_features(self, symbol: str) -> Optional[Dict]:
        """Compute volatility features from the rolling price buffer."""
        prices = self._price_buffers.get(symbol, [])
        if len(prices) < 25:
            return None

        series = pd.Series(prices)
        vol_df = self.VolatilityEngine.compute_all(series, self.config)
        latest = vol_df.iloc[-1]

        return {
            "rv20": float(np.nan_to_num(latest.get("rv20", 0), nan=0.08)),
            "rv5": float(np.nan_to_num(latest.get("rv5", 0), nan=0.08)),
            "iv": float(np.nan_to_num(latest.get("iv", 0), nan=0.10)),
            "vov": float(np.nan_to_num(latest.get("vov", 0), nan=0)),
            "rv_acceleration_3d": float(np.nan_to_num(latest.get("rv_acceleration_3d", 0), nan=0)),
            "iv_expansion_1d": float(np.nan_to_num(latest.get("iv_expansion_1d", 0), nan=0)),
            "iv_expansion_2d": float(np.nan_to_num(latest.get("iv_expansion_2d", 0), nan=0)),
            "gap_move": float(np.nan_to_num(latest.get("gap_move", 0), nan=0)),
            "rv_low_thresh": float(np.nan_to_num(latest.get("rv_low_thresh", self.config.low_vol_threshold), nan=self.config.low_vol_threshold)),
            "rv_high_thresh": float(np.nan_to_num(latest.get("rv_high_thresh", self.config.high_vol_threshold), nan=self.config.high_vol_threshold)),
        }

    def generate_signal(
        self,
        symbol: str,
        spot: float,
        equity: float,
        used_margin: float,
        current_dd_pct: float = 0.0,
        active_symbols: List[str] = None,
    ) -> Optional[Dict]:
        """
        Core signal generation function.
        Returns a trade decision dict or None (no signal).
        Uses V4 strategy components WITHOUT modification.
        """
        vf = self.get_vol_features(symbol)
        if vf is None:
            return None

        # 1. Regime assessment (V4 adaptive thresholds)
        regime = self.RegimeFilter.assess(
            rv20=vf["rv20"],
            rv_acceleration_3d=vf["rv_acceleration_3d"],
            iv_expansion_2d=vf["iv_expansion_2d"],
            config=self.config,
            rv_low_thresh=vf["rv_low_thresh"],
            rv_high_thresh=vf["rv_high_thresh"],
        )

        if not regime.allowed:
            return None

        # 2. Tail risk filter
        tail_scale, tail_reason = self.TailRiskFilter.compute_multiplier(
            iv_expansion_2d=vf["iv_expansion_2d"],
            rv_acceleration_3d=vf["rv_acceleration_3d"],
            gap_move=vf["gap_move"],
            config=self.config,
        )

        # 3. Estimate lot sizing
        lot_qty = 75 if symbol == "NIFTY" else 30  # standard lot sizes
        corr_scale = 0.70 if (active_symbols and len(active_symbols) > 0) else 1.0

        lots, position_scale, margin_required = self.PositionSizer.compute_lots(
            equity=equity,
            spot=spot,
            lot_quantity=lot_qty,
            rv20=vf["rv20"],
            vov=vf["vov"],
            regime_multiplier=regime.size_multiplier,
            signal_multiplier=regime.signal_multiplier,
            tail_scale=tail_scale,
            streak_mult=1.0,
            corr_scale=corr_scale,
            used_margin=used_margin,
            config=self.config,
            current_drawdown_pct=current_dd_pct,
            regime=regime.regime.value,
        )

        if lots <= 0:
            return None

        # 4. Estimate ATM premium (IV-based)
        dte = 3  # T-3 DTE entry
        T = dte / 365.0
        iv = max(vf["iv"], 0.05)
        atm_premium = spot * iv * np.sqrt(T) * 0.8  # rough straddle price approx

        return {
            "action": "ENTRY",
            "symbol": symbol,
            "strike": round(spot / 50) * 50,
            "premium": round(atm_premium, 2),
            "lots": lots,
            "lot_qty": lot_qty,
            "regime": regime.regime.value,
            "regime_multiplier": regime.size_multiplier,
            "rv20": vf["rv20"],
            "iv": vf["iv"],
            "tail_scale": tail_scale,
            "position_scale": position_scale,
            "margin_required": margin_required,
            "quality_score": 5.0,  # placeholder for paper mode
            "rationale": f"{regime.reason} | tail {tail_reason}",
        }


# ── Paper Trader Main Loop ─────────────────────────────────────────────

class PaperTrader:
    """
    Main paper trading orchestrator.
    Ties together: Feed → Strategy → Execution → Position Manager → Logger
    """

    def __init__(
        self,
        strategy_path: str,
        data_lake_path: str,
        feed_mode: str = "simulated",
        initial_capital: float = 10_000_000,
        tick_interval: float = 5.0,
        # Angel One credentials (only for live mode)
        api_key: str = "",
        client_code: str = "",
        password: str = "",
        totp_secret: str = "",
    ):
        self.tick_interval = tick_interval
        self._running = False

        # Components
        self.logger = PaperLogger()
        self.strategy = StrategyWrapper(strategy_path)
        if self.strategy is None:
            raise RuntimeError("[FATAL] Strategy initialization failed: self.strategy is None")
        self.engine = ExecutionEngine(self.logger)
        self.positions = PositionManager(
            execution_engine=self.engine,
            logger=self.logger,
            initial_capital=initial_capital,
        )
        self.feed = create_feed(
            mode=feed_mode,
            data_lake_path=data_lake_path,
            api_key=api_key,
            client_code=client_code,
            password=password,
            totp_secret=totp_secret,
            symbols=list(self.strategy.config.symbols),
        )

        # Separate feature engines per symbol (prevents price series contamination)
        self.feature_engines: Dict[str, FeatureEngine] = {
            symbol: FeatureEngine(maxlen=500)
            for symbol in self.strategy.config.symbols
        }
        self._feature_ready_logged: Dict[str, bool] = {
            symbol: False for symbol in self.strategy.config.symbols
        }

        # Telegram (may be injected later by LiveOperator)
        self.telegram = None

        # State
        self._last_signal_check = datetime.min
        self._signal_cooldown_seconds = max(tick_interval * 2, 10)
        self._expiry_date = self._next_weekly_expiry()
        self._live_signal_positions: Dict[str, Dict] = {}
        self.state_path = ROOT / "quant_repo" / "live" / "state.json"

        self.load_state()
        self._live_signal_positions = {
            p.symbol: True for p in self.positions.positions
        }

    @staticmethod
    def _next_weekly_expiry() -> datetime:
        """Estimate next Thursday expiry."""
        now = datetime.now()
        days_ahead = (3 - now.weekday()) % 7  # Thursday = 3
        if days_ahead == 0 and now.hour >= 15:
            days_ahead = 7
        return (now + timedelta(days=days_ahead)).replace(
            hour=15, minute=30, second=0, microsecond=0
        )

    def has_active_position(self, symbol: str) -> bool:
        """Check if there is an active position for a given symbol."""
        return any(pos.symbol == symbol for pos in self.positions.positions)

    def _sync_live_signal_positions(self):
        self._live_signal_positions = {
            p.symbol: True for p in self.positions.positions
        }

    def save_state(self):
        state = {
            "equity": self.positions.equity,
            "closed_pnl": self.positions.closed_pnl,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as state_file:
            json.dump(state, state_file)

    def load_state(self):
        try:
            with self.state_path.open("r", encoding="utf-8") as state_file:
                state = json.load(state_file)
            self.positions.closed_pnl = float(state.get("closed_pnl", self.positions.closed_pnl))
            self.positions.peak_equity = max(self.positions.peak_equity, float(state.get("equity", self.positions.equity)))
        except FileNotFoundError:
            print("[STATE] No previous state found")
        except Exception as exc:
            print(f"[STATE] Failed to load state: {exc}")

    def _close_symbol_positions(self, symbol: str, reason: str):
        """Close all active positions for a specific symbol."""
        to_close = [p for p in list(self.positions.positions) if p.symbol == symbol]
        for pos in to_close:
            self.positions.close_position(pos, reason, pos.current_price)
        if to_close:
            self.save_state()
            self._sync_live_signal_positions()

    @staticmethod
    def _format_days_held(entry_date: datetime, now_ts: datetime) -> str:
        held = max(now_ts - entry_date, timedelta(0))
        days = held.days
        hours = held.seconds // 3600
        return f"{days}d {hours}h"

    # ── Tick handler ────────────────────────────────────────────────────

    def _on_tick(self, price_or_tick, timestamp: Optional[datetime] = None):
        """Called on every incoming tick from the feed."""
        try:
            tick_obj = price_or_tick if isinstance(price_or_tick, Tick) else None
            if tick_obj is not None:
                if tick_obj.is_option:
                    return  # Option ticks update snapshot but don't trigger signals
                price = tick_obj.ltp
                timestamp = tick_obj.timestamp
                symbol = tick_obj.symbol

                # Keep existing strategy price buffer updates for entry checks.
                self.strategy.update_price(symbol, price)
            else:
                price = float(price_or_tick)
                timestamp = timestamp or datetime.now()
                symbol = None  # Cannot identify symbol without Tick object

            if not symbol:
                self.logger.error("[ERROR] Tick has no symbol identifier")
                return

            self.logger.info(f"[DEBUG] Tick received: {symbol} @ {price}")

            # Route tick to symbol-specific feature engine
            if symbol not in self.feature_engines:
                self.logger.error(f"[ERROR] Unknown symbol: {symbol}")
                return

            feature_engine = self.feature_engines[symbol]
            feature_engine.update(price, timestamp)

            # Check if ready for this symbol
            if not feature_engine.is_ready():
                buf_len = len(feature_engine.prices)
                if buf_len % 5 == 0 or buf_len == 1:
                    self.logger.info(f"[FEATURE] {symbol} warming up... ({buf_len}/30)")
                self.logger.info(f"[BLOCKED] {symbol} feature engine not ready")
                return

            # One-time ready announcement per symbol
            if not self._feature_ready_logged[symbol]:
                self._feature_ready_logged[symbol] = True
                self.logger.info(f"[FEATURE] ✅ {symbol} feature engine ready — regime detection active")
                if self.telegram and hasattr(self.telegram, "send"):
                    try:
                        self.telegram.send(f"✅ {symbol} feature engine ready — regime detection active")
                    except Exception:
                        pass

            features = feature_engine.compute_features()
            if not features:
                self.logger.info(f"[BLOCKED] {symbol} features missing")
                return

            self.logger.info(f"[DEBUG] {symbol} features ready — calling strategy")

            symbol_active = self.has_active_position(symbol)
            position_ctx = self._live_signal_positions.get(symbol)
            position_age_seconds = None
            position_pnl_pct = None
            days_to_expiry = None
            symbol_positions = [p for p in self.positions.positions if p.symbol == symbol]
            if symbol_positions:
                oldest_pos = min(symbol_positions, key=lambda p: p.entry_date)
                position_pnl_pct = oldest_pos.pnl_pct
                days_to_expiry = oldest_pos.dte
                position_age_seconds = max(0.0, (timestamp - oldest_pos.entry_date).total_seconds())

            # CRITICAL FIX: call strategy with symbol and symbol-specific features
            if hasattr(self.strategy, "on_tick"):
                raw_signal = self.strategy.on_tick(
                    symbol=symbol,
                    price=price,
                    features=features,
                    timestamp=timestamp,
                    has_position=symbol_active,
                    position_age_seconds=position_age_seconds,
                    position_pnl_pct=position_pnl_pct,
                    days_to_expiry=days_to_expiry,
                )

                # Normalize heterogeneous strategy outputs to a single signal string.
                signal = raw_signal
                if isinstance(raw_signal, dict):
                    signal = raw_signal.get("signal")
                elif isinstance(raw_signal, tuple):
                    signal = raw_signal[0] if raw_signal else None

                # Carry-forward HOLD log while position remains active.
                if signal in (None, "HOLD") and symbol_active:
                    oldest_pos = min(symbol_positions, key=lambda p: p.entry_date) if symbol_positions else None
                    if oldest_pos:
                        held = self._format_days_held(oldest_pos.entry_date, timestamp)
                        self.logger.info(
                            f"[HOLD] {symbol} position continues | Held {held} | "
                            f"PnL% {oldest_pos.pnl_pct * 100:.2f} | DTE {oldest_pos.dte}"
                        )
                    return

                print(f"[STATE CHECK] {symbol} active={self.has_active_position(symbol)}")

                if signal:
                    print(f"[SIGNAL RECEIVED] {symbol} -> {signal}")
                else:
                    return

                # Carry-forward execution bridge: while active, only EXIT is actionable.
                if signal == "ENTRY":
                    if self.has_active_position(symbol):
                        print(f"[BLOCKED] {symbol} already has real active position")
                        self.logger.info(f"[BLOCKED] {symbol} already has active position; duplicate ENTRY ignored")
                        return

                if signal == "ENTRY" and not symbol_active:
                    self.logger.info(f"[ORDER] {symbol} ENTRY executed")
                    self._live_signal_positions[symbol] = True

                    lot_qty = 75 if symbol == "NIFTY" else 30
                    strike = round(price / 50) * 50
                    premium = max(price * 0.01, 10.0)
                    opened = self.positions.open_position(
                        symbol=symbol,
                        strike=strike,
                        premium=premium,
                        lots=1,
                        lot_qty=lot_qty,
                        expiry_date=self._expiry_date,
                        regime="LIVE_SIGNAL",
                        quality_score=1.0,
                        position_scale=1.0,
                    )
                    if opened is not None:
                        self._sync_live_signal_positions()
                        self.save_state()
                elif signal == "ENTRY" and self.has_active_position(symbol):
                    self.logger.info(f"[BLOCKED] {symbol} already has active position; duplicate ENTRY ignored")
                    return

                if signal == "EXIT" and self.has_active_position(symbol):
                    self.logger.info(f"[EXIT] Closing {symbol} position from live signal")
                    self._close_symbol_positions(symbol, f"Live strategy EXIT for {symbol}")
                    self._sync_live_signal_positions()
            else:
                self.logger.error("[FATAL] Strategy has no on_tick method")

        except Exception as e:
            self.logger.error(f"[FATAL] Tick processing error: {e}")

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self):
        """Start the paper trading system."""
        self.logger.info("=" * 60)
        self.logger.info("PAPER TRADER STARTING")
        self.logger.info(f"Strategy: {self.strategy.strategy_path}")
        self.logger.info(f"Capital : Rs {self.positions.initial_capital:,.0f}")
        self.logger.info(f"Mode    : {type(self.feed).__name__}")
        self.logger.info(f"Interval: {self.tick_interval}s")
        self.logger.info("=" * 60)

        # Register tick callback
        self.feed.register_callback(self._on_tick)

        # Start feed in background
        self.feed.start()

        # Allow feed to populate initial ticks
        time.sleep(3)

        self._running = True

        # Graceful shutdown on Ctrl+C
        def _signal_handler(sig, frame):
            self.logger.info("Shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        try:
            self._main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _main_loop(self):
        """Core trading loop."""
        while self._running:
            try:
                snap = self.feed.snapshot

                # 1. Update position prices from feed
                self._update_position_prices(snap)

                # 2. Check exits on all open positions
                self._check_exits(snap)

                # 3. Check for new entry signals
                self._check_entries(snap)

                # 4. Record PnL snapshot
                self._record_pnl(snap)

                # 5. Print dashboard
                self._print_dashboard(snap)

                # 6. Sleep
                time.sleep(self.tick_interval)

            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(self.tick_interval)

    def _update_position_prices(self, snap: MarketSnapshot):
        """Update current prices for all open positions from feed."""
        for pos in self.positions.positions:
            # In paper mode, simulate a slight random walk around entry
            # In live mode, this would use real option LTP from feed
            if snap.nifty_spot > 0 or snap.banknifty_spot > 0:
                # Approximate straddle mark price based on spot movement
                spot = (
                    snap.nifty_spot if pos.symbol == "NIFTY"
                    else snap.banknifty_spot
                )
                if spot > 0:
                    move_pct = abs(spot - pos.strike) / pos.strike
                    # Rough straddle repricing: premium increases with distance from ATM
                    time_decay = max(0.0, 1.0 - (1.0 / max(pos.dte, 1)) * 0.3)
                    pos.current_price = pos.entry_premium * (
                        1.0 + move_pct * 2.5 - (1.0 - time_decay) * 0.2
                    )

    def _check_exits(self, snap: MarketSnapshot):
        """Run exit checks on all positions."""
        before_closed = self.positions.closed_pnl
        prices = {}
        for pos in self.positions.positions:
            key = f"{pos.symbol}_{int(pos.strike)}"
            prices[key] = pos.current_price

        self.positions.check_exits(prices)
        if self.positions.closed_pnl != before_closed:
            self.save_state()
            self._sync_live_signal_positions()

    def _check_entries(self, snap: MarketSnapshot):
        """Generate entry signals from strategy."""
        now = datetime.now()

        # ── MARKET HOURS GUARD ──────────────────────────────────────
        if not is_market_open():
            return  # silently skip — logged at operator level

        # Signal cooldown
        if (now - self._last_signal_check).seconds < self._signal_cooldown_seconds:
            return

        self._last_signal_check = now

        if self.positions.kill_switch_active:
            return

        active_symbols = [p.symbol for p in self.positions.positions]

        for symbol in self.strategy.config.symbols:
            if symbol in active_symbols:
                continue

            # Log feature engine status for this symbol
            features = self.feature_engines[symbol].compute_features()
            if features:
                self.logger.info(
                    f"[FEATURE] {symbol} Buffer={len(self.feature_engines[symbol].prices)} "
                    f"RV20={features['rv20']:.4f}"
                )

            spot = (
                snap.nifty_spot if symbol == "NIFTY"
                else snap.banknifty_spot
            )
            if spot <= 0:
                continue

            # Log strategy vol features for this symbol
            vf = self.strategy.get_vol_features(symbol)
            if vf is None:
                buf_len = len(self.strategy._price_buffers.get(symbol, []))
                self.logger.info(
                    f"[FEATURE] {symbol} strategy buffer warming up ({buf_len}/25)"
                )
                continue

            signal = self.strategy.generate_signal(
                symbol=symbol,
                spot=spot,
                equity=self.positions.equity,
                used_margin=sum(
                    p.entry_premium * p.lots * p.lot_qty
                    for p in self.positions.positions
                ),
                current_dd_pct=self.positions.drawdown_pct,
                active_symbols=active_symbols,
            )

            if signal is None and not active_symbols and spot > 0:
                lot_qty = 75 if symbol == "NIFTY" else 30
                signal = {
                    "action": "ENTRY",
                    "symbol": symbol,
                    "strike": round(spot / 50) * 50,
                    "premium": max(spot * 0.01, 10.0),
                    "lots": 1,
                    "lot_qty": lot_qty,
                    "regime": "FALLBACK",
                    "regime_multiplier": 1.0,
                    "rv20": vf["rv20"],
                    "iv": vf["iv"],
                    "tail_scale": 1.0,
                    "position_scale": 1.0,
                    "margin_required": spot * self.strategy.config.margin_per_lot_pct * lot_qty,
                    "quality_score": 1.0,
                    "rationale": "Fallback entry when strategy returns no signal",
                }

            # Log regime regardless of signal
            self.logger.info(
                f"[REGIME] {symbol} | RV20={vf['rv20']:.4f} | "
                f"Regime={'LOW' if vf['rv20'] < vf['rv_low_thresh'] else ('HIGH' if vf['rv20'] > vf['rv_high_thresh'] else 'NORMAL')}"
            )

            if signal is not None:
                self.logger.info(
                    f"[SIGNAL] {symbol} | {signal['regime']} | "
                    f"Strike {signal['strike']:.0f} | "
                    f"Premium ~{signal['premium']:.2f} | "
                    f"Lots {signal['lots']}"
                )

                self.positions.open_position(
                    symbol=signal["symbol"],
                    strike=signal["strike"],
                    premium=signal["premium"],
                    lots=signal["lots"],
                    lot_qty=signal["lot_qty"],
                    expiry_date=self._expiry_date,
                    regime=signal["regime"],
                    quality_score=signal["quality_score"],
                    position_scale=signal["position_scale"],
                )
                self.save_state()

    def _record_pnl(self, snap: MarketSnapshot):
        """Log PnL snapshot to CSV."""
        regime = "?"
        for sym in self.strategy.config.symbols:
            vf = self.strategy.get_vol_features(sym)
            if vf:
                rv = vf["rv20"]
                lo = vf["rv_low_thresh"]
                hi = vf["rv_high_thresh"]
                if rv < lo:
                    regime = "LOW VOL"
                elif rv <= hi:
                    regime = "NORMAL VOL"
                else:
                    regime = "HIGH VOL"
                break

        self.logger.record_pnl({
            "equity": round(self.positions.equity, 2),
            "open_pnl": round(self.positions.open_pnl, 2),
            "closed_pnl": round(self.positions.closed_pnl, 2),
            "drawdown_pct": round(self.positions.drawdown_pct, 2),
            "active_positions": self.positions.active_position_count,
            "trades_today": self.positions.trades_today,
            "regime": regime,
            "daily_return_pct": round(self.positions.daily_return_pct, 4),
        })

    def _print_dashboard(self, snap: MarketSnapshot):
        """Render console dashboard."""
        regime = "?"
        for sym in self.strategy.config.symbols:
            vf = self.strategy.get_vol_features(sym)
            if vf:
                rv = vf["rv20"]
                lo = vf["rv_low_thresh"]
                hi = vf["rv_high_thresh"]
                if rv < lo:
                    regime = "LOW VOL"
                elif rv <= hi:
                    regime = "NORMAL VOL"
                else:
                    regime = "HIGH VOL"
                break

        self.logger.print_dashboard(
            equity=self.positions.equity,
            open_pnl=self.positions.open_pnl,
            closed_pnl=self.positions.closed_pnl,
            drawdown_pct=self.positions.drawdown_pct,
            positions=self.positions.get_position_summaries(),
            trades_today=self.positions.trades_today,
            regime=regime,
            daily_return_pct=self.positions.daily_return_pct,
        )

    def _shutdown(self):
        """Clean shutdown."""
        self.logger.info("Shutting down paper trader...")
        self.feed.stop()
        if self.positions.positions:
            self.logger.info(
                f"Carry-forward mode active: keeping {len(self.positions.positions)} open position(s) "
                f"(no shutdown auto-close)."
            )
            self.save_state()
        self.logger.info(
            f"Final equity: Rs {self.positions.equity:,.0f} | "
            f"Closed PnL: Rs {self.positions.closed_pnl:+,.0f} | "
            f"Total fills: {self.engine.total_fills}"
        )
        self.logger.info("Paper trader stopped.")


# ── CLI Entry Point ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paper Trader for V4 Strategy")
    parser.add_argument(
        "--mode", default="simulated", choices=["simulated", "live"],
        help="Feed mode (default: simulated)",
    )
    parser.add_argument(
        "--strategy",
        default=str(ROOT / "quant_repo/strategies/short_vol/v4_regime_generalized.py"),
        help="Path to strategy module",
    )
    parser.add_argument(
        "--lake",
        default=str(ROOT / "data/master_fo_lake"),
        help="Data lake path (for simulated mode)",
    )
    parser.add_argument(
        "--capital", type=float, default=10_000_000,
        help="Initial capital (default: 10M)",
    )
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="Tick interval in seconds (default: 5)",
    )
    parser.add_argument("--api-key", default=os.environ.get("ANGEL_API_KEY", ""))
    parser.add_argument("--client-code", default=os.environ.get("ANGEL_CLIENT_CODE", ""))
    parser.add_argument("--password", default=os.environ.get("ANGEL_PASSWORD", ""))
    parser.add_argument("--totp-secret", default=os.environ.get("ANGEL_TOTP_SECRET", ""))

    args = parser.parse_args()

    trader = PaperTrader(
        strategy_path=args.strategy,
        data_lake_path=args.lake,
        feed_mode=args.mode,
        initial_capital=args.capital,
        tick_interval=args.interval,
        api_key=args.api_key,
        client_code=args.client_code,
        password=args.password,
        totp_secret=args.totp_secret,
    )

    trader.run()


if __name__ == "__main__":
    main()
