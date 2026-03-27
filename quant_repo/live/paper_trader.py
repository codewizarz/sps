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
from quant_repo.live.position_manager import (
    PaperPosition, PositionManager, RiskLimits,
)


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

        # Rolling price buffer for vol computation
        self._price_buffers: Dict[str, List[float]] = {}
        self._vol_cache: Dict[str, Dict] = {}

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

        # Feature engine (diagnostic companion — independent RV20 logging)
        self.feature_engine = FeatureEngine(maxlen=500)
        self._feature_ready_logged = False

        # Telegram (may be injected later by LiveOperator)
        self.telegram = None

        # State
        self._last_signal_check = datetime.min
        self._signal_cooldown_seconds = max(tick_interval * 2, 10)
        self._expiry_date = self._next_weekly_expiry()

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

    # ── Tick handler ────────────────────────────────────────────────────

    def _on_tick(self, tick: Tick):
        """Called on every incoming tick from the feed."""
        if tick.is_option:
            return  # Option ticks update snapshot but don't trigger signals

        # Update strategy price buffer
        self.strategy.update_price(tick.symbol, tick.ltp)

        # Update diagnostic feature engine
        self.feature_engine.update(tick.ltp, tick.timestamp)

        if not self.feature_engine.is_ready():
            # Log warmup progress every 5 ticks to avoid spam
            buf_len = len(self.feature_engine.prices)
            if buf_len % 5 == 0 or buf_len == 1:
                self.logger.info(f"[FEATURE] Warming up... ({buf_len}/30)")
            return

        # One-time Telegram alert when feature engine becomes ready
        if not self._feature_ready_logged:
            self._feature_ready_logged = True
            self.logger.info("[FEATURE] ✅ Feature engine ready — regime detection active")
            if self.telegram and hasattr(self.telegram, 'send'):
                try:
                    self.telegram.send("✅ Feature engine ready — regime detection active")
                except Exception:
                    pass

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
        prices = {}
        for pos in self.positions.positions:
            key = f"{pos.symbol}_{int(pos.strike)}"
            prices[key] = pos.current_price

        self.positions.check_exits(prices)

    def _check_entries(self, snap: MarketSnapshot):
        """Generate entry signals from strategy."""
        now = datetime.now()

        # Signal cooldown
        if (now - self._last_signal_check).seconds < self._signal_cooldown_seconds:
            return

        self._last_signal_check = now

        if self.positions.kill_switch_active:
            return

        active_symbols = [p.symbol for p in self.positions.positions]

        # Log feature engine status
        features = self.feature_engine.compute_features()
        if features:
            self.logger.info(
                f"[FEATURE] Buffer={len(self.feature_engine.prices)} "
                f"RV20={features['rv20']:.4f}"
            )

        for symbol in self.strategy.config.symbols:
            if symbol in active_symbols:
                continue

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
            self.logger.info(f"Closing {len(self.positions.positions)} open positions...")
            self.positions.close_all("Shutdown")
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
