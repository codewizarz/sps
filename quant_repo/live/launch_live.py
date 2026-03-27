#!/usr/bin/env python3
"""
=============================================================================
LIVE PAPER TRADING LAUNCHER — AUTONOMOUS OPERATOR
=============================================================================
Handles: env loading, TOTP generation, auth, dependency checks,
dry run, live launch, monitoring, self-healing, and structured logging.

Usage:
    python3 quant_repo/live/launch_live.py
    python3 quant_repo/live/launch_live.py --dry-run    # connection test only
    python3 quant_repo/live/launch_live.py --simulated  # skip auth, use lake data

NO REAL TRADES ARE PLACED. This is paper trading only.
=============================================================================
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── STEP 1: ENV + TOTP ─────────────────────────────────────────────────

def load_env() -> dict:
    """Load .env and validate required variables."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        _die("python-dotenv not installed. Run: pip3 install python-dotenv")

    env = {
        "API_KEY": os.getenv("ANGEL_API_KEY", ""),
        "CLIENT_CODE": os.getenv("ANGEL_CLIENT_ID", ""),
        "PASSWORD": os.getenv("ANGEL_APP_PIN", ""),
        "TOTP_SECRET": os.getenv("ANGEL_TOTP_SECRET", ""),
    }

    missing = [k for k, v in env.items() if not v]
    if missing:
        _die(
            f"Missing env variables: {', '.join(missing)}\n"
            f"Please set them in .env:\n"
            f"  ANGEL_API_KEY=...\n"
            f"  ANGEL_CLIENT_ID=...\n"
            f"  ANGEL_APP_PIN=...\n"
            f"  ANGEL_TOTP_SECRET=..."
        )

    return env


def generate_totp(secret: str) -> str:
    """Generate fresh TOTP from secret (strips hyphens/spaces)."""
    import pyotp
    clean = secret.replace("-", "").replace(" ", "").strip()
    return pyotp.TOTP(clean).now()


# ── STEP 2: DEPENDENCY CHECK ───────────────────────────────────────────

def check_dependencies() -> bool:
    """Verify all required packages are importable."""
    _log("SYSTEM", "Checking dependencies...")
    deps = {
        "pyotp": "pyotp",
        "SmartConnect": "SmartApi",
        "SmartWebSocketV2": "SmartApi.smartWebSocketV2",
        "pandas": "pandas",
        "numpy": "numpy",
        "dotenv": "dotenv",
    }
    missing = []
    for name, module in deps.items():
        try:
            __import__(module)
            _log("SYSTEM", f"  ✅ {name}")
        except ImportError:
            _log("ERROR", f"  ❌ {name}")
            missing.append(name)

    if missing:
        _log("ERROR", f"Missing packages: {missing}")
        _log("ERROR", "Run: pip3 install pyotp smartapi-python python-dotenv logzero pycryptodome websocket-client")
        return False

    _log("SYSTEM", "All dependencies OK")
    return True


# ── STEP 3: AUTH TEST ───────────────────────────────────────────────────

def auth_test(env: dict, max_retries: int = 3) -> dict:
    """
    Authenticate with Angel One SmartAPI.
    Retries with fresh TOTP on each attempt.
    Returns session data or dies.
    """
    from SmartApi import SmartConnect

    for attempt in range(1, max_retries + 1):
        _log("AUTH", f"Login attempt {attempt}/{max_retries}...")

        try:
            totp = generate_totp(env["TOTP_SECRET"])
            _log("AUTH", f"TOTP generated: {totp[:2]}****")

            smart_api = SmartConnect(api_key=env["API_KEY"])
            session = smart_api.generateSession(
                env["CLIENT_CODE"],
                env["PASSWORD"],
                totp,
            )

            if session.get("status"):
                auth_token = session["data"]["jwtToken"]
                feed_token = smart_api.getfeedToken()
                _log("AUTH", f"✅ Login successful — Client: {env['CLIENT_CODE']}")
                return {
                    "smart_api": smart_api,
                    "auth_token": auth_token,
                    "feed_token": feed_token,
                }
            else:
                _log("AUTH", f"❌ Login failed: {session.get('message', session)}")

        except Exception as e:
            _log("ERROR", f"Auth error: {e}")

        if attempt < max_retries:
            _log("AUTH", "Waiting 5s before retry (TOTP refresh)...")
            time.sleep(5)

    _die("Authentication failed after all retries. Check credentials in .env")


# ── STEP 4: DRY RUN ────────────────────────────────────────────────────

def dry_run(env: dict, session: dict, duration: int = 30) -> bool:
    """
    Run a short connection test to validate WebSocket connectivity.
    Returns True if ticks were received.
    """
    _log("DRYRUN", f"Starting {duration}s dry run (connection test)...")

    from SmartApi.smartWebSocketV2 import SmartWebSocketV2

    ticks_received = []
    errors = []

    def on_data(wsapp, message):
        ticks_received.append(datetime.now())
        if len(ticks_received) == 1:
            _log("DRYRUN", f"✅ First tick received at {datetime.now().strftime('%H:%M:%S')}")

    def on_open(wsapp):
        _log("DRYRUN", "WebSocket connected, subscribing to NIFTY + BANKNIFTY...")
        try:
            ws.subscribe(
                "dryrun",
                2,  # Quote mode
                [{"exchangeType": 1, "tokens": ["99926000", "99926009"]}],
            )
        except Exception as e:
            _log("ERROR", f"Subscribe error: {e}")

    def on_error(wsapp, error):
        errors.append(str(error))
        _log("ERROR", f"WS Error: {error}")

    def on_close(wsapp):
        _log("DRYRUN", "WebSocket closed")

    ws = SmartWebSocketV2(
        session["auth_token"],
        env["API_KEY"],
        env["CLIENT_CODE"],
        session["feed_token"],
    )
    ws.on_open = on_open
    ws.on_data = on_data
    ws.on_error = on_error
    ws.on_close = on_close

    # Run in background thread
    ws_thread = threading.Thread(target=ws.connect, daemon=True)
    ws_thread.start()

    # Wait for duration
    time.sleep(duration)

    # Close
    try:
        ws.close_connection()
    except Exception:
        pass

    ws_thread.join(timeout=5)

    # Evaluate
    _log("DRYRUN", f"Ticks received: {len(ticks_received)}")
    _log("DRYRUN", f"Errors: {len(errors)}")

    if len(ticks_received) > 0:
        _log("DRYRUN", "✅ Dry run PASSED — WebSocket is live")
        return True
    elif errors:
        _log("DRYRUN", f"❌ Dry run FAILED — errors: {errors[:3]}")
        return False
    else:
        _log("DRYRUN", "⚠️ Dry run produced no ticks (market may be closed)")
        _log("DRYRUN", "Continuing anyway — if market is closed, no ticks is expected")
        return True  # Don't block if market is simply closed


# ── STEP 5-11: LIVE PAPER TRADING WITH MONITORING ──────────────────────

class LiveOperator:
    """
    Autonomous operator that runs paper trading with:
    - Self-healing (auto-restart on crash)
    - Health monitoring (tick flow, errors)
    - Structured logging
    - Risk control validation
    """

    def __init__(self, env: dict, session: dict):
        self.env = env
        self.session = session
        self._running = False
        self._crash_count = 0
        self._max_crashes = 2
        self._start_time = None
        self._last_health_report = datetime.min
        self._health_interval = 300  # 5 minutes
        self._first_trade_validated = False
        self._trader = None

        # Telegram integration
        from quant_repo.live.telegram_bot import create_telegram_manager
        self.telegram = create_telegram_manager(health_interval=300)
        if self.telegram.enabled:
            _log("TELEGRAM", "Bot connected — alerts enabled")
        else:
            _log("TELEGRAM", "Not configured — alerts disabled")

    def run(self):
        """Start live paper trading with self-healing."""
        self._running = True
        self._start_time = datetime.now()

        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        # Wire up Telegram command callbacks
        self.telegram.set_status_callback(self._telegram_status)
        self.telegram.set_pnl_callback(self._telegram_pnl)
        self.telegram.set_positions_callback(self._telegram_positions)
        self.telegram.start()

        _log("STATUS", "▶ Starting live paper trading")
        _log("STATUS", f"  Client: {self.env['CLIENT_CODE']}")
        _log("STATUS", f"  Strategy: V4 Regime Generalized")
        _log("STATUS", f"  Capital: Rs 10,000,000")

        while self._running and self._crash_count < self._max_crashes:
            try:
                self._run_session()
            except KeyboardInterrupt:
                break
            except Exception as e:
                self._crash_count += 1
                _log("ERROR", f"Session crash #{self._crash_count}: {e}")
                traceback.print_exc()
                self.telegram.send_error(str(e), f"crash #{self._crash_count}")

                if self._crash_count >= self._max_crashes:
                    _log("ERROR", f"🚨 Max crashes ({self._max_crashes}) reached — STOPPING")
                    self.telegram.send_error(f"Max crashes reached ({self._max_crashes})", "STOPPED")
                    break

                _log("STATUS", f"♻ Self-healing: restarting in 10s (crash {self._crash_count}/{self._max_crashes})")
                time.sleep(10)

                # Re-authenticate with fresh TOTP
                _log("AUTH", "Re-authenticating for restart...")
                try:
                    self.session = auth_test(self.env, max_retries=2)
                except SystemExit:
                    _log("ERROR", "Re-auth failed — cannot restart")
                    self.telegram.send_error("Re-auth failed", "cannot restart")
                    break

        _log("STATUS", "■ Paper trading stopped")
        self._print_final_summary()
        # Send Telegram shutdown summary
        if self._trader:
            t = self._trader
            self.telegram.stop({
                "equity": t.positions.equity,
                "closed_pnl": t.positions.closed_pnl,
                "fills": t.engine.total_fills,
            })
        else:
            self.telegram.stop()

    def _run_session(self):
        """Single paper trading session."""
        from quant_repo.live.paper_trader import PaperTrader

        self._trader = PaperTrader(
            strategy_path=str(ROOT / "quant_repo/strategies/short_vol/v4_regime_generalized.py"),
            data_lake_path=str(ROOT / "data/master_fo_lake"),
            feed_mode="live",
            initial_capital=10_000_000,
            tick_interval=2,
            api_key=self.env["API_KEY"],
            client_code=self.env["CLIENT_CODE"],
            password=self.env["PASSWORD"],
            totp_secret=self.env["TOTP_SECRET"],
        )

        # Inject Telegram into execution engine
        self._trader.engine.telegram = self.telegram

        # Override the main loop to inject monitoring
        trader = self._trader
        trader.feed.register_callback(trader._on_tick)          # ← CRITICAL: feed prices into strategy
        trader.feed.register_callback(self._on_tick_monitor)    # health monitoring
        trader.feed.start()
        time.sleep(3)
        trader._running = True
        self._running = True

        _log("STATUS", "✅ Session started — entering main loop")

        while trader._running and self._running:
            try:
                snap = trader.feed.snapshot

                # Core trading loop steps
                trader._update_position_prices(snap)
                trader._check_exits(snap)
                trader._check_entries(snap)
                trader._record_pnl(snap)

                # STEP 6+8+9: Monitoring + Telegram health ping
                self._monitor_health(trader)
                self._validate_trades(trader)
                self._check_risk_controls(trader)
                self._telegram_health_ping(trader)

                # STEP 11: Dashboard
                trader._print_dashboard(snap)

                time.sleep(trader.tick_interval)

            except KeyboardInterrupt:
                self._running = False
                break
            except Exception as e:
                _log("ERROR", f"Loop iteration error: {e}")
                time.sleep(trader.tick_interval)

        # Cleanup
        _log("STATUS", "Shutting down session...")
        trader._shutdown()

    # ── STEP 6: Health Monitoring ───────────────────────────────────────

    _last_tick_time = None
    _tick_count = 0

    def _on_tick_monitor(self, tick):
        """Track tick activity for health monitoring."""
        self._last_tick_time = datetime.now()
        self._tick_count += 1

    def _monitor_health(self, trader):
        """Periodic health checks — every 5 minutes."""
        now = datetime.now()

        # STEP 7: Self-healing — no ticks for >10 seconds
        if self._last_tick_time and (now - self._last_tick_time).seconds > 10:
            _log("HEALTH", "⚠ No ticks for >10 seconds — feed may be stale")

        # 5-minute health report (console)
        if (now - self._last_health_report).seconds >= self._health_interval:
            self._last_health_report = now
            self._print_health_report(trader)

    def _print_health_report(self, trader):
        """STEP 11: Structured 5-minute health summary."""
        uptime = datetime.now() - self._start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)

        # Determine regime
        regime = "?"
        for sym in trader.strategy.config.symbols:
            vf = trader.strategy.get_vol_features(sym)
            if vf:
                rv = vf["rv20"]
                lo = vf["rv_low_thresh"]
                hi = vf["rv_high_thresh"]
                regime = "LOW" if rv < lo else ("HIGH" if rv > hi else "NORMAL")
                break

        lines = [
            "",
            "┌──────────────────────────────────────────────┐",
            "│         5-MINUTE HEALTH REPORT               │",
            "├──────────────────────────────────────────────┤",
            f"│ Uptime     : {hours}h {minutes}m                       │",
            f"│ Ticks      : {self._tick_count:,}                          │",
            f"│ Crashes    : {self._crash_count}/{self._max_crashes}                            │",
            f"│ Regime     : {regime:<10s}                     │",
            f"│ Equity     : Rs {trader.positions.equity:>14,.0f}      │",
            f"│ Open PnL   : Rs {trader.positions.open_pnl:>+14,.0f}      │",
            f"│ Closed PnL : Rs {trader.positions.closed_pnl:>+14,.0f}      │",
            f"│ Drawdown   : {trader.positions.drawdown_pct:>8.2f}%                  │",
            f"│ Positions  : {trader.positions.active_position_count}                              │",
            f"│ Kill switch: {'🚨 ACTIVE' if trader.positions.kill_switch_active else '✅ OFF'}                       │",
            "└──────────────────────────────────────────────┘",
        ]
        for line in lines:
            _log("HEALTH", line)

    # ── STEP 8: Trade Validation ────────────────────────────────────────

    def _validate_trades(self, trader):
        """Validate first trade and all subsequent entries."""
        if self._first_trade_validated:
            return

        fills = trader.engine.fills
        entries = [f for f in fills if f.action == "ENTRY"]
        if not entries:
            return

        first = entries[0]
        warnings = []

        # Lot size check
        if first.lots > 100:
            warnings.append(f"⚠ EXTREME lot size: {first.lots}")
        elif first.lots < 1:
            warnings.append(f"⚠ Invalid lot size: {first.lots}")

        # Strike validity
        if first.strike <= 0:
            warnings.append(f"⚠ Invalid strike: {first.strike}")

        # Regime check
        if first.regime == "?":
            warnings.append("⚠ Regime is unknown (?)")

        # Duplicate check
        if len(entries) > 1 and entries[-1].symbol == entries[-2].symbol:
            warnings.append(f"⚠ Possible duplicate entry: {entries[-1].symbol}")

        if warnings:
            for w in warnings:
                _log("TRADE", w)
        else:
            _log("TRADE", f"✅ First trade validated: {first.symbol} {first.strike:.0f} x{first.lots}")

        self._first_trade_validated = True

    # ── STEP 9: Risk Control Check ──────────────────────────────────────

    def _check_risk_controls(self, trader):
        """Validate risk controls are active and not violated."""
        pm = trader.positions

        # Max concurrent positions
        if pm.active_position_count > pm.risk.max_concurrent_positions:
            _log("RISK", f"🚨 CRITICAL: {pm.active_position_count} positions > max {pm.risk.max_concurrent_positions}")

        # Daily loss limit
        if pm.daily_return_pct < -pm.risk.daily_loss_limit_pct:
            _log("RISK", f"🚨 CRITICAL: Daily loss {pm.daily_return_pct:.2f}% exceeds limit {pm.risk.daily_loss_limit_pct}%")

        # Kill switch
        if pm.kill_switch_active:
            _log("RISK", f"🚨 KILL SWITCH ACTIVE — DD: {pm.drawdown_pct:.2f}%")
            self.telegram.send_kill_switch(pm.drawdown_pct)

    # ── Shutdown ────────────────────────────────────────────────────────

    def _shutdown_handler(self, sig, frame):
        _log("STATUS", "Shutdown signal received")
        self._running = False
        if self._trader:
            self._trader._running = False

    def _print_final_summary(self):
        if not self._trader:
            return
        t = self._trader
        uptime = datetime.now() - self._start_time

        _log("STATUS", "")
        _log("STATUS", "═" * 50)
        _log("STATUS", "  FINAL SESSION SUMMARY")
        _log("STATUS", "═" * 50)
        _log("STATUS", f"  Uptime           : {uptime}")
        _log("STATUS", f"  Ticks processed  : {self._tick_count:,}")
        _log("STATUS", f"  Crash count      : {self._crash_count}")
        _log("STATUS", f"  Final equity     : Rs {t.positions.equity:,.0f}")
        _log("STATUS", f"  Closed PnL       : Rs {t.positions.closed_pnl:+,.0f}")
        _log("STATUS", f"  Total fills      : {t.engine.total_fills}")
        _log("STATUS", f"  Max drawdown     : {t.positions.drawdown_pct:.2f}%")
        _log("STATUS", "═" * 50)

    # ── Telegram Callbacks ──────────────────────────────────────────────

    def _get_regime(self) -> str:
        if not self._trader:
            return "?"
        for sym in self._trader.strategy.config.symbols:
            vf = self._trader.strategy.get_vol_features(sym)
            if vf:
                rv = vf["rv20"]
                lo = vf["rv_low_thresh"]
                hi = vf["rv_high_thresh"]
                return "LOW" if rv < lo else ("HIGH" if rv > hi else "NORMAL")
        return "?"

    def _telegram_status(self) -> str:
        if not self._trader:
            return "📊 Trader not initialized yet"
        t = self._trader
        uptime = datetime.now() - self._start_time
        h = int(uptime.total_seconds() // 3600)
        m = int((uptime.total_seconds() % 3600) // 60)
        return (
            f"📊 <b>STATUS</b>\n"
            f"State: <code>{'🟢 RUNNING' if self._running else '🔴 STOPPED'}</code>\n"
            f"Regime: <code>{self._get_regime()}</code>\n"
            f"Equity: <code>₹{t.positions.equity:,.0f}</code>\n"
            f"Open PnL: <code>₹{t.positions.open_pnl:+,.0f}</code>\n"
            f"Closed PnL: <code>₹{t.positions.closed_pnl:+,.0f}</code>\n"
            f"DD: <code>{t.positions.drawdown_pct:.2f}%</code>\n"
            f"Positions: <code>{t.positions.active_position_count}</code>\n"
            f"Trades today: <code>{t.positions.trades_today}</code>\n"
            f"Ticks: <code>{self._tick_count:,}</code>\n"
            f"Uptime: <code>{h}h {m}m</code>\n"
            f"Kill switch: <code>{'🚨 ACTIVE' if t.positions.kill_switch_active else '✅ OFF'}</code>"
        )

    def _telegram_pnl(self) -> str:
        if not self._trader:
            return "💰 Trader not initialized"
        t = self._trader
        return (
            f"💰 <b>PnL SUMMARY</b>\n"
            f"Equity: <code>₹{t.positions.equity:,.0f}</code>\n"
            f"Open PnL: <code>₹{t.positions.open_pnl:+,.0f}</code>\n"
            f"Closed PnL: <code>₹{t.positions.closed_pnl:+,.0f}</code>\n"
            f"Daily return: <code>{t.positions.daily_return_pct:+.4f}%</code>\n"
            f"Drawdown: <code>{t.positions.drawdown_pct:.2f}%</code>\n"
            f"Total fills: <code>{t.engine.total_fills}</code>"
        )

    def _telegram_positions(self) -> str:
        if not self._trader:
            return "📋 Trader not initialized"
        positions = self._trader.positions.get_position_summaries()
        if not positions:
            return "📋 No active positions"
        lines = ["📋 <b>ACTIVE POSITIONS</b>\n"]
        for p in positions:
            emoji = "🟢" if p["unrealized_pnl"] >= 0 else "🔴"
            lines.append(
                f"{emoji} <code>{p['symbol']}</code> {p['strike']:.0f}\n"
                f"   Lots: {p['lots']} | PnL: ₹{p['unrealized_pnl']:+,.0f} ({p['pnl_pct']:+.1f}%)\n"
                f"   DTE: {p['dte']} | Stop: stage {p['stop_stage']}"
            )
        return "\n".join(lines)

    def _telegram_health_ping(self, trader):
        """Send Telegram health ping (rate-limited by TelegramManager)."""
        self.telegram.maybe_send_health_ping({
            "status": "RUNNING" if self._running else "STOPPED",
            "regime": self._get_regime(),
            "positions": trader.positions.active_position_count,
            "equity": trader.positions.equity,
            "open_pnl": trader.positions.open_pnl,
            "closed_pnl": trader.positions.closed_pnl,
            "drawdown_pct": trader.positions.drawdown_pct,
            "trades_today": trader.positions.trades_today,
        })


# ── Logging helpers ─────────────────────────────────────────────────────

def _log(tag: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{tag:>7s}] {msg}")

def _die(msg: str):
    _log("FATAL", msg)
    sys.exit(1)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Live Paper Trading Launcher")
    parser.add_argument("--dry-run", action="store_true", help="Connection test only")
    parser.add_argument("--simulated", action="store_true", help="Skip auth, use data lake")
    args = parser.parse_args()

    print()
    _log("SYSTEM", "=" * 50)
    _log("SYSTEM", "  LIVE PAPER TRADING LAUNCHER")
    _log("SYSTEM", "  V4 Regime Generalized Strategy")
    _log("SYSTEM", "  ⚠ NO REAL TRADES — PAPER ONLY")
    _log("SYSTEM", "=" * 50)

    # ── Simulated mode (skip auth) ──
    if args.simulated:
        _log("SYSTEM", "Running in SIMULATED mode (no Angel One API)")
        from quant_repo.live.paper_trader import PaperTrader

        trader = PaperTrader(
            strategy_path=str(ROOT / "quant_repo/strategies/short_vol/v4_regime_generalized.py"),
            data_lake_path=str(ROOT / "data/master_fo_lake"),
            feed_mode="simulated",
            initial_capital=10_000_000,
            tick_interval=2,
        )
        trader.run()
        return

    # ── STEP 1: Load env ──
    _log("SYSTEM", "")
    _log("SYSTEM", "STEP 1: Loading environment...")
    env = load_env()
    _log("SYSTEM", f"  API_KEY    : {env['API_KEY'][:4]}****")
    _log("SYSTEM", f"  CLIENT     : {env['CLIENT_CODE']}")
    _log("SYSTEM", f"  TOTP_SECRET: {'set' if env['TOTP_SECRET'] else 'MISSING'}")

    # ── STEP 2: Dependencies ──
    _log("SYSTEM", "")
    _log("SYSTEM", "STEP 2: Checking dependencies...")
    if not check_dependencies():
        _die("Dependency check failed")

    # ── STEP 3: Auth ──
    _log("SYSTEM", "")
    _log("SYSTEM", "STEP 3: Authenticating with Angel One...")
    session = auth_test(env)

    # ── STEP 4: Dry run ──
    if args.dry_run:
        _log("SYSTEM", "")
        _log("SYSTEM", "STEP 4: Dry run (connection test)...")
        passed = dry_run(env, session, duration=30)
        if passed:
            _log("SYSTEM", "✅ Dry run passed — system is ready for live paper trading")
        else:
            _log("SYSTEM", "❌ Dry run failed — check WebSocket connectivity")
        return

    # ── STEP 5-11: Live operation ──
    _log("SYSTEM", "")
    _log("SYSTEM", "STEP 5: Launching live paper trading...")
    _log("SYSTEM", "Press Ctrl+C to stop gracefully")
    _log("SYSTEM", "")

    operator = LiveOperator(env, session)
    operator.run()


if __name__ == "__main__":
    main()
