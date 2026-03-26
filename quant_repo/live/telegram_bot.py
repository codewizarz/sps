#!/usr/bin/env python3
"""
=============================================================================
PAPER TRADING — TELEGRAM BOT
=============================================================================
Real-time alerts + remote control via Telegram.

Features:
  - Trade alerts (entry/exit)
  - Error alerts (auth, WS, crash)
  - Health pings (every 5 min)
  - Remote commands: /status /pnl /logs /restart /stop /start /positions

Uses raw HTTP API (no heavyweight bot framework needed at runtime).
Command listener runs in a background thread — never blocks trading.
=============================================================================
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests


# ── Telegram API Client ─────────────────────────────────────────────────

class TelegramClient:
    """Lightweight Telegram Bot API wrapper using raw HTTP."""

    API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)
        self._last_send: Dict[str, datetime] = {}
        self._rate_limit_seconds = 2  # min gap between messages

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a text message. Returns True on success."""
        if not self._enabled:
            return False

        # Rate limit
        now = datetime.now()
        last = self._last_send.get("msg", datetime.min)
        if (now - last).total_seconds() < self._rate_limit_seconds:
            time.sleep(self._rate_limit_seconds)

        try:
            url = self.API.format(token=self.bot_token, method="sendMessage")
            resp = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": text[:4096],  # Telegram limit
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            self._last_send["msg"] = datetime.now()
            return resp.status_code == 200
        except Exception:
            return False

    def get_updates(self, offset: int = 0, timeout: int = 10) -> List[Dict]:
        """Long-poll for new messages/commands."""
        if not self._enabled:
            return []
        try:
            url = self.API.format(token=self.bot_token, method="getUpdates")
            resp = requests.get(
                url,
                params={"offset": offset, "timeout": timeout},
                timeout=timeout + 5,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("result", [])
        except Exception:
            pass
        return []


# ── Alert Formatters ────────────────────────────────────────────────────

class AlertFormatter:
    """Formats trading events into Telegram-friendly messages."""

    @staticmethod
    def trade_entry(trade: Dict) -> str:
        return (
            "📈 <b>ENTRY</b>\n"
            f"Symbol: <code>{trade.get('symbol', '?')}</code>\n"
            f"Strike: <code>{trade.get('strike', 0):.0f}</code>\n"
            f"Lots: <code>{trade.get('lots', 0)} × {trade.get('lot_qty', 0)}</code>\n"
            f"Premium: <code>₹{trade.get('premium', 0):.2f}</code>\n"
            f"Fill: <code>₹{trade.get('fill_price', 0):.2f}</code>\n"
            f"Regime: <code>{trade.get('regime', '?')}</code>\n"
            f"Scale: <code>{trade.get('position_scale', 0):.2f}</code>"
        )

    @staticmethod
    def trade_exit(trade: Dict) -> str:
        pnl = trade.get("net_pnl", 0)
        emoji = "🟢" if pnl >= 0 else "🔴"
        return (
            f"📉 <b>EXIT</b> {emoji}\n"
            f"Symbol: <code>{trade.get('symbol', '?')}</code>\n"
            f"Strike: <code>{trade.get('strike', 0):.0f}</code>\n"
            f"Reason: <code>{trade.get('exit_reason', '?')}</code>\n"
            f"PnL: <code>₹{pnl:+,.0f}</code>\n"
            f"Equity: <code>₹{trade.get('equity', 0):,.0f}</code>"
        )

    @staticmethod
    def error_alert(error_msg: str, action: str = "retrying") -> str:
        return (
            f"🚨 <b>ERROR</b>\n"
            f"Message: <code>{error_msg[:500]}</code>\n"
            f"Action: <code>{action}</code>\n"
            f"Time: <code>{datetime.now().strftime('%H:%M:%S')}</code>"
        )

    @staticmethod
    def health_ping(status: Dict) -> str:
        return (
            "💓 <b>HEALTH CHECK</b>\n"
            f"Status: <code>{status.get('status', '?')}</code>\n"
            f"Regime: <code>{status.get('regime', '?')}</code>\n"
            f"Positions: <code>{status.get('positions', 0)}</code>\n"
            f"Equity: <code>₹{status.get('equity', 0):,.0f}</code>\n"
            f"Open PnL: <code>₹{status.get('open_pnl', 0):+,.0f}</code>\n"
            f"Closed PnL: <code>₹{status.get('closed_pnl', 0):+,.0f}</code>\n"
            f"Drawdown: <code>{status.get('drawdown_pct', 0):.2f}%</code>\n"
            f"Trades today: <code>{status.get('trades_today', 0)}</code>\n"
            f"Uptime: <code>{status.get('uptime', '?')}</code>"
        )

    @staticmethod
    def startup_message() -> str:
        return (
            "🚀 <b>PAPER TRADER STARTED</b>\n"
            f"Strategy: <code>V4 Regime Generalized</code>\n"
            f"Capital: <code>₹10,000,000</code>\n"
            f"Time: <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>\n\n"
            "Commands:\n"
            "/status — Current state\n"
            "/pnl — Today's PnL\n"
            "/positions — Active positions\n"
            "/logs — Recent events\n"
            "/restart — Restart service\n"
            "/stop — Stop service\n"
            "/start — Start service"
        )

    @staticmethod
    def shutdown_message(summary: Dict) -> str:
        return (
            "⏹ <b>PAPER TRADER STOPPED</b>\n"
            f"Final Equity: <code>₹{summary.get('equity', 0):,.0f}</code>\n"
            f"Closed PnL: <code>₹{summary.get('closed_pnl', 0):+,.0f}</code>\n"
            f"Total Fills: <code>{summary.get('fills', 0)}</code>\n"
            f"Uptime: <code>{summary.get('uptime', '?')}</code>"
        )

    @staticmethod
    def kill_switch_alert(dd_pct: float) -> str:
        return (
            "🛑 <b>KILL SWITCH ACTIVATED</b>\n"
            f"Drawdown: <code>{dd_pct:.2f}%</code>\n"
            f"All trading halted.\n"
            f"Use /restart to reset."
        )


# ── Command Handler ─────────────────────────────────────────────────────

class CommandHandler:
    """
    Listens for Telegram commands in a background thread.
    Calls registered callbacks for each command.
    """

    def __init__(self, client: TelegramClient):
        self.client = client
        self._callbacks: Dict[str, Callable] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._update_offset = 0

    def register(self, command: str, callback: Callable):
        """Register a handler for /command."""
        self._callbacks[command.lstrip("/")] = callback

    def start(self):
        """Start the command listener in a background thread."""
        if not self.client.enabled:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=15)

    def _poll_loop(self):
        """Long-poll for incoming messages."""
        while self._running:
            try:
                updates = self.client.get_updates(
                    offset=self._update_offset, timeout=10
                )
                for update in updates:
                    self._update_offset = update["update_id"] + 1
                    self._handle_update(update)
            except Exception:
                time.sleep(5)

    def _handle_update(self, update: Dict):
        """Process a single update."""
        msg = update.get("message", {})
        text = msg.get("text", "").strip()
        chat_id = str(msg.get("chat", {}).get("id", ""))

        # Security: only respond to the configured chat
        if chat_id != self.client.chat_id:
            return

        if not text.startswith("/"):
            return

        cmd = text.split()[0].lstrip("/").split("@")[0].lower()
        callback = self._callbacks.get(cmd)

        if callback:
            try:
                result = callback()
                if result:
                    self.client.send_message(str(result))
            except Exception as e:
                self.client.send_message(f"⚠️ Command error: {e}")
        else:
            self.client.send_message(
                f"❓ Unknown command: /{cmd}\n\n"
                "Available: /status /pnl /positions /logs /restart /stop /start"
            )


# ── Telegram Integration Manager ───────────────────────────────────────

class TelegramManager:
    """
    High-level manager that ties together:
    - Alert sending (trade, error, health)
    - Command handling
    - Health ping scheduling

    Thread-safe. Never blocks trading.
    """

    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
        health_interval: int = 300,  # 5 minutes
    ):
        self.client = TelegramClient(bot_token, chat_id)
        self.formatter = AlertFormatter()
        self.commands = CommandHandler(self.client)
        self._health_interval = health_interval
        self._last_health_ping = datetime.min
        self._status_callback: Optional[Callable] = None
        self._pnl_callback: Optional[Callable] = None
        self._positions_callback: Optional[Callable] = None
        self._logs_callback: Optional[Callable] = None
        self._start_time = datetime.now()

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    def set_status_callback(self, cb: Callable):
        self._status_callback = cb

    def set_pnl_callback(self, cb: Callable):
        self._pnl_callback = cb

    def set_positions_callback(self, cb: Callable):
        self._positions_callback = cb

    def set_logs_callback(self, cb: Callable):
        self._logs_callback = cb

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self):
        """Start the Telegram system (non-blocking)."""
        if not self.enabled:
            return

        # Register built-in commands
        self.commands.register("status", self._cmd_status)
        self.commands.register("pnl", self._cmd_pnl)
        self.commands.register("positions", self._cmd_positions)
        self.commands.register("logs", self._cmd_logs)
        self.commands.register("restart", self._cmd_restart)
        self.commands.register("stop", self._cmd_stop)
        self.commands.register("start", self._cmd_start)
        self.commands.register("help", self._cmd_help)

        self.commands.start()
        self._start_time = datetime.now()

        # Send startup message
        self.client.send_message(self.formatter.startup_message())

    def stop(self, summary: Dict = None):
        """Stop the Telegram system."""
        if not self.enabled:
            return
        if summary:
            summary["uptime"] = self._format_uptime()
            self.client.send_message(self.formatter.shutdown_message(summary))
        self.commands.stop()

    # ── Alert Methods ───────────────────────────────────────────────────

    def send_trade_entry(self, trade: Dict):
        """Send trade entry alert (non-blocking)."""
        if not self.enabled:
            return
        threading.Thread(
            target=self.client.send_message,
            args=(self.formatter.trade_entry(trade),),
            daemon=True,
        ).start()

    def send_trade_exit(self, trade: Dict):
        """Send trade exit alert (non-blocking)."""
        if not self.enabled:
            return
        threading.Thread(
            target=self.client.send_message,
            args=(self.formatter.trade_exit(trade),),
            daemon=True,
        ).start()

    def send_error(self, error_msg: str, action: str = "retrying"):
        """Send error alert (non-blocking)."""
        if not self.enabled:
            return
        threading.Thread(
            target=self.client.send_message,
            args=(self.formatter.error_alert(error_msg, action),),
            daemon=True,
        ).start()

    def send_kill_switch(self, dd_pct: float):
        """Send kill switch alert."""
        if not self.enabled:
            return
        self.client.send_message(self.formatter.kill_switch_alert(dd_pct))

    def maybe_send_health_ping(self, status: Dict):
        """Send health ping if interval has elapsed. Non-blocking."""
        if not self.enabled:
            return
        now = datetime.now()
        if (now - self._last_health_ping).total_seconds() < self._health_interval:
            return
        self._last_health_ping = now
        status["uptime"] = self._format_uptime()
        threading.Thread(
            target=self.client.send_message,
            args=(self.formatter.health_ping(status),),
            daemon=True,
        ).start()

    # ── Command Implementations ─────────────────────────────────────────

    def _cmd_status(self) -> str:
        if self._status_callback:
            return self._status_callback()
        return "📊 Status callback not configured"

    def _cmd_pnl(self) -> str:
        if self._pnl_callback:
            return self._pnl_callback()
        return "💰 PnL callback not configured"

    def _cmd_positions(self) -> str:
        if self._positions_callback:
            return self._positions_callback()
        return "📋 No positions callback"

    def _cmd_logs(self) -> str:
        if self._logs_callback:
            return self._logs_callback()
        # Default: read last 20 lines of events.log
        try:
            log_path = Path(__file__).parent / "logs" / "events.log"
            if log_path.exists():
                lines = log_path.read_text().strip().split("\n")[-20:]
                return "📄 <b>Recent Logs</b>\n<pre>" + "\n".join(lines) + "</pre>"
            return "📄 No log file found"
        except Exception as e:
            return f"📄 Log read error: {e}"

    def _cmd_restart(self) -> str:
        self.client.send_message("♻️ Restarting trader service...")
        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", "trader"],
                timeout=10,
                capture_output=True,
            )
            return "✅ Restart command sent"
        except Exception as e:
            return f"❌ Restart failed: {e}"

    def _cmd_stop(self) -> str:
        self.client.send_message("⏹ Stopping trader service...")
        try:
            subprocess.run(
                ["sudo", "systemctl", "stop", "trader"],
                timeout=10,
                capture_output=True,
            )
            return "✅ Stop command sent"
        except Exception as e:
            return f"❌ Stop failed: {e}"

    def _cmd_start(self) -> str:
        try:
            subprocess.run(
                ["sudo", "systemctl", "start", "trader"],
                timeout=10,
                capture_output=True,
            )
            return "✅ Start command sent"
        except Exception as e:
            return f"❌ Start failed: {e}"

    def _cmd_help(self) -> str:
        return (
            "📖 <b>Available Commands</b>\n\n"
            "/status — System status\n"
            "/pnl — Today's PnL summary\n"
            "/positions — Active positions\n"
            "/logs — Last 20 log lines\n"
            "/restart — Restart service\n"
            "/stop — Stop service\n"
            "/start — Start service\n"
            "/help — This message"
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _format_uptime(self) -> str:
        delta = datetime.now() - self._start_time
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        return f"{hours}h {minutes}m"


# ── Factory ─────────────────────────────────────────────────────────────

def create_telegram_manager(
    health_interval: int = 300,
) -> TelegramManager:
    """Create TelegramManager from environment variables."""
    from dotenv import load_dotenv
    load_dotenv()

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        print("[TELEGRAM] Bot token or chat ID not set — alerts disabled")

    return TelegramManager(
        bot_token=bot_token,
        chat_id=chat_id,
        health_interval=health_interval,
    )
