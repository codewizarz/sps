#!/usr/bin/env python3
"""
=============================================================================
PAPER TRADING — LOGGER
=============================================================================
Centralized logging for the paper trading system.
Writes:
  - events.log       (all events, structured)
  - trades_live.csv  (trade records)
  - pnl_live.csv     (periodic PnL snapshots)
=============================================================================
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class PaperLogger:
    """Centralized file + console logger for the paper trading system."""

    def __init__(self, output_dir: str = "quant_repo/live/logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Event log (file + console) ---
        self.log = logging.getLogger("paper_trader")
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False

        # Remove existing handlers
        self.log.handlers.clear()

        # File handler
        fh = logging.FileHandler(
            self.output_dir / "events.log", mode="a", encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
        )
        self.log.addHandler(fh)

        # Console handler (INFO+)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        )
        self.log.addHandler(ch)

        # --- CSV writers ---
        self._trades_path = self.output_dir / "trades_live.csv"
        self._pnl_path = self.output_dir / "pnl_live.csv"
        self._init_csv(
            self._trades_path,
            [
                "timestamp", "trade_id", "symbol", "action", "strike",
                "premium", "lots", "lot_qty", "slippage_adjusted_price",
                "side", "regime", "exit_reason", "gross_pnl", "net_pnl",
                "position_scale", "quality_score",
            ],
        )
        self._init_csv(
            self._pnl_path,
            [
                "timestamp", "equity", "open_pnl", "closed_pnl",
                "drawdown_pct", "active_positions", "trades_today",
                "regime", "daily_return_pct",
            ],
        )

    @staticmethod
    def _init_csv(path: Path, headers: List[str]):
        """Create CSV with headers if it does not exist."""
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(headers)

    # ── events ──────────────────────────────────────────────────────────

    def info(self, msg: str):
        self.log.info(msg)

    def warning(self, msg: str):
        self.log.warning(msg)

    def error(self, msg: str):
        self.log.error(msg)

    def debug(self, msg: str):
        self.log.debug(msg)

    # ── trade CSV ───────────────────────────────────────────────────────

    def record_trade(self, row: Dict):
        """Append a single trade record to trades_live.csv."""
        row.setdefault("timestamp", datetime.now().isoformat())
        with open(self._trades_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._read_headers(self._trades_path))
            writer.writerow(row)

    # ── PnL CSV ─────────────────────────────────────────────────────────

    def record_pnl(self, row: Dict):
        """Append a PnL snapshot to pnl_live.csv."""
        row.setdefault("timestamp", datetime.now().isoformat())
        with open(self._pnl_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._read_headers(self._pnl_path))
            writer.writerow(row)

    @staticmethod
    def _read_headers(path: Path) -> List[str]:
        with open(path, "r") as f:
            return next(csv.reader(f))

    # ── console dashboard ───────────────────────────────────────────────

    def print_dashboard(
        self,
        equity: float,
        open_pnl: float,
        closed_pnl: float,
        drawdown_pct: float,
        positions: List[Dict],
        trades_today: int,
        regime: str,
        daily_return_pct: float,
    ):
        """Clear-screen console dashboard for live monitoring."""
        width = 68
        now = datetime.now().strftime("%H:%M:%S")

        lines = [
            "",
            "═" * width,
            f"  PAPER TRADER DASHBOARD  │  {now}",
            "═" * width,
            f"  Equity:      Rs {equity:>14,.0f}   │  Regime: {regime}",
            f"  Open PnL:    Rs {open_pnl:>14,.0f}   │  Trades today: {trades_today}",
            f"  Closed PnL:  Rs {closed_pnl:>14,.0f}   │  Daily return: {daily_return_pct:+.2f}%",
            f"  Drawdown:    {drawdown_pct:>8.2f}%         │",
            "─" * width,
        ]

        if positions:
            lines.append("  ACTIVE POSITIONS:")
            for p in positions:
                lines.append(
                    f"    {p.get('symbol','?'):10s} | Strike {p.get('strike',0):.0f} | "
                    f"Lots {p.get('lots',0)} | PnL Rs {p.get('unrealized_pnl',0):+,.0f} | "
                    f"{p.get('regime','?')}"
                )
        else:
            lines.append("  No active positions")

        lines.append("═" * width)

        # Clear and print
        os.system("clear" if os.name != "nt" else "cls")
        print("\n".join(lines))
