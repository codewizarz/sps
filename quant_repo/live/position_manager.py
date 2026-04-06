#!/usr/bin/env python3
"""
=============================================================================
PAPER TRADING — POSITION MANAGER
=============================================================================
Tracks open paper positions, computes live PnL, handles exit signals
(SL / TP / time / delta), and enforces risk limits.
=============================================================================
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from quant_repo.live.execution_engine import ExecutionEngine, SimulatedFill
from quant_repo.live.logger import PaperLogger


@dataclass
class PaperPosition:
    """A single open paper position (short straddle)."""
    position_id: str
    symbol: str
    strike: float
    entry_date: datetime
    expiry_date: datetime
    entry_premium: float        # slippage-adjusted fill price
    lots: int
    lot_qty: int
    regime: str
    position_scale: float
    quality_score: float

    # Live tracking
    current_price: float = 0.0
    best_pnl_pct: float = 0.0
    stop_price: float = 0.0
    stop_stage: int = 0         # 0=initial, 1=breakeven, 2=profit-lock
    partial_taken: bool = False
    original_lots: int = 0

    def __post_init__(self):
        self.original_lots = self.lots
        self.current_price = self.entry_premium

    @property
    def unrealized_pnl(self) -> float:
        return (self.entry_premium - self.current_price) * self.lots * self.lot_qty

    @property
    def pnl_pct(self) -> float:
        if self.entry_premium <= 0:
            return 0.0
        return (self.entry_premium - self.current_price) / self.entry_premium

    @property
    def dte(self) -> int:
        return max((self.expiry_date - datetime.now()).days, 0)


@dataclass
class RiskLimits:
    """Hard risk controls for the paper trading system."""
    max_trades_per_cycle: int = 2
    daily_loss_limit_pct: float = 2.0      # -2% max daily loss
    kill_switch_dd_pct: float = 12.0       # kill switch at 12% drawdown
    max_concurrent_positions: int = 2

    # Exit rules (matching V4)
    stop_loss_multiple: float = 1.35
    breakeven_activation_pct: float = 0.25
    profit_lock_activation_pct: float = 0.45
    profit_lock_stop_pct: float = 0.20
    profit_target_pct: float = 0.65
    time_exit_pct: float = 0.60
    time_exit_min_decay: float = 0.15


class PositionManager:
    """
    Manages paper positions with full exit logic and risk enforcement.
    Thread-safe for concurrent tick updates.
    """

    def __init__(
        self,
        execution_engine: ExecutionEngine,
        logger: PaperLogger,
        initial_capital: float = 10_000_000,
        risk_limits: Optional[RiskLimits] = None,
    ):
        self.engine = execution_engine
        self.logger = logger
        self.initial_capital = initial_capital
        self.risk = risk_limits or RiskLimits()

        self.positions: List[PaperPosition] = []
        self.closed_pnl: float = 0.0
        self.peak_equity: float = initial_capital
        self.kill_switch_active: bool = False
        self._lock = threading.Lock()

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        return self.initial_capital + self.closed_pnl + self.open_pnl

    @property
    def open_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions)

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (1.0 - self.equity / self.peak_equity) * 100.0)

    @property
    def daily_pnl(self) -> float:
        today = datetime.now().date()
        today_exits = sum(
            f.net_pnl for f in self.engine.fills
            if f.action == "EXIT" and f.timestamp.date() == today
        )
        return today_exits + self.open_pnl

    @property
    def daily_return_pct(self) -> float:
        start_equity = max(self.equity - self.daily_pnl, 1.0)
        return (self.daily_pnl / start_equity) * 100.0

    @property
    def trades_today(self) -> int:
        return self.engine.entry_fills_today

    @property
    def active_position_count(self) -> int:
        return len(self.positions)

    # ── Risk checks ─────────────────────────────────────────────────────

    def can_enter_trade(self) -> Tuple[bool, str]:
        """Check all risk gates before allowing entry."""
        if self.kill_switch_active:
            return False, "KILL SWITCH active"

        if self.active_position_count >= self.risk.max_concurrent_positions:
            return False, f"Max concurrent positions ({self.risk.max_concurrent_positions})"

        if self.trades_today >= self.risk.max_trades_per_cycle:
            return False, f"Max trades per cycle ({self.risk.max_trades_per_cycle})"

        # Daily loss limit
        if self.daily_return_pct < -self.risk.daily_loss_limit_pct:
            return False, f"Daily loss limit hit ({self.daily_return_pct:.2f}%)"

        # Drawdown kill switch
        if self.drawdown_pct >= self.risk.kill_switch_dd_pct:
            self.kill_switch_active = True
            self.logger.warning(
                f"🚨 KILL SWITCH ACTIVATED: Drawdown {self.drawdown_pct:.2f}% "
                f">= {self.risk.kill_switch_dd_pct}%"
            )
            return False, "Kill switch triggered"

        return True, "OK"

    # ── Open position ───────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        strike: float,
        premium: float,
        lots: int,
        lot_qty: int,
        expiry_date: datetime,
        regime: str,
        quality_score: float = 0.0,
        position_scale: float = 1.0,
    ) -> Optional[PaperPosition]:
        """Open a new paper position via the execution engine."""
        with self._lock:
            ok, reason = self.can_enter_trade()
            if not ok:
                self.logger.info(f"[RISK] Entry blocked: {reason}")
                return None

            fill = self.engine.execute_entry(
                symbol=symbol,
                strike=strike,
                premium=premium,
                lots=lots,
                lot_qty=lot_qty,
                regime=regime,
                quality_score=quality_score,
                position_scale=position_scale,
            )

            pos = PaperPosition(
                position_id=fill.fill_id,
                symbol=symbol,
                strike=strike,
                entry_date=datetime.now(),
                expiry_date=expiry_date,
                entry_premium=fill.fill_price,
                lots=lots,
                lot_qty=lot_qty,
                regime=regime,
                position_scale=position_scale,
                quality_score=quality_score,
                stop_price=fill.fill_price * self.risk.stop_loss_multiple,
            )
            self.positions.append(pos)
            return pos

    # ── Exit logic ──────────────────────────────────────────────────────

    def check_exits(self, current_prices: Dict[str, float]):
        """
        Run exit checks on all open positions.
        current_prices: {f"{symbol}_{strike}": current_straddle_price}
        """
        with self._lock:
            to_close: List[Tuple[PaperPosition, str, float]] = []

            for pos in self.positions:
                key = f"{pos.symbol}_{int(pos.strike)}"
                price = current_prices.get(key, pos.current_price)
                pos.current_price = price

                exit_reason, exit_price = self._evaluate_exit(pos, price)
                if exit_reason:
                    to_close.append((pos, exit_reason, exit_price))

            for pos, reason, price in to_close:
                self._close_position(pos, reason, price)

    def _evaluate_exit(
        self, pos: PaperPosition, current_price: float
    ) -> Tuple[Optional[str], float]:
        """Mirror V4 convex exit logic."""
        r = self.risk
        pnl_pct = pos.pnl_pct
        pos.best_pnl_pct = max(pos.best_pnl_pct, pnl_pct)

        # Trail to breakeven
        if pnl_pct >= r.breakeven_activation_pct and pos.stop_stage < 1:
            pos.stop_stage = 1
            pos.stop_price = min(pos.stop_price, pos.entry_premium)

        # Profit lock trail
        if pnl_pct >= r.profit_lock_activation_pct and pos.stop_stage < 2:
            pos.stop_stage = 2
            pos.stop_price = min(
                pos.stop_price,
                pos.entry_premium * (1.0 - r.profit_lock_stop_pct),
            )

        # Stop triggered
        if current_price >= pos.stop_price:
            labels = {0: "Stop Loss", 1: "Trail Breakeven", 2: "Profit Lock Trail"}
            return labels.get(pos.stop_stage, "Stop Loss"), pos.stop_price

        # Profit target
        if pnl_pct >= r.profit_target_pct:
            return "Profit Target", current_price

        # Time exit
        entry_to_expiry = max((pos.expiry_date - pos.entry_date).days, 1)
        elapsed = max((datetime.now() - pos.entry_date).days, 0)
        time_pct = elapsed / entry_to_expiry
        if time_pct >= r.time_exit_pct and pnl_pct < r.time_exit_min_decay:
            return "Time Exit", current_price

        # Expiry close
        if pos.dte <= 0:
            return "Expiry Close", current_price

        return None, current_price

    def _close_position(self, pos: PaperPosition, reason: str, price: float):
        """Execute exit and update accounting."""
        fill = self.engine.execute_exit(
            symbol=pos.symbol,
            strike=pos.strike,
            current_price=price,
            entry_fill_price=pos.entry_premium,
            lots=pos.lots,
            lot_qty=pos.lot_qty,
            regime=pos.regime,
            exit_reason=reason,
        )
        self.closed_pnl += fill.net_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.positions.remove(pos)

    def close_position(self, pos: PaperPosition, reason: str, price: Optional[float] = None) -> bool:
        """Public close API for external signal handlers.

        Returns True when a position was closed, False when it was already absent.
        """
        with self._lock:
            if pos not in self.positions:
                return False
            self._close_position(pos, reason, pos.current_price if price is None else price)
            return True

    # ── Force close all ─────────────────────────────────────────────────

    def close_all(self, reason: str = "Manual close"):
        """Emergency close all positions."""
        with self._lock:
            for pos in list(self.positions):
                self._close_position(pos, reason, pos.current_price)

    # ── Dashboard data ──────────────────────────────────────────────────

    def get_position_summaries(self) -> List[Dict]:
        return [
            {
                "symbol": p.symbol,
                "strike": p.strike,
                "lots": p.lots,
                "entry_premium": p.entry_premium,
                "current_price": p.current_price,
                "unrealized_pnl": p.unrealized_pnl,
                "pnl_pct": p.pnl_pct * 100,
                "regime": p.regime,
                "dte": p.dte,
                "stop_stage": p.stop_stage,
            }
            for p in self.positions
        ]
