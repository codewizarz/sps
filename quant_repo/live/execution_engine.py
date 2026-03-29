#!/usr/bin/env python3
"""
=============================================================================
PAPER TRADING — EXECUTION ENGINE
=============================================================================
Simulates trade execution with realistic slippage and cost modeling.
NO real orders are placed. All fills are synthetic.
=============================================================================
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from quant_repo.live.logger import PaperLogger
from quant_repo.live.market_hours import is_market_open


@dataclass
class SimulatedFill:
    """Represents a simulated order fill."""
    fill_id: str
    timestamp: datetime
    symbol: str
    action: str           # "ENTRY" or "EXIT"
    side: str             # "SELL" (short straddle entry) or "BUY" (covering)
    strike: float
    premium: float        # raw market price
    fill_price: float     # slippage-adjusted price
    lots: int
    lot_qty: int
    regime: str
    exit_reason: str
    quality_score: float
    position_scale: float
    gross_pnl: float
    net_pnl: float


class ExecutionEngine:
    """
    Paper execution engine.
    - Simulates fills at LTP ± slippage
    - Tracks all fills in memory + CSV
    - NO real API calls
    """

    def __init__(
        self,
        logger: PaperLogger,
        slippage_pct: float = 0.005,      # 0.5% default
        brokerage_per_lot: float = 25.0,
        stt_pct: float = 0.0005,
        telegram=None,
    ):
        self.logger = logger
        self.slippage_pct = slippage_pct
        self.brokerage_per_lot = brokerage_per_lot
        self.stt_pct = stt_pct
        self.fills: List[SimulatedFill] = []
        self._trade_counter = 0
        self.telegram = telegram  # Optional TelegramManager

    def execute_entry(
        self,
        symbol: str,
        strike: float,
        premium: float,
        lots: int,
        lot_qty: int,
        regime: str,
        quality_score: float = 0.0,
        position_scale: float = 1.0,
    ) -> SimulatedFill:
        """
        Simulate a SHORT STRADDLE entry.
        For short selling, slippage works against us: we receive LESS premium.
        fill_price = premium * (1 - slippage)
        """
        # ── HARD SAFETY BLOCK ───────────────────────────────────────
        if not is_market_open():
            self.logger.error(
                f"[BLOCK] ENTRY BLOCKED — market closed | {symbol} {strike:.0f}"
            )
            raise RuntimeError(
                f"Market closed — execution blocked for {symbol} {strike:.0f}"
            )

        self._trade_counter += 1
        fill_price = premium * (1.0 - self.slippage_pct)

        fill = SimulatedFill(
            fill_id=f"PT-{self._trade_counter:04d}-{uuid.uuid4().hex[:6]}",
            timestamp=datetime.now(),
            symbol=symbol,
            action="ENTRY",
            side="SELL",
            strike=strike,
            premium=premium,
            fill_price=fill_price,
            lots=lots,
            lot_qty=lot_qty,
            regime=regime,
            exit_reason="",
            quality_score=quality_score,
            position_scale=position_scale,
            gross_pnl=0.0,
            net_pnl=0.0,
        )

        self.fills.append(fill)
        self.logger.record_trade({
            "trade_id": fill.fill_id,
            "symbol": symbol,
            "action": "ENTRY",
            "strike": strike,
            "premium": premium,
            "lots": lots,
            "lot_qty": lot_qty,
            "slippage_adjusted_price": fill_price,
            "side": "SELL",
            "regime": regime,
            "exit_reason": "",
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "position_scale": position_scale,
            "quality_score": quality_score,
        })

        self.logger.info(
            f"[EXEC] ENTRY {symbol} | Strike {strike:.0f} | "
            f"Premium {premium:.2f} → Fill {fill_price:.2f} | "
            f"Lots {lots}×{lot_qty} | {regime}"
        )

        # Telegram alert
        if self.telegram:
            self.telegram.send_trade_entry({
                "symbol": symbol, "strike": strike, "premium": premium,
                "fill_price": fill_price, "lots": lots, "lot_qty": lot_qty,
                "regime": regime, "position_scale": position_scale,
            })

        return fill

    def execute_exit(
        self,
        symbol: str,
        strike: float,
        current_price: float,
        entry_fill_price: float,
        lots: int,
        lot_qty: int,
        regime: str,
        exit_reason: str,
    ) -> SimulatedFill:
        """
        Simulate a BUY-TO-CLOSE exit.
        For buying back, slippage works against us: we pay MORE.
        fill_price = current_price * (1 + slippage)
        """
        self._trade_counter += 1
        fill_price = current_price * (1.0 + self.slippage_pct)

        # PnL = (entry_premium - exit_price) * lots * lot_qty (short position)
        gross_pnl = (entry_fill_price - fill_price) * lots * lot_qty
        costs = self._compute_costs(entry_fill_price, lots, lot_qty)
        net_pnl = gross_pnl - costs

        fill = SimulatedFill(
            fill_id=f"PT-{self._trade_counter:04d}-{uuid.uuid4().hex[:6]}",
            timestamp=datetime.now(),
            symbol=symbol,
            action="EXIT",
            side="BUY",
            strike=strike,
            premium=current_price,
            fill_price=fill_price,
            lots=lots,
            lot_qty=lot_qty,
            regime=regime,
            exit_reason=exit_reason,
            quality_score=0.0,
            position_scale=0.0,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
        )

        self.fills.append(fill)
        self.logger.record_trade({
            "trade_id": fill.fill_id,
            "symbol": symbol,
            "action": "EXIT",
            "strike": strike,
            "premium": current_price,
            "lots": lots,
            "lot_qty": lot_qty,
            "slippage_adjusted_price": fill_price,
            "side": "BUY",
            "regime": regime,
            "exit_reason": exit_reason,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "position_scale": 0.0,
            "quality_score": 0.0,
        })

        self.logger.info(
            f"[EXEC] EXIT  {symbol} | Strike {strike:.0f} | "
            f"Price {current_price:.2f} → Fill {fill_price:.2f} | "
            f"PnL Rs {net_pnl:+,.0f} | {exit_reason}"
        )

        # Telegram alert
        if self.telegram:
            self.telegram.send_trade_exit({
                "symbol": symbol, "strike": strike, "exit_reason": exit_reason,
                "net_pnl": net_pnl, "equity": 0,  # filled by caller if needed
            })

        return fill

    def _compute_costs(
        self, premium: float, lots: int, lot_qty: int
    ) -> float:
        """Transaction cost model matching V4 strategy."""
        notional = premium * lots * lot_qty
        brokerage = self.brokerage_per_lot * lots * 2  # entry + exit
        stt = notional * self.stt_pct
        slippage_cost = notional * self.slippage_pct
        return brokerage + stt + slippage_cost

    @property
    def total_fills(self) -> int:
        return len(self.fills)

    @property
    def entry_fills_today(self) -> int:
        today = datetime.now().date()
        return sum(
            1 for f in self.fills
            if f.action == "ENTRY" and f.timestamp.date() == today
        )

    def get_closed_pnl(self) -> float:
        return sum(f.net_pnl for f in self.fills if f.action == "EXIT")
