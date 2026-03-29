#!/usr/bin/env python3
"""
=============================================================================
MARKET HOURS — IST Market Session Guard
=============================================================================
Reusable market hours + holiday checks for the live trading system.
Ensures NO trades are executed outside NSE market hours.

Market hours:  09:15 – 15:30 IST (Mon–Fri, excluding NSE holidays)
Square-off:    15:20 IST (10 min before close)
=============================================================================
"""

from __future__ import annotations

from datetime import datetime, time, date, timezone, timedelta
from typing import Tuple

# IST = UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))

# NSE market session times (IST)
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
SQUARE_OFF_TIME = time(15, 20)   # auto square-off trigger
PRE_OPEN_START = time(9, 0)      # pre-open session (no trading)

# ── NSE Holidays 2026 (confirmed + provisional) ────────────────────────
# Source: NSE India circular — update annually
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 17),   # Mahashivratri (provisional)
    date(2026, 3, 10),   # Holi (provisional)
    date(2026, 3, 30),   # Id-Ul-Fitr (provisional)
    date(2026, 4, 2),    # Ram Navami (provisional)
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 6, 6),    # Bakri Id (provisional)
    date(2026, 7, 6),    # Muharram (provisional)
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 16),   # Janmashtami (provisional)
    date(2026, 9, 4),    # Milad-Un-Nabi (provisional)
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra (provisional)
    date(2026, 11, 9),   # Diwali (Laxmi Puja) (provisional)
    date(2026, 11, 10),  # Diwali (Balipratipada) (provisional)
    date(2026, 11, 27),  # Guru Nanak Jayanti (provisional)
    date(2026, 12, 25),  # Christmas
}


def now_ist() -> datetime:
    """Current time in IST."""
    return datetime.now(IST)


def is_market_open() -> bool:
    """
    Check if NSE market is currently open.

    Returns True ONLY if:
      - It's a weekday (Mon–Fri)
      - It's NOT an NSE holiday
      - Current time is between 09:15 and 15:30 IST
    """
    now = now_ist()
    today = now.date()
    current_time = now.time()

    # Weekend check
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Holiday check
    if today in NSE_HOLIDAYS_2026:
        return False

    # Time window check
    if current_time < MARKET_OPEN or current_time >= MARKET_CLOSE:
        return False

    return True


def is_square_off_time() -> bool:
    """
    Check if it's time for EOD auto square-off (15:20 IST).

    Returns True if current time is between 15:20 and 15:30 IST
    on a trading day.
    """
    now = now_ist()
    today = now.date()
    current_time = now.time()

    # Only on trading days
    if now.weekday() >= 5:
        return False
    if today in NSE_HOLIDAYS_2026:
        return False

    return SQUARE_OFF_TIME <= current_time < MARKET_CLOSE


def is_trading_day() -> bool:
    """Check if today is a trading day (weekday + not holiday)."""
    now = now_ist()
    today = now.date()
    if now.weekday() >= 5:
        return False
    if today in NSE_HOLIDAYS_2026:
        return False
    return True


def market_status() -> Tuple[str, str]:
    """
    Human-readable market status.
    Returns (status_emoji, status_text).
    """
    now = now_ist()
    today = now.date()
    current_time = now.time()

    if now.weekday() >= 5:
        return "🔴", "CLOSED (weekend)"
    if today in NSE_HOLIDAYS_2026:
        return "🔴", "CLOSED (NSE holiday)"
    if current_time < MARKET_OPEN:
        return "🟡", f"PRE-MARKET (opens {MARKET_OPEN.strftime('%H:%M')})"
    if current_time >= MARKET_CLOSE:
        return "🔴", f"CLOSED (closed {MARKET_CLOSE.strftime('%H:%M')})"
    if current_time >= SQUARE_OFF_TIME:
        return "🟠", "SQUARE-OFF WINDOW (15:20–15:30)"
    return "🟢", "OPEN"


def time_to_market_open() -> timedelta:
    """Time remaining until next market open. Returns timedelta."""
    now = now_ist()
    today = now.date()

    # Find next trading day
    candidate = today
    for _ in range(10):  # max 10 days lookahead
        if candidate.weekday() < 5 and candidate not in NSE_HOLIDAYS_2026:
            open_dt = datetime.combine(candidate, MARKET_OPEN, tzinfo=IST)
            if open_dt > now:
                return open_dt - now
        candidate += timedelta(days=1)

    return timedelta(hours=999)  # fallback
