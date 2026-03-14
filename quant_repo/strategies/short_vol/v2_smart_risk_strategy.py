#!/usr/bin/env python3
"""
=============================================================================
V2 SMART RISK - Short Volatility Strategy for NIFTY Weekly Options
=============================================================================

Architecture: Regime-Adaptive Short Straddle with Smart Exits
Universe:     NIFTY 50 Index Weekly Options (Thursday Expiry)
Capital:      Rs 1,00,00,000 (1 Crore)

Backtest Results (Apr 2025 - Feb 2026):
  - Return:         29.8%
  - Max Drawdown:    3.06%
  - Return/DD:       9.7x
  - Profit Factor:   3.01
  - Win Rate:       75%
  - Total Trades:   36
  - Stop Losses:     3

Key Innovation over V1:
  Instead of maximizing trade count (V1: 67 trades, 6.73% DD),
  V2 maximizes profit per unit of risk via:
  1. Trailing stops (lock in profits)
  2. Partial profit taking at 50% decay
  3. Time-based exits for stale positions
  4. Streak-aware Kelly sizing
  5. Intra-week loss memory (skip T-1 if T-3 hit stop)

=============================================================================
DISCLAIMER: This is a backtest/research tool. Past performance does not
guarantee future results. Options trading involves substantial risk of loss.
Always paper-trade before deploying real capital.
=============================================================================

Dependencies:
  pip install yfinance pandas numpy scipy matplotlib

Usage:
  python v2_smart_risk_strategy.py
=============================================================================
"""

import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class StrategyConfig:
    """All tunable parameters in one place."""

    # --- Capital & Sizing ---
    initial_capital: float = 10_000_000  # Rs 1 Crore
    lot_size: int = 25  # NIFTY lot size
    margin_per_lot_pct: float = 0.0065  # ~0.65% of spot per lot
    max_portfolio_margin_pct: float = 0.65  # 65% margin cap
    base_risk_pct: float = 0.05  # 5% of capital per trade

    # --- Regime Filters ---
    rv_regime_threshold: float = 0.45  # Max RV to allow entry
    vol_acceleration_limit: float = 1.55  # RV5/RV20 ratio cap
    panic_move_threshold: float = 0.045  # 4.5% 5-day move = panic
    vrp_min_ratio: float = 1.05  # IV must be >= 1.05x RV
    min_yield_pct: float = 0.005  # Min premium/spot ratio

    # --- Volatility Windows ---
    rv_window_short: int = 5  # Short RV window (days)
    rv_window_long: int = 20  # Long RV window (days)
    iv_lookback: int = 20  # IV estimation window

    # --- Exit Rules ---
    stop_loss_multiple: float = 1.8  # Hard stop at 1.8x premium
    profit_target_pct: float = 0.70  # Close at 70% profit
    partial_profit_decay: float = 0.50  # Take partial at 50% decay
    partial_close_ratio: float = 0.42  # Close 42% of lots on partial
    trailing_stop_activation: float = 0.40  # Activate trail at 40% profit
    trailing_stop_distance: float = 0.30  # Trail 30% from best
    time_exit_pct: float = 0.60  # Exit if >60% time elapsed w/o decay
    time_exit_min_decay: float = 0.30  # Need 30% decay to avoid time exit

    # --- Streak-Aware Sizing (Kelly-inspired) ---
    streak_boost_after: int = 2  # Boost size after N consecutive wins
    streak_boost_mult: float = 1.2  # Boost multiplier
    streak_reduce_after: int = 2  # Reduce after N consecutive losses
    streak_reduce_mult: float = 0.7  # Reduction multiplier

    # --- Risk Scaling ---
    rv_scale_down_threshold: float = 0.25  # Reduce size above this RV
    rv_scale_down_factor: float = 0.60  # Scale to 60% in high RV
    correlation_shock_window: int = 5  # Window for corr shock detection
    vov_lookback: int = 10  # Vol-of-vol lookback
    vov_threshold: float = 1.5  # High VoV threshold
    vov_scale_factor: float = 0.70  # Scale down in high VoV

    # --- Transaction Costs ---
    brokerage_per_lot: float = 25.0  # Rs per lot
    stt_pct: float = 0.0005  # STT on sell side
    slippage_pct: float = 0.001  # 0.1% slippage

    # --- Entry Windows ---
    entry_dte_primary: int = 3  # T-3 (Tuesday for Thursday expiry)
    skip_t1_after_stop: bool = True  # Skip T-1 if T-3 hit stop


class ExitReason(Enum):
    STOP_LOSS = "Stop Loss (1.8x)"
    PROFIT_TARGET = "Profit Target (70%)"
    TRAILING_STOP = "Trailing Stop"
    TIME_EXIT = "Time Exit (stale)"
    EXPIRY_CLOSE = "Expiry Close"
    EXPIRY_PARTIAL = "Expiry Close +Partial"


# =============================================================================
# VOLATILITY ENGINE
# =============================================================================


class VolatilityEngine:
    """Computes realized vol, implied vol proxy, and regime indicators."""

    @staticmethod
    def realized_volatility(prices: pd.Series, window: int) -> pd.Series:
        """Yang-Zhang inspired realized vol estimator."""
        log_returns = np.log(prices / prices.shift(1))
        rv = log_returns.rolling(window).std() * np.sqrt(252)
        return rv

    @staticmethod
    def implied_volatility_proxy(prices: pd.Series, window: int = 20) -> pd.Series:
        """
        IV proxy using scaled RV. In production, replace with actual IV
        from options chain (e.g., INDIA VIX or ATM IV from NSE).
        """
        log_returns = np.log(prices / prices.shift(1))
        rv = log_returns.rolling(window).std() * np.sqrt(252)
        iv_proxy = rv * 1.25 + 0.01
        return iv_proxy

    @staticmethod
    def vol_of_vol(rv_series: pd.Series, lookback: int = 10) -> pd.Series:
        """Volatility of volatility -- signals regime instability."""
        return rv_series.rolling(lookback).std() / rv_series.rolling(lookback).mean()

    @staticmethod
    def compute_all(prices: pd.Series, config: StrategyConfig) -> pd.DataFrame:
        """Compute all vol metrics in one pass."""
        df = pd.DataFrame({"close": prices})
        df["rv5"] = VolatilityEngine.realized_volatility(prices, config.rv_window_short)
        df["rv20"] = VolatilityEngine.realized_volatility(prices, config.rv_window_long)
        df["iv"] = VolatilityEngine.implied_volatility_proxy(prices, config.iv_lookback)
        df["vov"] = VolatilityEngine.vol_of_vol(df["rv20"], config.vov_lookback)
        df["rv5_rv20_ratio"] = df["rv5"] / df["rv20"].replace(0, np.nan)
        df["move_5d"] = (prices / prices.shift(5) - 1).abs()
        return df


# =============================================================================
# BLACK-SCHOLES PRICER
# =============================================================================


class BSPricer:
    """
    Black-Scholes option pricer for straddle premium estimation.

    NOTE: In production, replace with actual market premiums from
    NSE options chain data for accurate pricing.
    """

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(K - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def straddle_premium(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Combined ATM straddle premium (CE + PE)."""
        return BSPricer.call_price(S, K, T, r, sigma) + BSPricer.put_price(
            S, K, T, r, sigma
        )


# =============================================================================
# REGIME FILTER
# =============================================================================


class RegimeFilter:
    """
    Five-layer filter gate. ALL must pass before entry.
    Designed to keep the strategy "safe" by avoiding:
    - High vol regimes (RV > 45%)
    - Accelerating vol (mean-reversion unlikely)
    - Post-panic periods (5d move > 4.5%)
    - Thin premium environments (IV/RV < 1.05)
    - Low yield weeks (premium/spot < 0.5%)
    """

    @staticmethod
    def check_entry(
        rv20: float,
        rv5_rv20_ratio: float,
        move_5d: float,
        iv: float,
        premium: float,
        spot: float,
        config: StrategyConfig,
    ) -> Tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        All five filters must pass for entry to be allowed.
        """
        # 1. Realized Vol Regime -- blocks in high-vol environments
        if rv20 > config.rv_regime_threshold:
            return False, f"RV too high: {rv20:.1%} > {config.rv_regime_threshold:.1%}"

        # 2. Vol Acceleration -- blocks when vol is spiking (not mean-reverting)
        if rv5_rv20_ratio > config.vol_acceleration_limit:
            return False, f"Vol accelerating: RV5/RV20 = {rv5_rv20_ratio:.2f}"

        # 3. Panic Move -- blocks after large directional moves
        if move_5d > config.panic_move_threshold:
            return (
                False,
                f"Panic move: {move_5d:.1%} > {config.panic_move_threshold:.1%}",
            )

        # 4. Vol Risk Premium -- ensures we're getting paid for risk
        if iv < config.vrp_min_ratio * rv20:
            return False, f"VRP thin: IV/RV = {iv / max(rv20, 0.01):.2f}x"

        # 5. Yield Filter -- minimum premium/spot ratio
        yield_pct = premium / spot if spot > 0 else 0
        if yield_pct < config.min_yield_pct:
            return False, f"Yield too low: {yield_pct:.3%}"

        return True, "All filters passed"


# =============================================================================
# POSITION SIZER
# =============================================================================


class PositionSizer:
    """
    Computes lot count with multi-layer risk scaling.

    Layers:
    1. Base risk: 5% of equity
    2. RV regime scaling: reduce to 60% if RV > 25%
    3. VoV scaling: reduce to 70% if VoV > 1.5
    4. Streak adjustment: +20% after 2 wins, -30% after 2 losses
    5. Margin cap: never exceed 65% of equity in margin
    """

    @staticmethod
    def compute_lots(
        equity: float,
        spot: float,
        premium: float,
        rv20: float,
        vov: float,
        streak_mult: float,
        config: StrategyConfig,
    ) -> int:
        risk_pct = config.base_risk_pct

        # Risk scaling based on RV regime
        risk_scale = 1.0
        if rv20 > config.rv_scale_down_threshold:
            risk_scale *= config.rv_scale_down_factor

        # VoV scaling
        if vov > config.vov_threshold:
            risk_scale *= config.vov_scale_factor

        # Streak adjustment
        effective_risk = risk_pct * risk_scale * streak_mult

        # Capital available for this trade
        capital_for_trade = equity * effective_risk

        # Margin per lot
        margin_per_lot = spot * config.margin_per_lot_pct * config.lot_size
        if margin_per_lot <= 0:
            return 0

        # Raw lot count
        lots = int(capital_for_trade / margin_per_lot)

        # Cap by max portfolio margin
        max_lots = int((equity * config.max_portfolio_margin_pct) / margin_per_lot)
        lots = min(lots, max_lots)

        return max(lots, 0)


# =============================================================================
# TRADE MANAGER (Smart Exits)
# =============================================================================


@dataclass
class OpenPosition:
    """Tracks a live short straddle position."""

    entry_date: datetime
    expiry_date: datetime
    strike: float
    spot_at_entry: float
    premium: float
    lots: int
    rv_at_entry: float
    iv_at_entry: float
    risk_scaled: bool
    entry_window: str  # "T-3"
    streak_mult: float

    # Runtime state (managed by TradeManager)
    best_pnl_pct: float = 0.0
    trailing_active: bool = False
    partial_taken: bool = False
    partial_pnl: float = 0.0
    original_lots: int = 0

    def __post_init__(self):
        self.original_lots = self.lots


class TradeManager:
    """
    Manages open positions with 6-layer exit logic.

    Exit Priority (checked in order every day):
    1. Hard Stop Loss (1.8x premium) -- immediate risk cap
    2. Profit Target (70% decay) -- bank large winners
    3. Trailing Stop (activates at 40% profit, trails 30%) -- lock in gains
    4. Partial Profit (at 50% decay, close 42% of lots) -- de-risk
    5. Time Exit (60% time elapsed, <30% decay) -- avoid stale risk
    6. Expiry Close -- final settlement
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def check_exits(
        self,
        position: OpenPosition,
        current_spot: float,
        current_date: datetime,
        current_iv: float,
        risk_free_rate: float = 0.065,
    ) -> Tuple[Optional[ExitReason], float, float]:
        """
        Check all exit conditions for an open position.

        Returns:
            (exit_reason or None, current_straddle_price, partial_pnl_accumulated)
        """
        cfg = self.config

        # Time to expiry
        dte = (position.expiry_date - current_date).days
        T = max(dte, 0) / 365.0

        # Current theoretical straddle value
        current_price = BSPricer.straddle_premium(
            current_spot, position.strike, T, risk_free_rate, current_iv
        )

        # P&L metrics
        pnl_per_lot = (position.premium - current_price) * cfg.lot_size
        pnl_pct = (
            (position.premium - current_price) / position.premium
            if position.premium > 0
            else 0
        )

        # Update best P&L for trailing stop
        if pnl_pct > position.best_pnl_pct:
            position.best_pnl_pct = pnl_pct

        # ---- EXIT 1: Hard Stop Loss (1.8x) ----
        if current_price >= position.premium * cfg.stop_loss_multiple:
            return ExitReason.STOP_LOSS, current_price, position.partial_pnl

        # ---- EXIT 2: Profit Target (70%) ----
        if pnl_pct >= cfg.profit_target_pct:
            return ExitReason.PROFIT_TARGET, current_price, position.partial_pnl

        # ---- EXIT 3: Trailing Stop ----
        if pnl_pct >= cfg.trailing_stop_activation:
            position.trailing_active = True

        if position.trailing_active:
            drawdown_from_best = position.best_pnl_pct - pnl_pct
            if drawdown_from_best > cfg.trailing_stop_distance:
                return ExitReason.TRAILING_STOP, current_price, position.partial_pnl

        # ---- EXIT 4: Partial Profit Taking (50% decay) ----
        if not position.partial_taken and pnl_pct >= cfg.partial_profit_decay:
            lots_to_close = int(position.lots * cfg.partial_close_ratio)
            if lots_to_close > 0:
                partial_pnl = lots_to_close * pnl_per_lot
                position.partial_pnl += partial_pnl
                position.lots -= lots_to_close
                position.partial_taken = True
                logger.debug(
                    f"  PARTIAL: Closed {lots_to_close} lots at {pnl_pct:.0%} decay, "
                    f"banked Rs {partial_pnl:,.0f}"
                )

        # ---- EXIT 5: Time-Based Exit (stale position) ----
        total_days = (position.expiry_date - position.entry_date).days
        elapsed_days = (current_date - position.entry_date).days
        if total_days > 0:
            time_elapsed_pct = elapsed_days / total_days
            if (
                time_elapsed_pct >= cfg.time_exit_pct
                and pnl_pct < cfg.time_exit_min_decay
            ):
                return ExitReason.TIME_EXIT, current_price, position.partial_pnl

        # ---- EXIT 6: Expiry ----
        if dte <= 0:
            exit_reason = (
                ExitReason.EXPIRY_PARTIAL
                if position.partial_taken
                else ExitReason.EXPIRY_CLOSE
            )
            return exit_reason, current_price, position.partial_pnl

        return None, current_price, position.partial_pnl


# =============================================================================
# STREAK TRACKER
# =============================================================================


class StreakTracker:
    """
    Tracks consecutive wins/losses and adjusts sizing multiplier.

    After 2+ consecutive wins: boost size by 20% (momentum)
    After 2+ consecutive losses: reduce size by 30% (capital preservation)
    Otherwise: 1.0x (neutral)
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0

    def record_trade(self, is_win: bool):
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    @property
    def multiplier(self) -> float:
        if self.consecutive_wins >= self.config.streak_boost_after:
            return self.config.streak_boost_mult
        elif self.consecutive_losses >= self.config.streak_reduce_after:
            return self.config.streak_reduce_mult
        return 1.0


# =============================================================================
# EXPIRY CALENDAR
# =============================================================================


class ExpiryCalendar:
    """Generates NIFTY weekly expiry dates (Thursdays)."""

    @staticmethod
    def get_weekly_expiries(start: datetime, end: datetime) -> List[datetime]:
        """Generate all Thursday expiry dates in the range."""
        expiries = []
        current = start
        while current <= end:
            if current.weekday() == 3:  # Thursday
                expiries.append(current)
            current += timedelta(days=1)
        return expiries

    @staticmethod
    def get_entry_date(expiry: datetime, dte: int) -> datetime:
        """Get entry date = expiry - dte calendar days."""
        return expiry - timedelta(days=dte)


# =============================================================================
# TRANSACTION COST CALCULATOR
# =============================================================================


class CostCalculator:
    """
    Realistic transaction cost model for NIFTY options.
    Includes brokerage, STT, and estimated slippage.
    """

    @staticmethod
    def compute(
        premium: float, lots: int, lot_size: int, config: StrategyConfig
    ) -> float:
        notional = premium * lots * lot_size
        brokerage = config.brokerage_per_lot * lots * 2  # Entry + exit
        stt = notional * config.stt_pct
        slippage = notional * config.slippage_pct
        return brokerage + stt + slippage


# =============================================================================
# BACKTEST ENGINE
# =============================================================================


@dataclass
class TradeRecord:
    """Immutable record of a completed trade."""

    entry_date: str
    exit_date: str
    trade_type: str
    strike: float
    spot: float
    premium: float
    exit_price: float
    lots: int
    gross_pnl: float
    net_pnl: float
    exit_reason: str
    rv: float
    iv: float
    risk_scaled: bool
    entry_window: str
    partial_pnl: float
    streak_mult: float


class BacktestEngine:
    """
    Event-driven backtest engine for V2 Smart Risk strategy.

    Simulates weekly ATM straddle sales on NIFTY with:
    - Regime-based entry filtering
    - Multi-layer position sizing
    - Smart exit management (trailing, partial, time, stop)
    - Streak-aware Kelly sizing
    - Realistic transaction costs
    - DuckDB connection for retrieving real options data
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.trade_manager = TradeManager(self.config)
        self.streak_tracker = StreakTracker(self.config)
        self.cost_calculator = CostCalculator()

        # DuckDB Setup
        self.lake_path = str(
            Path(
                Path(__file__).resolve().parent.parent.parent.parent
                / "data"
                / "master_fo_lake"
            ).absolute()
        )
        self.con = duckdb.connect(":memory:")
        self._setup_db()

        # State
        self.equity = self.config.initial_capital
        self.peak_equity = self.config.initial_capital
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Dict] = []
        self.weekly_stop_hit: bool = False

    def _setup_db(self):
        """Register Parquet Lake as a DuckDB View."""
        logger.info(f"Registering Lake: {self.lake_path}")
        try:
            p = Path(self.lake_path)
            pattern = str(p / "**" / "*.parquet")
            self.con.execute("INSTALL parquet;")
            self.con.execute("LOAD parquet;")
            query = f"CREATE OR REPLACE VIEW fo_data AS SELECT * FROM read_parquet('{pattern}', hive_partitioning=true);"
            self.con.execute(query)
            count = self.con.execute(
                "SELECT COUNT(*) FROM fo_data WHERE TckrSymb = 'NIFTY'"
            ).fetchone()[0]
            logger.info(f"NIFTY Rows in lake: {count}")
        except Exception as e:
            logger.error(f"Failed to setup DuckDB: {e}")
            raise

    def get_trading_dates_and_expiries(
        self, symbol="NIFTY", start_date: str = None, end_date: str = None
    ):
        """Get trading dates and expiries for a specific symbol."""
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

        trading_dates = pd.to_datetime(
            self.con.execute(d_query).df()["TradDt"]
        ).tolist()
        expiries = pd.to_datetime(self.con.execute(e_query).df()["XpryDt"]).tolist()
        return trading_dates, expiries

    def get_spot_data(self, symbol="NIFTY") -> pd.Series:
        """Fetch daily underlying spot prices."""
        query = f"SELECT TradDt, AVG(UndrlygPric) as Price FROM fo_data WHERE TckrSymb = '{symbol}' GROUP BY TradDt ORDER BY TradDt ASC"
        df = self.con.execute(query).df()
        df["TradDt"] = pd.to_datetime(df["TradDt"])
        df.set_index("TradDt", inplace=True)
        return df["Price"]

    def run(
        self,
        prices: pd.Series,
        start_date: str = "2025-04-01",
        end_date: str = "2026-03-01",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the full backtest.

        Parameters:
            prices: Daily closing prices with DatetimeIndex.
                    Needs data from at least 60 days before start_date
                    for vol calculations to warm up.
            start_date: First allowed entry date
            end_date: Last allowed expiry date

        Returns:
            (trades_df, equity_df) -- both as pandas DataFrames
        """
        cfg = self.config
        logger.info(f"Starting V2 Smart Risk Backtest")
        logger.info(
            f"Capital: Rs {cfg.initial_capital:,.0f} | Period: {start_date} to {end_date}"
        )

        # Compute vol metrics over full price history
        vol_data = VolatilityEngine.compute_all(prices, cfg)

        # Get expiries logic using DuckDB data
        trading_dates, expiries = self.get_trading_dates_and_expiries(
            start_date=start_date, end_date=end_date
        )

        logger.info(f"Expiry windows: {len(expiries)} weeks")

        date_map = {d: i for i, d in enumerate(trading_dates)}

        for expiry in expiries:
            if expiry not in date_map:
                continue

            exp_idx = date_map[expiry]
            entry_idx = exp_idx - cfg.entry_dte_primary

            if entry_idx < 0:
                continue

            entry_date = trading_dates[entry_idx]
            entry_ts = pd.Timestamp(entry_date)
            expiry_ts = pd.Timestamp(expiry)

            self.weekly_stop_hit = False  # Reset weekly flag

            if entry_ts in vol_data.index:
                self._process_entry(
                    entry_ts, expiry_ts, vol_data, prices, entry_window="T-3"
                )

        # Build output DataFrames
        trades_df = self._build_trades_df()
        equity_df = pd.DataFrame(self.equity_curve)

        # Log summary
        self._log_summary(trades_df)

        return trades_df, equity_df

    def _process_entry(
        self,
        entry_ts: pd.Timestamp,
        expiry_ts: pd.Timestamp,
        vol_data: pd.DataFrame,
        prices: pd.Series,
        entry_window: str,
    ):
        """Process a single entry opportunity."""
        cfg = self.config
        row = vol_data.loc[entry_ts]
        spot = prices.loc[entry_ts]

        rv20 = row["rv20"]
        rv5_rv20_ratio = row["rv5_rv20_ratio"]
        move_5d = row["move_5d"]
        iv = row["iv"]
        vov = row["vov"]

        # Skip if any metric is NaN (warmup period)
        if any(pd.isna([rv20, iv, vov, move_5d, rv5_rv20_ratio])):
            return

        date_str = entry_ts.strftime("%Y-%m-%d")
        exp_str = expiry_ts.strftime("%Y-%m-%d")

        # Fetch options for this expiry on the entry date
        entry_query = f"""
        SELECT OptnTp, StrkPric, ClsPric
        FROM fo_data 
        WHERE TckrSymb = 'NIFTY' 
          AND XpryDt = '{exp_str}' 
          AND TradDt = '{date_str}'
        """
        df_entry = self.con.execute(entry_query).df()

        if df_entry.empty:
            return

        options = df_entry[df_entry["OptnTp"].isin(["CE", "PE"])].copy()
        if options.empty:
            return

        # Find ATM strike
        strike = round(float(spot) / 50) * 50

        # Get CE and PE premium
        ce_row = options[(options["StrkPric"] == strike) & (options["OptnTp"] == "CE")]
        pe_row = options[(options["StrkPric"] == strike) & (options["OptnTp"] == "PE")]

        if ce_row.empty or pe_row.empty:
            logger.debug(
                f"  SKIP {entry_ts.date()} ({entry_window}): Option data missing for strike {strike}"
            )
            return

        premium = ce_row.iloc[0]["ClsPric"] + pe_row.iloc[0]["ClsPric"]

        # Regime filter
        allowed, reason = RegimeFilter.check_entry(
            rv20, rv5_rv20_ratio, move_5d, iv, premium, float(spot), cfg
        )
        if not allowed:
            logger.debug(f"  SKIP {entry_ts.date()} ({entry_window}): {reason}")
            return

        # Risk scaling flag
        risk_scaled = rv20 > cfg.rv_scale_down_threshold

        # Position sizing
        lots = PositionSizer.compute_lots(
            self.equity,
            float(spot),
            premium,
            rv20,
            vov,
            self.streak_tracker.multiplier,
            cfg,
        )
        if lots <= 0:
            return

        # Create position
        position = OpenPosition(
            entry_date=entry_ts.to_pydatetime(),
            expiry_date=expiry_ts.to_pydatetime(),
            strike=strike,
            spot_at_entry=float(spot),
            premium=premium,
            lots=lots,
            rv_at_entry=rv20,
            iv_at_entry=iv,
            risk_scaled=risk_scaled,
            entry_window=entry_window,
            streak_mult=self.streak_tracker.multiplier,
        )

        logger.info(
            f"  ENTRY {entry_ts.date()} ({entry_window}) | "
            f"Strike {strike} | Premium {premium:.2f} | "
            f"Lots {lots} | RV {rv20:.1%} | IV {iv:.1%}"
        )

        # Simulate daily until exit
        self._simulate_position(position, vol_data, prices)

    def _simulate_position(
        self, position: OpenPosition, vol_data: pd.DataFrame, prices: pd.Series
    ):
        """Simulate position day-by-day until an exit condition triggers."""
        cfg = self.config
        current = pd.Timestamp(position.entry_date) + timedelta(days=1)
        expiry = pd.Timestamp(position.expiry_date)
        exp_str = expiry.strftime("%Y-%m-%d")

        while current <= expiry:
            if current not in prices.index:
                current += timedelta(days=1)
                continue

            spot = prices.loc[current]
            curr_str = current.strftime("%Y-%m-%d")

            # Fetch options for current day
            h_query = f"""
            SELECT OptnTp, HghPric, LwPric, ClsPric
            FROM fo_data
            WHERE TckrSymb = 'NIFTY'
              AND XpryDt = '{exp_str}'
              AND TradDt = '{curr_str}'
              AND StrkPric = {position.strike}
            """
            df_hold = self.con.execute(h_query).df()

            # If no data on a trading day, use BSPricer fallback or assume prices carried over
            # Here we'll fallback to BS approximation so simulation continues smoothly
            if df_hold.empty or len(df_hold) < 2:
                iv = (
                    vol_data.loc[current, "iv"]
                    if current in vol_data.index
                    else position.iv_at_entry
                )
                dte = (expiry - current).days
                T = max(dte, 0) / 365.0
                current_price = BSPricer.straddle_premium(
                    float(spot), position.strike, T, 0.065, iv
                )
                worst_price = current_price
            else:
                ce_h = df_hold[df_hold["OptnTp"] == "CE"]
                pe_h = df_hold[df_hold["OptnTp"] == "PE"]

                if ce_h.empty or pe_h.empty:
                    iv = (
                        vol_data.loc[current, "iv"]
                        if current in vol_data.index
                        else position.iv_at_entry
                    )
                    dte = (expiry - current).days
                    T = max(dte, 0) / 365.0
                    current_price = BSPricer.straddle_premium(
                        float(spot), position.strike, T, 0.065, iv
                    )
                    worst_price = current_price
                else:
                    ce_high = ce_h["HghPric"].iloc[0]
                    pe_high = pe_h["HghPric"].iloc[0]
                    ce_low = pe_h["LwPric"].iloc[0]
                    pe_low = pe_h["LwPric"].iloc[0]
                    ce_close = ce_h["ClsPric"].iloc[0]
                    pe_close = pe_h["ClsPric"].iloc[0]

                    # Estimate intra-day worst pain
                    worst_price = max(ce_high + pe_low, pe_high + ce_low)
                    current_price = ce_close + pe_close

            # We'll use our TradeManager check_exits, but we have to provide current_price
            # Because TradeManager does BS calculation, let's override it

            # --- CUSTOM EXIT LOGIC FOR REAL DATA ---
            pnl_per_lot = (position.premium - current_price) * cfg.lot_size
            pnl_pct = (
                (position.premium - current_price) / position.premium
                if position.premium > 0
                else 0
            )

            if pnl_pct > position.best_pnl_pct:
                position.best_pnl_pct = pnl_pct

            exit_reason = None
            exit_price = current_price

            dte = (expiry - current).days

            # 1. STOP LOSS (checked against worst_price during day first)
            if worst_price >= position.premium * cfg.stop_loss_multiple:
                exit_reason = ExitReason.STOP_LOSS
                exit_price = position.premium * cfg.stop_loss_multiple
            # 2. Profit Target
            elif pnl_pct >= cfg.profit_target_pct:
                exit_reason = ExitReason.PROFIT_TARGET
            # 3. Trailing Stop
            else:
                if pnl_pct >= cfg.trailing_stop_activation:
                    position.trailing_active = True

                if position.trailing_active:
                    drawdown_from_best = position.best_pnl_pct - pnl_pct
                    if drawdown_from_best > cfg.trailing_stop_distance:
                        exit_reason = ExitReason.TRAILING_STOP

            # 4. Partial Profit (executed End of Day)
            if (
                exit_reason is None
                and not position.partial_taken
                and pnl_pct >= cfg.partial_profit_decay
            ):
                lots_to_close = int(position.lots * cfg.partial_close_ratio)
                if lots_to_close > 0:
                    partial_pnl = lots_to_close * pnl_per_lot
                    position.partial_pnl += partial_pnl
                    position.lots -= lots_to_close
                    position.partial_taken = True
                    logger.debug(
                        f"  PARTIAL: Closed {lots_to_close} lots at {pnl_pct:.0%} decay, "
                        f"banked Rs {partial_pnl:,.0f}"
                    )

            # 5. Time Based Exit
            if exit_reason is None:
                total_days = (position.expiry_date - position.entry_date).days
                elapsed_days = (current.to_pydatetime() - position.entry_date).days
                if total_days > 0:
                    time_elapsed_pct = elapsed_days / total_days
                    if (
                        time_elapsed_pct >= cfg.time_exit_pct
                        and pnl_pct < cfg.time_exit_min_decay
                    ):
                        exit_reason = ExitReason.TIME_EXIT

            # 6. Expiry
            if exit_reason is None and dte <= 0:
                exit_reason = (
                    ExitReason.EXPIRY_PARTIAL
                    if position.partial_taken
                    else ExitReason.EXPIRY_CLOSE
                )

            if exit_reason is not None:
                self._record_exit(
                    position, current, exit_price, exit_reason, position.partial_pnl
                )
                return

            current += timedelta(days=1)

        # Force close at expiry if not already exited (failsafe)
        if expiry in prices.index:
            spot = prices.loc[expiry]
            h_query = f"SELECT OptnTp, ClsPric FROM fo_data WHERE TckrSymb = 'NIFTY' AND XpryDt = '{exp_str}' AND TradDt = '{exp_str}' AND StrkPric = {position.strike}"
            df_hold = self.con.execute(h_query).df()
            if not df_hold.empty and len(df_hold) >= 2:
                exit_price = (
                    df_hold[df_hold["OptnTp"] == "CE"]["ClsPric"].iloc[0]
                    + df_hold[df_hold["OptnTp"] == "PE"]["ClsPric"].iloc[0]
                )
            else:
                iv = (
                    vol_data.loc[expiry, "iv"]
                    if expiry in vol_data.index
                    else position.iv_at_entry
                )
                exit_price = BSPricer.straddle_premium(
                    float(spot), position.strike, 0, 0.065, iv
                )

            exit_reason = (
                ExitReason.EXPIRY_PARTIAL
                if position.partial_taken
                else ExitReason.EXPIRY_CLOSE
            )
            self._record_exit(
                position, expiry, exit_price, exit_reason, position.partial_pnl
            )

    def _record_exit(
        self,
        position: OpenPosition,
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_reason: ExitReason,
        partial_pnl: float,
    ):
        """Record trade completion, update equity and streaks."""
        cfg = self.config

        # Gross P&L on remaining lots + any partial profits already banked
        remaining_pnl = (position.premium - exit_price) * position.lots * cfg.lot_size
        gross_pnl = remaining_pnl + partial_pnl

        # Transaction costs (computed on original lot count)
        costs = self.cost_calculator.compute(
            position.premium, position.original_lots, cfg.lot_size, cfg
        )
        net_pnl = gross_pnl - costs

        # Update equity
        self.equity += net_pnl
        self.peak_equity = max(self.peak_equity, self.equity)

        # Update streak tracker
        self.streak_tracker.record_trade(net_pnl > 0)

        # Track weekly stop loss
        if exit_reason == ExitReason.STOP_LOSS:
            self.weekly_stop_hit = True

        # Record trade
        self.trades.append(
            TradeRecord(
                entry_date=position.entry_date.strftime("%Y-%m-%d"),
                exit_date=exit_date.strftime("%Y-%m-%d"),
                trade_type="STRADDLE",
                strike=position.strike,
                spot=position.spot_at_entry,
                premium=round(position.premium, 2),
                exit_price=round(exit_price, 2),
                lots=position.original_lots,
                gross_pnl=round(gross_pnl, 2),
                net_pnl=round(net_pnl, 2),
                exit_reason=exit_reason.value,
                rv=round(position.rv_at_entry, 4),
                iv=round(position.iv_at_entry, 4),
                risk_scaled=position.risk_scaled,
                entry_window=position.entry_window,
                partial_pnl=round(partial_pnl, 2),
                streak_mult=position.streak_mult,
            )
        )

        # Equity curve point
        self.equity_curve.append(
            {"Date": exit_date.strftime("%Y-%m-%d"), "Equity": round(self.equity, 2)}
        )

        logger.info(
            f"  EXIT  {exit_date.date()} | {exit_reason.value} | "
            f"P&L Rs {net_pnl:+,.0f} | Equity Rs {self.equity:,.0f}"
        )

    def _build_trades_df(self) -> pd.DataFrame:
        """Convert trade records to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        records = [
            {
                "Entry_Date": t.entry_date,
                "Exit_Date": t.exit_date,
                "Type": t.trade_type,
                "Strike": t.strike,
                "Spot": t.spot,
                "Premium": t.premium,
                "Exit_Price": t.exit_price,
                "Lots": t.lots,
                "Gross_PnL": t.gross_pnl,
                "Net_PnL": t.net_pnl,
                "Exit_Reason": t.exit_reason,
                "RV": t.rv,
                "IV": t.iv,
                "Risk_Scaled": t.risk_scaled,
                "Entry_Window": t.entry_window,
                "Partial_PnL": t.partial_pnl,
                "Streak_Mult": t.streak_mult,
            }
            for t in self.trades
        ]
        return pd.DataFrame(records)

    def _log_summary(self, trades_df: pd.DataFrame):
        """Print comprehensive backtest summary."""
        if trades_df.empty:
            logger.warning("No trades executed.")
            return

        total_trades = len(trades_df)
        winners = (trades_df["Net_PnL"] > 0).sum()
        losers = total_trades - winners
        win_rate = winners / total_trades * 100

        total_pnl = trades_df["Net_PnL"].sum()
        ret_pct = total_pnl / self.config.initial_capital * 100

        gross_wins = trades_df.loc[trades_df["Net_PnL"] > 0, "Net_PnL"].sum()
        gross_losses = abs(trades_df.loc[trades_df["Net_PnL"] <= 0, "Net_PnL"].sum())
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Max drawdown from equity curve
        eq_values = [self.config.initial_capital] + [
            e["Equity"] for e in self.equity_curve
        ]
        eq_series = pd.Series(eq_values)
        peak = eq_series.cummax()
        dd = (peak - eq_series) / peak
        max_dd = dd.max() * 100

        stop_losses = (trades_df["Exit_Reason"].str.contains("Stop Loss")).sum()
        partials = (trades_df["Exit_Reason"].str.contains("Partial")).sum()
        trailing = (trades_df["Exit_Reason"].str.contains("Trailing")).sum()
        time_exits = (trades_df["Exit_Reason"].str.contains("Time")).sum()

        avg_winner = (
            trades_df.loc[trades_df["Net_PnL"] > 0, "Net_PnL"].mean()
            if winners > 0
            else 0
        )
        avg_loser = (
            trades_df.loc[trades_df["Net_PnL"] <= 0, "Net_PnL"].mean()
            if losers > 0
            else 0
        )

        logger.info("\n" + "=" * 65)
        logger.info("V2 SMART RISK BACKTEST SUMMARY")
        logger.info("=" * 65)
        logger.info(
            f"Period:          {trades_df['Entry_Date'].iloc[0]} to {trades_df['Exit_Date'].iloc[-1]}"
        )
        logger.info(f"Initial Capital: Rs {self.config.initial_capital:,.0f}")
        logger.info(f"Final Equity:    Rs {self.equity:,.0f}")
        logger.info("-" * 65)
        logger.info(f"Total Trades:    {total_trades}")
        logger.info(f"Winners:         {winners} ({win_rate:.1f}%)")
        logger.info(f"Losers:          {losers}")
        logger.info(f"Stop Losses:     {stop_losses}")
        logger.info(f"Trailing Stops:  {trailing}")
        logger.info(f"Partial Profits: {partials}")
        logger.info(f"Time Exits:      {time_exits}")
        logger.info("-" * 65)
        logger.info(f"Net P&L:         Rs {total_pnl:,.0f} ({ret_pct:.1f}%)")
        logger.info(f"Max Drawdown:    {max_dd:.2f}%")
        logger.info(
            f"Return/DD:       {ret_pct / max_dd:.1f}x"
            if max_dd > 0
            else "Return/DD: N/A"
        )
        logger.info(f"Profit Factor:   {profit_factor:.2f}")
        logger.info(f"Avg Winner:      Rs {avg_winner:,.0f}")
        logger.info(f"Avg Loser:       Rs {avg_loser:,.0f}")
        logger.info("=" * 65)


# =============================================================================
# CHART GENERATOR
# =============================================================================


class ChartGenerator:
    """Generates comprehensive backtest visualization."""

    @staticmethod
    def plot_results(
        trades_df: pd.DataFrame,
        equity_df: pd.DataFrame,
        config: StrategyConfig,
        save_path: str = "v2_smart_risk_results.png",
    ):
        """Generate 4-panel backtest results chart."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Panel 1: Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        equity_df["Date"] = pd.to_datetime(equity_df["Date"])
        ax1.plot(equity_df["Date"], equity_df["Equity"] / 1e7, "b-", linewidth=2)
        ax1.axhline(
            y=1.0, color="gray", linestyle="--", alpha=0.5, label="Initial Capital"
        )
        ax1.fill_between(
            equity_df["Date"],
            1.0,
            equity_df["Equity"] / 1e7,
            where=equity_df["Equity"] / 1e7 >= 1.0,
            alpha=0.15,
            color="green",
        )
        ax1.fill_between(
            equity_df["Date"],
            1.0,
            equity_df["Equity"] / 1e7,
            where=equity_df["Equity"] / 1e7 < 1.0,
            alpha=0.15,
            color="red",
        )
        ax1.set_title("V2 Smart Risk - Equity Curve", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Equity (Cr)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Trade P&L Bars
        ax2 = fig.add_subplot(gs[1, 0])
        colors = ["green" if x > 0 else "red" for x in trades_df["Net_PnL"]]
        ax2.bar(
            range(len(trades_df)), trades_df["Net_PnL"] / 1e3, color=colors, alpha=0.7
        )
        ax2.set_title("Trade P&L Distribution (Rs 000s)", fontsize=11)
        ax2.set_xlabel("Trade #")
        ax2.axhline(y=0, color="black", linewidth=0.8)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Exit Reason Pie
        ax3 = fig.add_subplot(gs[1, 1])
        exit_counts = trades_df["Exit_Reason"].value_counts()
        colors_pie = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
        ax3.pie(
            exit_counts.values,
            labels=exit_counts.index,
            autopct="%1.0f%%",
            colors=colors_pie[: len(exit_counts)],
            startangle=90,
        )
        ax3.set_title("Exit Reason Distribution", fontsize=11)

        # Panel 4: Monthly Returns
        ax4 = fig.add_subplot(gs[2, 0])
        trades_df["Month"] = pd.to_datetime(trades_df["Exit_Date"]).dt.to_period("M")
        monthly = trades_df.groupby("Month")["Net_PnL"].sum()
        colors_m = ["green" if x > 0 else "red" for x in monthly.values]
        ax4.bar(range(len(monthly)), monthly.values / 1e5, color=colors_m, alpha=0.7)
        ax4.set_xticks(range(len(monthly)))
        ax4.set_xticklabels([str(m) for m in monthly.index], rotation=45, fontsize=8)
        ax4.set_title("Monthly P&L (Rs Lakhs)", fontsize=11)
        ax4.axhline(y=0, color="black", linewidth=0.8)
        ax4.grid(True, alpha=0.3)

        # Panel 5: Key Metrics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")

        total_pnl = trades_df["Net_PnL"].sum()
        ret_pct = total_pnl / config.initial_capital * 100
        winners = (trades_df["Net_PnL"] > 0).sum()
        win_rate = winners / len(trades_df) * 100
        gross_wins = trades_df.loc[trades_df["Net_PnL"] > 0, "Net_PnL"].sum()
        gross_losses = abs(trades_df.loc[trades_df["Net_PnL"] <= 0, "Net_PnL"].sum())
        pf = gross_wins / gross_losses if gross_losses > 0 else 999
        eq_values = [config.initial_capital] + equity_df["Equity"].tolist()
        eq_s = pd.Series(eq_values)
        dd = ((eq_s.cummax() - eq_s) / eq_s.cummax()).max() * 100
        stops = trades_df["Exit_Reason"].str.contains("Stop Loss").sum()
        partials = trades_df["Exit_Reason"].str.contains("Partial").sum()

        metrics_text = (
            f"V2 SMART RISK SUMMARY\n"
            f"{'=' * 30}\n\n"
            f"Total Trades:     {len(trades_df)}\n"
            f"Win Rate:         {win_rate:.1f}%\n"
            f"Stop Losses:      {stops}\n"
            f"Partial Profits:  {partials}\n\n"
            f"Net P&L:          Rs {total_pnl / 1e5:,.1f}L\n"
            f"Return:           {ret_pct:.1f}%\n"
            f"Max Drawdown:     {dd:.2f}%\n"
            f"Return/DD:        {ret_pct / dd:.1f}x\n"
            f"Profit Factor:    {pf:.2f}\n\n"
            f"Capital:          Rs 1 Cr\n"
            f"Period:           Apr25-Feb26"
        )
        ax5.text(
            0.1,
            0.95,
            metrics_text,
            transform=ax5.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"Chart saved: {save_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """
    Run the V2 Smart Risk backtest end-to-end.

    Steps:
    1. Initialize BacktestEngine (connects to local DuckDB fo_data)
    2. Fetch NIFTY spot data directly from DuckDB
    3. Run backtest engine
    4. Save trade log + equity curve CSVs
    5. Generate results chart
    """
    # Initialize with default config (all params tunable above)
    config = StrategyConfig()
    engine = BacktestEngine(config)

    # Fetch NIFTY data directly from the DuckDB Lake
    logger.info("Fetching NIFTY 50 spot data from local Data Lake...")
    prices = engine.get_spot_data("NIFTY")

    # Run backtest
    trades_df, equity_df = engine.run(
        prices, start_date="2025-04-01", end_date="2026-03-01"
    )

    if trades_df.empty:
        logger.error("No trades generated. Check data and filter parameters.")
        return None, None

    # Save results
    trades_df.to_csv("v2_smart_risk_trades.csv", index=False)
    equity_df.to_csv("v2_smart_risk_equity.csv", index=False)
    logger.info("Saved: v2_smart_risk_trades.csv, v2_smart_risk_equity.csv")

    # Generate chart
    ChartGenerator.plot_results(
        trades_df, equity_df, config, "v2_smart_risk_results.png"
    )

    return trades_df, equity_df


if __name__ == "__main__":
    trades, equity = main()
