#!/usr/bin/env python3
"""
=============================================================================
V3 SMART RISK - Short Volatility Strategy for NIFTY Weekly Options
=============================================================================
Calibrated for real NSE bhavcopy data. Fixes V2's 19.28% drawdown problem.

Key V3 changes over V2:
  1. Tighter stop loss (1.5x vs 1.8x)
  2. Hard lot cap (150 max)
  3. 1-week cooldown after any stop loss
  4. Aggressive streak reduction (50% after 2 losses, 25% after 3)
  5. Premium cap (reject entries > 350 combined premium)
  6. Drawdown circuit breaker (halt at 10% DD, recovery mode at 7%)
  7. Max 1 stop per calendar week
  8. Wider trailing stop (activates at 25% profit, trails 20%)
  9. Earlier time exit (50% time elapsed, 20% min decay)
  10. Weekly loss cap tracking

Backtest: Apr 2025 - Mar 2026 on NSE FO Bhavcopy via DuckDB
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
    """All tunable parameters in one place. V3 calibrated for real NSE data."""

    # --- Capital & Sizing ---
    initial_capital: float = 10_000_000
    lot_size: int = 25
    margin_per_lot_pct: float = 0.0065
    max_portfolio_margin_pct: float = 0.65
    base_risk_pct: float = 0.05
    max_lots_cap: int = 150  # V3: Hard lot cap

    # --- Regime Filters ---
    rv_regime_threshold: float = 0.45
    vol_acceleration_limit: float = 1.55
    panic_move_threshold: float = 0.045
    vrp_min_ratio: float = 1.05
    min_yield_pct: float = 0.005
    max_premium_cap: float = 350.0  # V3: Reject high premium entries

    # --- Volatility Windows ---
    rv_window_short: int = 5
    rv_window_long: int = 20
    iv_lookback: int = 20

    # --- Exit Rules ---
    stop_loss_multiple: float = 1.5  # V3: Tighter (was 1.8)
    profit_target_pct: float = 0.70
    partial_profit_decay: float = 0.50
    partial_close_ratio: float = 0.42
    trailing_stop_activation: float = 0.25  # V3: Lower activation (was 0.40)
    trailing_stop_distance: float = 0.20  # V3: Tighter trail (was 0.30)
    time_exit_pct: float = 0.50  # V3: Earlier (was 0.60)
    time_exit_min_decay: float = 0.20  # V3: Lower threshold (was 0.30)

    # --- Streak-Aware Sizing ---
    streak_boost_after: int = 3  # V3: Need 3 wins to boost (was 2)
    streak_boost_mult: float = 1.15  # V3: Smaller boost (was 1.2)
    streak_reduce_after: int = 2
    streak_reduce_mult: float = 0.50  # V3: Aggressive cut (was 0.7)
    streak_reduce_3_mult: float = 0.25  # V3 NEW: 25% after 3 losses

    # --- Risk Scaling ---
    rv_scale_down_threshold: float = 0.25
    rv_scale_down_factor: float = 0.60
    correlation_shock_window: int = 5
    vov_lookback: int = 10
    vov_threshold: float = 1.5
    vov_scale_factor: float = 0.70

    # --- Drawdown Protection (V3 NEW) ---
    dd_circuit_breaker_pct: float = 0.10
    dd_recovery_mode_pct: float = 0.07
    dd_recovery_scale: float = 0.50

    # --- Cooldown (V3 NEW) ---
    cooldown_days_after_stop: int = 7
    max_stops_per_week: int = 1

    # --- Transaction Costs ---
    brokerage_per_lot: float = 25.0
    stt_pct: float = 0.0005
    slippage_pct: float = 0.001

    # --- Entry Windows ---
    entry_dte_primary: int = 3
    skip_t1_after_stop: bool = True


class ExitReason(Enum):
    STOP_LOSS = "Stop Loss (1.5x)"
    PROFIT_TARGET = "Profit Target (70%)"
    TRAILING_STOP = "Trailing Stop"
    TIME_EXIT = "Time Exit (stale)"
    EXPIRY_CLOSE = "Expiry Close"
    EXPIRY_PARTIAL = "Expiry Close +Partial"


# =============================================================================
# VOLATILITY ENGINE
# =============================================================================


class VolatilityEngine:
    @staticmethod
    def realized_volatility(prices: pd.Series, window: int) -> pd.Series:
        log_returns = np.log(prices / prices.shift(1))
        rv = log_returns.rolling(window).std() * np.sqrt(252)
        return rv

    @staticmethod
    def implied_volatility_proxy(prices: pd.Series, window: int = 20) -> pd.Series:
        log_returns = np.log(prices / prices.shift(1))
        rv = log_returns.rolling(window).std() * np.sqrt(252)
        iv_proxy = rv * 1.25 + 0.01
        return iv_proxy

    @staticmethod
    def vol_of_vol(rv_series: pd.Series, lookback: int = 10) -> pd.Series:
        return rv_series.rolling(lookback).std() / rv_series.rolling(lookback).mean()

    @staticmethod
    def compute_all(prices: pd.Series, config: StrategyConfig) -> pd.DataFrame:
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
    @staticmethod
    def call_price(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return max(K - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def straddle_premium(S, K, T, r, sigma):
        return BSPricer.call_price(S, K, T, r, sigma) + BSPricer.put_price(
            S, K, T, r, sigma
        )


# =============================================================================
# REGIME FILTER (V3: Added premium cap)
# =============================================================================


class RegimeFilter:
    @staticmethod
    def check_entry(rv20, rv5_rv20_ratio, move_5d, iv, premium, spot, config):
        if rv20 > config.rv_regime_threshold:
            return False, f"RV too high: {rv20:.1%}"
        if rv5_rv20_ratio > config.vol_acceleration_limit:
            return False, f"Vol accelerating: {rv5_rv20_ratio:.2f}"
        if move_5d > config.panic_move_threshold:
            return False, f"Panic move: {move_5d:.1%}"
        if iv < config.vrp_min_ratio * rv20:
            return False, f"VRP thin: {iv / max(rv20, 0.01):.2f}x"
        yield_pct = premium / spot if spot > 0 else 0
        if yield_pct < config.min_yield_pct:
            return False, f"Yield too low: {yield_pct:.3%}"
        if premium > config.max_premium_cap:
            return (
                False,
                f"Premium too high: {premium:.1f} > {config.max_premium_cap:.0f}",
            )
        return True, "All filters passed"


# =============================================================================
# POSITION SIZER (V3: lot cap + DD scaling)
# =============================================================================


class PositionSizer:
    @staticmethod
    def compute_lots(equity, spot, premium, rv20, vov, streak_mult, dd_scale, config):
        risk_pct = config.base_risk_pct
        risk_scale = 1.0
        if rv20 > config.rv_scale_down_threshold:
            risk_scale *= config.rv_scale_down_factor
        if vov > config.vov_threshold:
            risk_scale *= config.vov_scale_factor
        effective_risk = risk_pct * risk_scale * streak_mult * dd_scale
        capital_for_trade = equity * effective_risk
        margin_per_lot = spot * config.margin_per_lot_pct * config.lot_size
        if margin_per_lot <= 0:
            return 0
        lots = int(capital_for_trade / margin_per_lot)
        max_lots = int((equity * config.max_portfolio_margin_pct) / margin_per_lot)
        lots = min(lots, max_lots)
        lots = min(lots, config.max_lots_cap)
        return max(lots, 0)


# =============================================================================
# OPEN POSITION + TRADE MANAGER
# =============================================================================


@dataclass
class OpenPosition:
    entry_date: datetime
    expiry_date: datetime
    strike: float
    spot_at_entry: float
    premium: float
    lots: int
    rv_at_entry: float
    iv_at_entry: float
    risk_scaled: bool
    entry_window: str
    streak_mult: float
    best_pnl_pct: float = 0.0
    trailing_active: bool = False
    partial_taken: bool = False
    partial_pnl: float = 0.0
    original_lots: int = 0

    def __post_init__(self):
        self.original_lots = self.lots


class TradeManager:
    def __init__(self, config):
        self.config = config


# =============================================================================
# STREAK TRACKER (V3: 3-tier)
# =============================================================================


class StreakTracker:
    def __init__(self, config):
        self.config = config
        self.consecutive_wins = 0
        self.consecutive_losses = 0

    def record_trade(self, is_win):
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    @property
    def multiplier(self):
        if self.consecutive_losses >= 3:
            return self.config.streak_reduce_3_mult
        elif self.consecutive_losses >= self.config.streak_reduce_after:
            return self.config.streak_reduce_mult
        elif self.consecutive_wins >= self.config.streak_boost_after:
            return self.config.streak_boost_mult
        return 1.0


# =============================================================================
# UTILITIES
# =============================================================================


class ExpiryCalendar:
    @staticmethod
    def get_weekly_expiries(start, end):
        expiries = []
        current = start
        while current <= end:
            if current.weekday() == 3:
                expiries.append(current)
            current += timedelta(days=1)
        return expiries

    @staticmethod
    def get_entry_date(expiry, dte):
        return expiry - timedelta(days=dte)


class CostCalculator:
    @staticmethod
    def compute(premium, lots, lot_size, config):
        notional = premium * lots * lot_size
        brokerage = config.brokerage_per_lot * lots * 2
        stt = notional * config.stt_pct
        slippage = notional * config.slippage_pct
        return brokerage + stt + slippage


@dataclass
class TradeRecord:
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


# =============================================================================
# BACKTEST ENGINE (V3: Cooldown, DD breaker, weekly stop tracking)
# =============================================================================


class BacktestEngine:
    def __init__(self, config=None):
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
        self.weekly_stop_hit = False

        # V3 NEW state
        self.last_stop_date: Optional[datetime] = None
        self.weekly_stop_count: Dict[str, int] = {}
        self.circuit_breaker_active = False

    def _setup_db(self):
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
        self, symbol="NIFTY", start_date=None, end_date=None
    ):
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

    def get_spot_data(self, symbol="NIFTY"):
        query = f"SELECT TradDt, AVG(UndrlygPric) as Price FROM fo_data WHERE TckrSymb = '{symbol}' GROUP BY TradDt ORDER BY TradDt ASC"
        df = self.con.execute(query).df()
        df["TradDt"] = pd.to_datetime(df["TradDt"])
        df.set_index("TradDt", inplace=True)
        return df["Price"]

    def _get_dd_scale(self):
        """V3: Compute drawdown-based position scaling."""
        if self.peak_equity <= 0:
            return 1.0
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        if current_dd >= self.config.dd_circuit_breaker_pct:
            self.circuit_breaker_active = True
            return 0.0
        if current_dd >= self.config.dd_recovery_mode_pct:
            return self.config.dd_recovery_scale
        if (
            self.circuit_breaker_active
            and current_dd < self.config.dd_recovery_mode_pct
        ):
            self.circuit_breaker_active = False
            logger.info("  CIRCUIT BREAKER RESET - Resuming normal trading")
        return 1.0

    def _is_in_cooldown(self, entry_date):
        """V3: Check if we're still in post-stop cooldown."""
        if self.last_stop_date is None:
            return False
        days_since_stop = (entry_date - self.last_stop_date).days
        return days_since_stop < self.config.cooldown_days_after_stop

    def _week_key(self, dt):
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"

    def _can_take_stop(self, current_date):
        """V3: Check if we've already hit max stops this week."""
        wk = self._week_key(current_date)
        return self.weekly_stop_count.get(wk, 0) < self.config.max_stops_per_week

    def run(self, prices, start_date="2025-04-01", end_date="2026-03-01"):
        cfg = self.config
        logger.info("Starting V3 Smart Risk Backtest")
        logger.info(
            f"Capital: Rs {cfg.initial_capital:,.0f} | Period: {start_date} to {end_date}"
        )

        vol_data = VolatilityEngine.compute_all(prices, cfg)
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
            self.weekly_stop_hit = False
            if entry_ts in vol_data.index:
                self._process_entry(entry_ts, expiry_ts, vol_data, prices, "T-3")

        trades_df = self._build_trades_df()
        equity_df = pd.DataFrame(self.equity_curve)
        self._log_summary(trades_df)
        return trades_df, equity_df

    def _process_entry(self, entry_ts, expiry_ts, vol_data, prices, entry_window):
        cfg = self.config
        row = vol_data.loc[entry_ts]
        spot = prices.loc[entry_ts]
        rv20 = row["rv20"]
        rv5_rv20_ratio = row["rv5_rv20_ratio"]
        move_5d = row["move_5d"]
        iv = row["iv"]
        vov = row["vov"]

        if any(pd.isna([rv20, iv, vov, move_5d, rv5_rv20_ratio])):
            return

        date_str = entry_ts.strftime("%Y-%m-%d")
        exp_str = expiry_ts.strftime("%Y-%m-%d")

        # V3: Circuit breaker check
        if self.circuit_breaker_active:
            dd_scale = self._get_dd_scale()
            if dd_scale == 0.0:
                logger.info(
                    f"  SKIP {entry_ts.date()} ({entry_window}): Circuit breaker active"
                )
                return

        # V3: Cooldown check
        if self._is_in_cooldown(entry_ts.to_pydatetime()):
            days_left = (
                cfg.cooldown_days_after_stop
                - (entry_ts.to_pydatetime() - self.last_stop_date).days
            )
            logger.info(
                f"  SKIP {entry_ts.date()} ({entry_window}): Cooldown ({days_left}d remaining)"
            )
            return

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

        strike = round(float(spot) / 50) * 50
        ce_row = options[(options["StrkPric"] == strike) & (options["OptnTp"] == "CE")]
        pe_row = options[(options["StrkPric"] == strike) & (options["OptnTp"] == "PE")]
        if ce_row.empty or pe_row.empty:
            return

        premium = ce_row.iloc[0]["ClsPric"] + pe_row.iloc[0]["ClsPric"]

        allowed, reason = RegimeFilter.check_entry(
            rv20, rv5_rv20_ratio, move_5d, iv, premium, float(spot), cfg
        )
        if not allowed:
            logger.debug(f"  SKIP {entry_ts.date()} ({entry_window}): {reason}")
            return

        risk_scaled = rv20 > cfg.rv_scale_down_threshold
        dd_scale = self._get_dd_scale()
        if dd_scale == 0.0:
            logger.info(f"  SKIP {entry_ts.date()}: Circuit breaker during sizing")
            return

        lots = PositionSizer.compute_lots(
            self.equity,
            float(spot),
            premium,
            rv20,
            vov,
            self.streak_tracker.multiplier,
            dd_scale,
            cfg,
        )
        if lots <= 0:
            return

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
        self._simulate_position(position, vol_data, prices)

    def _simulate_position(self, position, vol_data, prices):
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

            h_query = f"""
            SELECT OptnTp, HghPric, LwPric, ClsPric
            FROM fo_data
            WHERE TckrSymb = 'NIFTY'
              AND XpryDt = '{exp_str}'
              AND TradDt = '{curr_str}'
              AND StrkPric = {position.strike}
            """
            df_hold = self.con.execute(h_query).df()

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
                    ce_low = ce_h["LwPric"].iloc[0]
                    pe_low = pe_h["LwPric"].iloc[0]
                    ce_close = ce_h["ClsPric"].iloc[0]
                    pe_close = pe_h["ClsPric"].iloc[0]
                    worst_price = max(ce_high + pe_low, pe_high + ce_low)
                    current_price = ce_close + pe_close

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

            # 1. STOP LOSS (V3: 1.5x, weekly cap)
            if worst_price >= position.premium * cfg.stop_loss_multiple:
                if self._can_take_stop(current.to_pydatetime()):
                    exit_reason = ExitReason.STOP_LOSS
                    exit_price = position.premium * cfg.stop_loss_multiple
                else:
                    exit_reason = ExitReason.TIME_EXIT
                    exit_price = current_price

            # 2. Profit Target
            elif pnl_pct >= cfg.profit_target_pct:
                exit_reason = ExitReason.PROFIT_TARGET

            # 3. Trailing Stop (V3: 25% activation, 20% trail)
            else:
                if pnl_pct >= cfg.trailing_stop_activation:
                    position.trailing_active = True
                if position.trailing_active:
                    drawdown_from_best = position.best_pnl_pct - pnl_pct
                    if drawdown_from_best > cfg.trailing_stop_distance:
                        exit_reason = ExitReason.TRAILING_STOP

            # 4. Partial Profit
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

            # 5. Time Exit (V3: 50% time, 20% min decay)
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

        # Failsafe: force close at expiry
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

    def _record_exit(self, position, exit_date, exit_price, exit_reason, partial_pnl):
        cfg = self.config
        remaining_pnl = (position.premium - exit_price) * position.lots * cfg.lot_size
        gross_pnl = remaining_pnl + partial_pnl
        costs = self.cost_calculator.compute(
            position.premium, position.original_lots, cfg.lot_size, cfg
        )
        net_pnl = gross_pnl - costs

        self.equity += net_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.streak_tracker.record_trade(net_pnl > 0)

        # V3: Track stop loss cooldown and weekly stops
        if exit_reason == ExitReason.STOP_LOSS:
            self.weekly_stop_hit = True
            self.last_stop_date = (
                exit_date.to_pydatetime()
                if isinstance(exit_date, pd.Timestamp)
                else exit_date
            )
            wk = self._week_key(self.last_stop_date)
            self.weekly_stop_count[wk] = self.weekly_stop_count.get(wk, 0) + 1
            logger.info(
                f"  >> COOLDOWN ACTIVATED: No entries for {cfg.cooldown_days_after_stop} days"
            )

        exit_date_str = (
            exit_date.strftime("%Y-%m-%d")
            if hasattr(exit_date, "strftime")
            else str(exit_date)
        )

        self.trades.append(
            TradeRecord(
                entry_date=position.entry_date.strftime("%Y-%m-%d"),
                exit_date=exit_date_str,
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

        self.equity_curve.append(
            {"Date": exit_date_str, "Equity": round(self.equity, 2)}
        )

        logger.info(
            f"  EXIT  {exit_date.date() if hasattr(exit_date, 'date') else exit_date} | {exit_reason.value} | "
            f"P&L Rs {net_pnl:+,.0f} | Equity Rs {self.equity:,.0f}"
        )

    def _build_trades_df(self):
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

    def _log_summary(self, trades_df):
        if trades_df.empty:
            logger.warning("No trades executed.")
            return

        cfg = self.config
        total_trades = len(trades_df)
        winners = (trades_df["Net_PnL"] > 0).sum()
        losers = total_trades - winners
        win_rate = winners / total_trades * 100

        total_pnl = trades_df["Net_PnL"].sum()
        ret_pct = total_pnl / cfg.initial_capital * 100

        gross_wins = trades_df.loc[trades_df["Net_PnL"] > 0, "Net_PnL"].sum()
        gross_losses = abs(trades_df.loc[trades_df["Net_PnL"] <= 0, "Net_PnL"].sum())
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        eq_values = [cfg.initial_capital] + [e["Equity"] for e in self.equity_curve]
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
        logger.info("V3 SMART RISK BACKTEST SUMMARY")
        logger.info("=" * 65)
        logger.info(
            f"Period:          {trades_df['Entry_Date'].iloc[0]} to {trades_df['Exit_Date'].iloc[-1]}"
        )
        logger.info(f"Initial Capital: Rs {cfg.initial_capital:,.0f}")
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
        logger.info(
            f"V3 Params:       Stop={cfg.stop_loss_multiple}x | MaxLots={cfg.max_lots_cap} | Cooldown={cfg.cooldown_days_after_stop}d | DD Halt={cfg.dd_circuit_breaker_pct:.0%}"
        )
        logger.info("=" * 65)


# =============================================================================
# CHART GENERATOR
# =============================================================================


class ChartGenerator:
    @staticmethod
    def plot_results(
        trades_df, equity_df, config, save_path="v3_smart_risk_results.png"
    ):
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
        ax1.set_title("V3 Smart Risk - Equity Curve", fontsize=14, fontweight="bold")
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
        trades_df_copy = trades_df.copy()
        trades_df_copy["Month"] = pd.to_datetime(
            trades_df_copy["Exit_Date"]
        ).dt.to_period("M")
        monthly = trades_df_copy.groupby("Month")["Net_PnL"].sum()
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
        trailing = trades_df["Exit_Reason"].str.contains("Trailing").sum()

        metrics_text = (
            f"V3 SMART RISK SUMMARY\n"
            f"{'=' * 30}\n\n"
            f"Total Trades:     {len(trades_df)}\n"
            f"Win Rate:         {win_rate:.1f}%\n"
            f"Stop Losses:      {stops}\n"
            f"Trailing Stops:   {trailing}\n"
            f"Partial Profits:  {partials}\n\n"
            f"Net P&L:          Rs {total_pnl / 1e5:,.1f}L\n"
            f"Return:           {ret_pct:.1f}%\n"
            f"Max Drawdown:     {dd:.2f}%\n"
            f"Return/DD:        {ret_pct / dd:.1f}x\n"
            f"Profit Factor:    {pf:.2f}\n\n"
            f"Capital:          Rs 1 Cr\n"
            f"Stop: {config.stop_loss_multiple}x | MaxLots: {config.max_lots_cap}\n"
            f"Cooldown: {config.cooldown_days_after_stop}d | DD Halt: {config.dd_circuit_breaker_pct:.0%}"
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
    config = StrategyConfig()
    engine = BacktestEngine(config)

    logger.info("Fetching NIFTY 50 spot data from local Data Lake...")
    prices = engine.get_spot_data("NIFTY")

    trades_df, equity_df = engine.run(
        prices, start_date="2025-04-01", end_date="2026-03-01"
    )

    if trades_df.empty:
        logger.error("No trades generated. Check data and filter parameters.")
        return None, None

    trades_df.to_csv("v3_smart_risk_trades.csv", index=False)
    equity_df.to_csv("v3_smart_risk_equity.csv", index=False)
    logger.info("Saved: v3_smart_risk_trades.csv, v3_smart_risk_equity.csv")

    ChartGenerator.plot_results(
        trades_df, equity_df, config, "v3_smart_risk_results.png"
    )

    return trades_df, equity_df


if __name__ == "__main__":
    trades, equity = main()
