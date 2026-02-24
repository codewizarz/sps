"""
backtest_short_vol.py

Production-Grade Short Volatility Backtest Engine
-------------------------------------------------
Strategy:
    - Sell ATM Straddle (CE+PE) on every Weekly Expiry (Thursday).
    - Entry: T-0 (Expiry Day) Market Open (proxy via 'Open' price if avail, else 'Close' of prev day?
      Actually, let's use Expiry Day 'Open' price from the daily candle).
    - Exit: T-0 (Expiry Day) Market Close (Settlement).
    - Frequency: Weekly.

Technology:
    - DuckDB: Streaming SQL execution on Parquet Lake.
    - Zero-Memory Overhead: No pandas loading of full dataset.

Metrics:
    - Win Rate, Sharpe, Max Drawdown, Equity Curve.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Configuration
LAKE_PATH = str(
    Path(
        Path(__file__).resolve().parent.parent.parent.parent / "data" / "master_fo_lake"
    ).absolute()
)
DB_PATH = ":memory:"  # In-memory DuckDB
CAPITAL = 10_000_000  # 1 Crore
SLIPPAGE_PCT = 0.002  # 0.2% round trip
TRANSACTION_COST = 50  # Flat fee per leg
LOT_SIZE = 50  # NIFTY contract size
LOT_SMARGIN_PCT = 0.20  # 20% of notional

UNIVERSE = ["NIFTY", "BANKNIFTY", "FINNIFTY"]


def get_margin_pct(rv):
    """
    Dynamic SPAN approximation based on realized volatility.
    """
    if rv < 0.15:
        return 0.12  # Calm regime
    elif rv < 0.25:
        return 0.20  # Normal regime
    elif rv < 0.35:
        return 0.28  # Elevated risk
    else:
        return 0.40  # Crisis margin expansion


class PortfolioVolEngine:
    def __init__(self, lake_path=LAKE_PATH):
        self.lake_path = lake_path
        self.con = duckdb.connect(DB_PATH)
        self.active_symbols = []
        self.rv_series_map = {}
        self.equity = CAPITAL
        self.portfolio_returns = []

        self._setup_db()
        self._validate_universe()

    def _validate_universe(self):
        """Check which symbols are available in the Lake."""
        print("[Init] Validating Universe...")
        for sym in UNIVERSE:
            try:
                count = self.con.execute(
                    f"SELECT COUNT(*) FROM fo_data WHERE TckrSymb = '{sym}'"
                ).fetchone()[0]
                if count > 0:
                    print(f"  + {sym} (Rows: {count})")
                    self.active_symbols.append(sym)
                else:
                    print(f"  - {sym} (Missing)")
            except Exception as e:
                print(f"  - {sym} (Error: {e})")

        if not self.active_symbols:
            raise RuntimeError("No valid symbols found in Universe!")

    def _setup_db(self):
        """Register Parquet Lake as a DuckDB View."""
        print(f"[Init] Registering Lake: {self.lake_path}")

        try:
            # Robust Path Handling for Hive Partitioning
            # We target the root directory recursively for year=XXXX partitions
            p = Path(self.lake_path).absolute()
            pattern = str(p / "**" / "*.parquet")

            self.con.execute("INSTALL parquet;")
            self.con.execute("LOAD parquet;")

            # Create View
            query = f"""
            CREATE OR REPLACE VIEW fo_data AS 
            SELECT * FROM read_parquet('{pattern}', hive_partitioning=true);
            """
            self.con.execute(query)

            # Validation Check (Requested)
            # Immediately after registering the lake, run:
            count_nifty = self.con.execute(
                "SELECT COUNT(*) FROM fo_data WHERE TckrSymb = 'NIFTY'"
            ).fetchone()[0]

            if count_nifty == 0:
                raise RuntimeError(
                    "NIFTY not found in FO lake. Verify bhavcopy ingestion."
                )
            else:
                print(f"[Validation] NIFTY rows detected: {count_nifty}")

            # General Verify
            total_count = self.con.execute("SELECT count(*) FROM fo_data").fetchone()[0]
            print(f"[Init] Lake Registered. Total Rows: {total_count}")

        except Exception as e:
            raise RuntimeError(f"Failed to setup DuckDB: {e}")

    def get_trading_dates_and_expiries(self, symbol):
        """Get trading dates and expiries for a specific symbol."""
        # Dates
        d_query = f"""
        SELECT DISTINCT TradDt 
        FROM fo_data 
        WHERE TckrSymb = '{symbol}'
        ORDER BY TradDt ASC
        """
        trading_dates = pd.to_datetime(
            self.con.execute(d_query).df()["TradDt"]
        ).tolist()

        # Expiries
        e_query = f"""
        SELECT DISTINCT XpryDt 
        FROM fo_data 
        WHERE TckrSymb = '{symbol}'
        ORDER BY XpryDt ASC
        """
        expiries = pd.to_datetime(self.con.execute(e_query).df()["XpryDt"]).tolist()

        return trading_dates, expiries

    def _compute_portfolio_rv(self, silent=False):
        """Precompute RV for all active symbols."""
        if not silent:
            print("[Init] Computing Portfolio Volatility...")

        for sym in self.active_symbols:
            try:
                query = f"""
                SELECT TradDt, AVG(UndrlygPric) as Price
                FROM fo_data
                WHERE TckrSymb = '{sym}'
                GROUP BY TradDt
                ORDER BY TradDt ASC
                """
                df = self.con.execute(query).df()

                if df.empty:
                    continue

                df["TradDt"] = pd.to_datetime(df["TradDt"])
                df.set_index("TradDt", inplace=True)
                df["LogRet"] = np.log(df["Price"] / df["Price"].shift(1))
                df["Move_3"] = np.abs(np.log(df["Price"] / df["Price"].shift(3)))
                rv = df["LogRet"].rolling(window=10).std() * np.sqrt(252)
                rv5 = df["LogRet"].rolling(window=5).std() * np.sqrt(252)
                rv20 = df["LogRet"].rolling(window=20).std() * np.sqrt(252)

                self.rv_series_map[sym] = {
                    "RV": rv,
                    "RV5": rv5,
                    "RV20": rv20,
                    "LogRet": df["LogRet"],
                    "Move_3": df["Move_3"],
                }
                if not silent:
                    print(f"  + {sym} RV Computed ({len(rv)} points)")
            except Exception as e:
                if not silent:
                    print(f"  - {sym} RV Failed: {e}")

    def _compute_correlations(self):
        """Compute rolling correlation matrix (10d Short-Term & 60d Baseline)."""
        # Align all returns
        ret_data = {}
        for sym in self.active_symbols:
            if "LogRet" in self.rv_series_map.get(sym, {}):
                ret_data[sym] = self.rv_series_map[sym]["LogRet"]

        if not ret_data:
            self.corr_10 = pd.Series(dtype=float)
            self.corr_60 = pd.Series(dtype=float)
            return

        df_ret = pd.DataFrame(ret_data)

        # Helper for rolling max pairwise
        def get_rolling_max(window):
            rolling_corr = df_ret.rolling(window=window).corr()
            max_corr_series = pd.Series(index=df_ret.index, dtype=float)
            for date, frame in rolling_corr.groupby(level=0):
                vals = frame.values
                np.fill_diagonal(vals, 0)
                max_corr_series.loc[date] = np.max(vals)
            return max_corr_series

        self.corr_10 = get_rolling_max(10)
        self.corr_60 = get_rolling_max(60)

    def _compute_vov(self):
        """Compute VoV metrics for each symbol."""
        # Using RV series as the 'Volatility' input since we don't have historical IV.
        # VoV = StdDev(daily vol changes over last 20 sessions)
        # Vol_5d_Change = (Current - 5d_ago) / 5d_ago

        for sym, data in self.rv_series_map.items():
            if "RV" not in data:
                continue

            # 1. Get Vol Series (using RV)
            vol_series = data["RV"]

            # 2. Compute 5-day % Change
            # Shift 5 days
            vol_change_5d = vol_series.pct_change(periods=5)

            # 3. Compute VoV
            # Daily change
            vol_change_1d = vol_series.diff()

            # Clean Data (Drop NaNs/Infs for calculation safety)
            # Rolling operations handle NaNs (skip), but Infs need to be removed or replaced.
            vol_change_1d = vol_change_1d.replace([np.inf, -np.inf], np.nan)

            vov = vol_change_1d.rolling(window=20).std()

            # 4. Compute Percentile (1-year rolling)
            # 252 days. "history length < 200 -> SKIP".
            # "VoV percentile MUST NOT be computed unless we have MIN 252...".
            # We use min_periods=252 to enforce warmup.
            # We also track Count to validate history length at runtime.

            vov_90 = vov.rolling(window=252, min_periods=252).quantile(0.90)
            vov_count = vov.rolling(window=252).count()

            self.rv_series_map[sym]["VolChange5d"] = vol_change_5d
            self.rv_series_map[sym]["VoV"] = vov
            self.rv_series_map[sym]["VoV_90"] = vov_90
            self.rv_series_map[sym]["VoV_Count"] = vov_count

    def simulate_portfolio(self, rv_threshold=0.35, verbose=True):
        # 1. Initialization
        if verbose:
            self._compute_portfolio_rv()
            self._compute_correlations()
            self._compute_vov()
        else:
            # Silent compute
            self._compute_portfolio_rv(silent=True)
            self._compute_correlations()
            self._compute_vov()

        # 2. Build Master Timeline & Schedule
        if verbose:
            print("[Backtest] Building Portfolio Schedule...")
        master_dates = set()
        symbol_schedules = {}  # sym -> {entry_date: (expiry_date, expiry_obj)}

        for sym in self.active_symbols:
            td, exps = self.get_trading_dates_and_expiries(sym)
            master_dates.update(td)

            # Map Dates -> Indices for lookback
            date_map = {d: i for i, d in enumerate(td)}
            entry_map = {}

            for exp in exps:
                if exp not in date_map:
                    continue
                exp_idx = date_map[exp]

                # Entry: T-3
                entry_idx = exp_idx - 3
                if entry_idx < 0:
                    continue

                entry_date = td[entry_idx]
                entry_map[entry_date] = exp

            symbol_schedules[sym] = entry_map

        sorted_dates = sorted(list(master_dates))
        if verbose:
            print(f"[Backtest] Simulation Start: {len(sorted_dates)} sessions")

        # 3. Simulation Loop
        open_positions = []  # list of dicts
        closed_trades = []
        equity_curve = []

        # Portfolio Safety State
        current_regime = "NORMAL"
        blocked_trades = 0
        blocked_acceleration = 0

        # Correlation Risk State
        # Metrics
        shock_days = 0
        trades_scaled = 0
        blocked_margin = 0
        vov_triggers = 0

        # Post-Shock Cooldown State
        cooldown_active = False
        cooldown_days_remaining = 0
        cooldown_activations = 0

        # Daily Portfolio Kill Switch State
        kill_switch_active = False
        kill_switch_activations = 0
        day_start_equity = self.equity

        # VoV Warmup Tracking
        vov_warmup_logged = False

        # Crisis Long Vol Tracking
        crisis_trades = 0

        t0_start = time.perf_counter()

        for curr_date in sorted_dates:
            # --- START OF DAY: RESET KILL SWITCH ---
            day_start_equity = self.equity
            kill_switch_active = False
            vov_warmup_logged = False

            # --- UPDATE COOLDOWN STATE ---
            if cooldown_active:
                cooldown_days_remaining -= 1
                if cooldown_days_remaining <= 0:
                    cooldown_active = False
                    if verbose:
                        print(f"[POST-SHOCK MODE] Cooldown Expired — Risk Restored")

            date_str = curr_date.strftime("%Y-%m-%d")

            # --- A. Update Metrics & Check Exits ---
            # Release margin first

            active_pos_next = []

            for pos in open_positions:
                sym = pos["Symbol"]
                expiry = pos["Expiry"]

                # Fetch OHLC for this Position (Strike)
                # We need Option Prices
                h_query = f"""
                SELECT OptnTp, OpnPric, HghPric, LwPric, ClsPric
                FROM fo_data
                WHERE TckrSymb = '{sym}'
                  AND XpryDt = '{expiry.strftime("%Y-%m-%d")}'
                  AND TradDt = '{date_str}'
                  AND StrkPric = {pos["Strike"]}
                """
                df_hold = self.con.execute(h_query).df()

                exit_signal = False
                exit_reason = ""

                if df_hold.empty:
                    # No data? hold.
                    active_pos_next.append(pos)
                    continue

                ce_h = df_hold[df_hold["OptnTp"] == "CE"]
                pe_h = df_hold[df_hold["OptnTp"] == "PE"]

                if ce_h.empty or pe_h.empty:
                    active_pos_next.append(pos)
                    continue

                # Prices
                ce_high = ce_h["HghPric"].iloc[0]
                ce_low = ce_h["LwPric"].iloc[0]
                pe_high = pe_h["HghPric"].iloc[0]
                pe_low = pe_h["LwPric"].iloc[0]

                ce_close = ce_h["ClsPric"].iloc[0]
                pe_close = pe_h["ClsPric"].iloc[0]

                is_expiry = curr_date == expiry
                buyback_cost = 0.0

                if pos.get("Is_Crisis", False):
                    pos["Days_Held"] += 1
                    curr_val = ce_close + pe_close

                    sym_metrics = self.rv_series_map.get(pos["Symbol"], {})
                    p_rv5 = sym_metrics.get("RV5", pd.Series()).get(curr_date, np.nan)
                    p_rv20 = sym_metrics.get("RV20", pd.Series()).get(curr_date, np.nan)
                    vol_norm = False
                    if pd.notna(p_rv5) and pd.notna(p_rv20) and p_rv20 > 0:
                        if (p_rv5 / p_rv20) < 1.1:
                            vol_norm = True

                    if curr_val >= pos["Premium"] * 1.8:
                        exit_signal = True
                        exit_reason = "CRISIS EXIT - TARGET"
                        buyback_cost = pos["Premium"] * 1.8
                    elif curr_val <= pos["Premium"] * 0.55:
                        exit_signal = True
                        exit_reason = "CRISIS EXIT - STOP"
                        buyback_cost = pos["Premium"] * 0.55
                    elif vol_norm:
                        exit_signal = True
                        exit_reason = "CRISIS EXIT - NORMALIZED"
                        buyback_cost = curr_val
                    elif pos["Days_Held"] >= 10:
                        exit_signal = True
                        exit_reason = "CRISIS EXIT - MAX HOLD"
                        buyback_cost = curr_val
                    elif is_expiry:
                        exit_signal = True
                        exit_reason = "CRISIS EXIT - EXPIRY"
                        buyback_cost = curr_val

                    if exit_signal:
                        # Long PnL computation: (Exit Price - Entry Price) * Size
                        points_pnl = buyback_cost - pos["Premium"]
                        gross_pnl = points_pnl * pos["Size"]

                        turnover = (pos["Premium"] + buyback_cost) * pos["Size"]
                        slippage = turnover * SLIPPAGE_PCT
                        comm = TRANSACTION_COST * 2
                        net_pnl = gross_pnl - slippage - comm

                        self.equity += net_pnl
                        pos["Exit_Date"] = date_str
                        pos["Buyback_Cost"] = buyback_cost
                        pos["PnL_Net"] = net_pnl
                        pos["Exit_Reason"] = exit_reason
                        closed_trades.append(pos)
                        if verbose:
                            print(
                                f"[{exit_reason}] {pos['Symbol']} | PnL: {net_pnl:.2f}"
                            )

                else:
                    # 1. Stop Loss (Intraday Worst)
                    worst_val = max(ce_high + pe_low, pe_high + ce_low)
                    stop_val = pos["Premium"] * 2.0

                    # 2. Profit Take (Intraday Best)
                    best_val = ce_low + pe_low
                    target_val = pos["Premium"] * 0.30

                    if worst_val >= stop_val:
                        exit_signal = True
                        exit_reason = "Stop Loss (2x)"
                        buyback_cost = stop_val
                    elif best_val <= target_val:
                        exit_signal = True
                        exit_reason = "Profit Take (70%)"
                        buyback_cost = target_val
                    elif is_expiry:
                        exit_signal = True
                        exit_reason = "Expiry Close"
                        buyback_cost = ce_close + pe_close

                    if exit_signal:
                        # Calculate PnL
                        # For short trades: PnL = (Entry - Exit) * Size
                        points_pnl = pos["Premium"] - buyback_cost
                        gross_pnl = points_pnl * pos["Size"]

                        # Costs
                        turnover = (pos["Premium"] + buyback_cost) * pos["Size"]
                        slippage = turnover * SLIPPAGE_PCT
                        comm = TRANSACTION_COST * 2
                        net_pnl = gross_pnl - slippage - comm

                        # Update Equity
                        self.equity += net_pnl

                        # Log
                        pos["Exit_Date"] = date_str
                        pos["Buyback_Cost"] = buyback_cost
                        pos["PnL_Net"] = net_pnl
                        pos["Exit_Reason"] = exit_reason
                        closed_trades.append(pos)

                if not exit_signal:
                    active_pos_next.append(pos)

            open_positions = active_pos_next

            # --- B. Check Entries ---
            # KILL SWITCH CHECK: Monitor intraday drawdown
            if self.equity < day_start_equity:
                intraday_dd = (self.equity - day_start_equity) / day_start_equity
                if intraday_dd <= -0.02 and not kill_switch_active:
                    kill_switch_active = True
                    kill_switch_activations += 1
                    if verbose:
                        print("[KILL SWITCH TRIGGERED]")
                        print(f"    Intraday DD: {intraday_dd:.2%}")
                        print("    New trades blocked until next session.")

            # Update Correlation Shock Status (Regime Shift)
            c10 = self.corr_10.get(curr_date, 0.0)
            c60 = self.corr_60.get(curr_date, 0.0)

            shock_active = False

            # 1. Regime Shift Trigger
            # ShortTerm >= Baseline + 0.12
            regime_cond = c10 >= (c60 + 0.12)

            # 2. Hard Crisis Trigger
            # ShortTerm >= 0.97
            crisis_cond = c10 >= 0.97

            if regime_cond or crisis_cond:
                shock_active = True
                shock_days += 1

            # 1. Update Market Regime (NIFTY Proxy)
            nifty_data = self.rv_series_map.get("NIFTY", {})
            nifty_rv = nifty_data.get("RV", pd.Series()).get(curr_date, 0.0)
            n_rv5 = nifty_data.get("RV5", pd.Series()).get(curr_date, np.nan)
            n_rv20 = nifty_data.get("RV20", pd.Series()).get(curr_date, np.nan)
            n_move3 = nifty_data.get("Move_3", pd.Series()).get(curr_date, np.nan)

            vol_accel = (
                n_rv5 / n_rv20
                if (pd.notna(n_rv5) and pd.notna(n_rv20) and n_rv20 > 0)
                else 0.0
            )

            is_crisis = False
            if (vol_accel >= 1.40 and c10 >= 0.95) or (
                pd.notna(n_move3) and n_move3 >= 0.04
            ):
                is_crisis = True

            if is_crisis:
                start_regime = "CRISIS"
            elif nifty_rv > rv_threshold:
                start_regime = "HIGH_RISK"
            else:
                start_regime = "NORMAL"

            # Log Shift
            if start_regime != current_regime:
                if verbose:
                    if start_regime == "CRISIS":
                        print(f"[CRISIS REGIME ACTIVATED]")
                    elif current_regime == "CRISIS" and start_regime != "CRISIS":
                        print(
                            f"[REGIME SHIFT] CRISIS -> {start_regime} (RV: {nifty_rv:.1%})"
                        )
                    else:
                        print(
                            f"[REGIME SHIFT] {current_regime} -> {start_regime} (RV: {nifty_rv:.1%})"
                        )
                current_regime = start_regime

            # Calculate Current Margin Usage
            current_margin_used = sum(p["Margin_Locked"] for p in open_positions)

            for sym in self.active_symbols:
                # KILL SWITCH: Block all new entries if triggered
                if kill_switch_active:
                    continue

                # Crisis Logic Branch
                if current_regime == "CRISIS":
                    # Check if already hold crisis position for this symbol
                    has_crisis = any(
                        p["Symbol"] == sym and p.get("Is_Crisis", False)
                        for p in open_positions
                    )
                    if has_crisis:
                        continue

                    # Find nearest future expiry
                    exps_all = sorted(list(set(symbol_schedules[sym].values())))
                    future_exps = [e for e in exps_all if e >= curr_date]
                    if not future_exps:
                        continue
                    expiry = future_exps[0]

                    entry_query = f"""
                    SELECT 
                        TckrSymb, TradDt, OptnTp, StrkPric, 
                        OpnPric, HghPric, LwPric, ClsPric, 
                        UndrlygPric, OpnIntrst, TtlTradgVol 
                    FROM fo_data 
                    WHERE TckrSymb = '{sym}' 
                      AND XpryDt = '{expiry.strftime("%Y-%m-%d")}' 
                      AND TradDt = '{date_str}'
                    """
                    df_entry = self.con.execute(entry_query).df()
                    if df_entry.empty:
                        continue

                    spot_entry = df_entry["UndrlygPric"].iloc[0]
                    if pd.isna(spot_entry) or spot_entry <= 0:
                        continue

                    options = df_entry[df_entry["OptnTp"].isin(["CE", "PE"])].copy()
                    if options.empty:
                        continue
                    strikes = options["StrkPric"].unique()
                    candidates = []

                    for stk in strikes:
                        ce = options[
                            (options["StrkPric"] == stk) & (options["OptnTp"] == "CE")
                        ]
                        pe = options[
                            (options["StrkPric"] == stk) & (options["OptnTp"] == "PE")
                        ]
                        if ce.empty or pe.empty:
                            continue

                        ce_entry = ce["ClsPric"].iloc[0]
                        pe_entry = pe["ClsPric"].iloc[0]
                        if (
                            ce["OpnIntrst"].iloc[0] < 1000
                            or pe["OpnIntrst"].iloc[0] < 1000
                        ):
                            continue
                        if (
                            ce["TtlTradgVol"].iloc[0] < 2000
                            or pe["TtlTradgVol"].iloc[0] < 2000
                        ):
                            continue

                        diff = abs(stk - spot_entry)
                        candidates.append(
                            {
                                "Strike": stk,
                                "SpotDiff": diff,
                                "CE": ce_entry,
                                "PE": pe_entry,
                                "Prem": ce_entry + pe_entry,
                            }
                        )

                    if not candidates:
                        continue
                    best = min(candidates, key=lambda x: x["SpotDiff"])

                    # RULE 4: Minimum Premium Safety
                    if best["Prem"] < 0.005 * spot_entry:
                        continue

                    risk_pct = 0.02
                    risk_budget = self.equity * risk_pct
                    risk_per_lot = best["Prem"] * LOT_SIZE
                    candidate_lots = (
                        int(np.floor(risk_budget / risk_per_lot))
                        if risk_per_lot > 0
                        else 0
                    )
                    if candidate_lots < 1 and self.equity >= risk_per_lot:
                        candidate_lots = 1

                    original_lots = candidate_lots

                    # RULE 1: Max 5 lots per symbol
                    candidate_lots = min(candidate_lots, 5)

                    # RULE 2: Single trade capital limit
                    trade_premium = best["Prem"] * LOT_SIZE
                    max_lots_by_trade_limit = (
                        int((0.03 * self.equity) / trade_premium)
                        if trade_premium > 0
                        else 0
                    )
                    candidate_lots = min(candidate_lots, max_lots_by_trade_limit)

                    # RULE 3: Total Long Vol Exposure Cap
                    crisis_premium_spent = sum(
                        p["Premium"] * p["Size"]
                        for p in open_positions
                        if p.get("Is_Crisis", False)
                    )
                    new_trade_premium = candidate_lots * trade_premium
                    if crisis_premium_spent + new_trade_premium > 0.10 * self.equity:
                        continue

                    # RULE 5: Integer Safety
                    if candidate_lots < 1:
                        continue

                    if candidate_lots < original_lots:
                        print("[LOT SAFETY REDUCTION APPLIED]")

                    lots = candidate_lots

                    margin_locked = best["Prem"] * LOT_SIZE * lots
                    pos_record = {
                        "Symbol": sym,
                        "Entry_Date": date_str,
                        "Expiry": expiry,
                        "Strike": best["Strike"],
                        "Premium": best["Prem"],
                        "Size": lots * LOT_SIZE,
                        "Appx_Spot": spot_entry,
                        "Margin_Locked": margin_locked,
                        "Margin_Pct": 0.0,
                        "Is_Crisis": True,
                        "Days_Held": 0,
                    }
                    open_positions.append(pos_record)
                    crisis_trades += 1
                    if verbose:
                        print(
                            f"[CRISIS TRADE OPENED] {sym} | Strike: {best['Strike']} | Lots: {lots}"
                        )

                    continue  # Skip normal short vol logic for this symbol in Crisis Regime

                # Check for Normal Entry Trigger
                if curr_date not in symbol_schedules[sym]:
                    continue

                # Check Global Filter
                if current_regime == "HIGH_RISK":
                    if verbose:
                        print(
                            f"[GLOBAL FILTER] High volatility regime detected — blocking new positions ({sym})."
                        )
                    blocked_trades += 1
                    continue

                # Check Volatility Acceleration Filter
                # Compute vol_acceleration = RV_5 / RV_20
                sym_metrics = self.rv_series_map.get(sym, {})
                rv5 = sym_metrics.get("RV5", pd.Series()).get(curr_date, np.nan)
                rv20 = sym_metrics.get("RV20", pd.Series()).get(curr_date, np.nan)

                if not pd.isna(rv5) and not pd.isna(rv20) and rv20 > 0:
                    vol_acceleration = rv5 / rv20
                    if vol_acceleration > 1.35:
                        if verbose:
                            print(
                                f"[ACCELERATION FILTER] Spiking Vol ({vol_acceleration:.2f}) — blocking {sym}"
                            )
                        blocked_acceleration += 1
                        continue

                # Check VoV Risk Regime
                # Retrieve metrics
                sym_data = self.rv_series_map.get(sym, {})
                vc5 = sym_data.get("VolChange5d", pd.Series()).get(curr_date, 0.0)
                vov_val = sym_data.get("VoV", pd.Series()).get(curr_date, np.nan)
                vov_90 = sym_data.get("VoV_90", pd.Series()).get(curr_date, np.nan)
                vov_count = sym_data.get("VoV_Count", pd.Series()).get(curr_date, 0)

                vov_active = False
                vov_enabled = False

                # VoV WARMUP CHECK: Require minimum 20 sessions
                VoV_WARMUP_PERIOD = 20

                if vov_count >= VoV_WARMUP_PERIOD:
                    vov_enabled = True

                    # Check Safety Guardrails (only if enabled)
                    if vov_count < 200:
                        pass
                    else:
                        # Check Triggers
                        spike_cond = vc5 >= 0.35

                        pct_cond = False
                        if not pd.isna(vov_90) and not pd.isna(vov_val):
                            if vov_val >= vov_90:
                                pct_cond = True

                        if spike_cond or pct_cond:
                            vov_active = True
                            vov_triggers += 1
                else:
                    # VoV not enabled - log once per session
                    if verbose and not vov_warmup_logged:
                        print("[VoV Disabled \u2014 Warmup]")
                        print(
                            f"    History: {int(vov_count) if pd.notna(vov_count) else 0} / {VoV_WARMUP_PERIOD} sessions"
                        )
                        vov_warmup_logged = True

                # --- POST-SHOCK TRIGGER DETECTION ---
                # Trigger if:
                # 1. Acceleration filter would block
                # 2. VoV crisis active
                # 3. Correlation shock > 0.95

                post_shock_trigger = False

                # Check acceleration (already computed above)
                if not pd.isna(rv5) and not pd.isna(rv20) and rv20 > 0:
                    vol_acceleration = rv5 / rv20
                    if vol_acceleration > 1.35:
                        post_shock_trigger = True

                # Check VoV
                if vov_active and vov_enabled:
                    post_shock_trigger = True

                # Check extreme correlation
                if c10 > 0.95:
                    post_shock_trigger = True

                # Activate cooldown if triggered
                if post_shock_trigger and not cooldown_active:
                    cooldown_active = True
                    cooldown_days_remaining = 10
                    cooldown_activations += 1
                    if verbose:
                        print(f"[POST-SHOCK MODE ACTIVATED]")
                        print(f"    Cooldown Days: 10")

                expiry = symbol_schedules[sym][curr_date]

                # Attempt Entry Logic (Phase 1)
                entry_query = f"""
                SELECT 
                    TckrSymb, TradDt, OptnTp, StrkPric, 
                    OpnPric, HghPric, LwPric, ClsPric, 
                    UndrlygPric, OpnIntrst, TtlTradgVol 
                FROM fo_data 
                WHERE TckrSymb = '{sym}' 
                  AND XpryDt = '{expiry.strftime("%Y-%m-%d")}' 
                  AND TradDt = '{date_str}'
                """
                df_entry = self.con.execute(entry_query).df()

                if df_entry.empty:
                    continue

                try:
                    spot_entry = df_entry["UndrlygPric"].iloc[0]
                    if pd.isna(spot_entry) or spot_entry == 0:
                        continue
                except Exception as e:
                    if verbose:
                        print(f"[{sym}] Skip Logic Error: {e}")
                    continue

                # Panic Filter (T-5 Check)
                # We need T-5 from current date for this symbol
                # Optimization: Run simple query
                t5_query = f"""
                SELECT UndrlygPric FROM fo_data 
                WHERE TckrSymb = '{sym}' AND TradDt < '{date_str}'
                ORDER BY TradDt DESC LIMIT 1 OFFSET 4
                """
                df_t5 = self.con.execute(t5_query).df()
                if not df_t5.empty:
                    spot_t5 = df_t5["UndrlygPric"].iloc[0]
                    if spot_t5 > 0:
                        ret_5d = (spot_entry / spot_t5) - 1
                        if abs(ret_5d) > 0.03:
                            # Skipped Panic
                            continue

                # Candidate Hunt
                options = df_entry[df_entry["OptnTp"].isin(["CE", "PE"])].copy()
                if options.empty:
                    continue

                strikes = options["StrkPric"].unique()
                candidates = []

                for stk in strikes:
                    ce = options[
                        (options["StrkPric"] == stk) & (options["OptnTp"] == "CE")
                    ]
                    pe = options[
                        (options["StrkPric"] == stk) & (options["OptnTp"] == "PE")
                    ]
                    if ce.empty or pe.empty:
                        continue

                    ce_entry = ce["ClsPric"].iloc[0]
                    pe_entry = pe["ClsPric"].iloc[0]

                    # Filters
                    if ce["OpnIntrst"].iloc[0] < 500 or pe["OpnIntrst"].iloc[0] < 500:
                        continue
                    if (
                        ce["TtlTradgVol"].iloc[0] < 1000
                        or pe["TtlTradgVol"].iloc[0] < 1000
                    ):
                        continue
                    if (ce_entry + pe_entry) < 2.0:
                        continue

                    diff = abs(ce_entry - pe_entry)
                    candidates.append(
                        {
                            "Strike": stk,
                            "Diff": diff,
                            "CE": ce_entry,
                            "PE": pe_entry,
                            "Prem": ce_entry + pe_entry,
                        }
                    )

                if not candidates:
                    continue

                best = min(candidates, key=lambda x: x["Diff"])

                # Yield Filter
                yield_pct = (best["Prem"] / spot_entry) * 100
                if yield_pct < 0.8:
                    continue

                # VRP Filter
                rv_series = self.rv_series_map.get(sym, {}).get("RV", pd.Series())
                rv = rv_series.get(curr_date, np.nan)
                if pd.isna(rv):
                    # Lookback logic difficult here without keeping history?
                    # Approximation: Assume if missing, skip or use prev.
                    continue

                # Brenner-Subrahmanyam ATM IV Approximation
                days_to_expiry = (expiry.date() - curr_date.date()).days
                if days_to_expiry <= 0:
                    continue

                t_years = days_to_expiry / 365.0
                iv_valid = best["Prem"] / (0.4 * spot_entry * np.sqrt(t_years))

                if rv <= 0 or iv_valid <= (rv * 1.15):
                    continue

                # Print validation logic (do not spam verbose setting)
                print("[IV VALIDATED]")
                print(f"IV: {iv_valid:.2%}")
                print(f"RV: {rv:.2%}")
                print(f"Ratio: {(iv_valid / rv):.2f}")

                # --- SIZING & MARGIN CHECK ---

                # 1. Calculate Required Margin
                margin_factor = get_margin_pct(rv)
                margin_req_per_lot = spot_entry * LOT_SIZE * margin_factor

                # 2. Position Limits & Sizing

                # Dynamic Position Scaling (Correlation Risk & VoV Risk)
                # Base Risk
                risk_pct = 0.05

                # Interaction Logic
                scaled_msg = ""

                if shock_active and vov_active:
                    risk_pct = 0.02
                    scaled_msg = "5% -> 2% (Dual Shock)"
                elif vov_active:
                    risk_pct = 0.025
                    scaled_msg = "5% -> 2.5% (VoV Crisis)"
                elif shock_active:
                    risk_pct = 0.035
                    scaled_msg = "5% -> 3.5% (Correlation Shock)"

                # POST-SHOCK COOLDOWN OVERRIDE
                # Apply 50% scaling if in cooldown
                # CRITICAL: Only apply VoV scaling if VoV is enabled
                if cooldown_active:
                    risk_pct = risk_pct * 0.5
                    if scaled_msg:
                        scaled_msg = f"{scaled_msg} + Post-Shock 50%"
                    else:
                        scaled_msg = "5% -> 2.5% (Post-Shock Cooldown)"
                elif vov_active and not vov_enabled:
                    # VoV triggered but not enabled - do NOT scale
                    # Reset to base or correlation-only scaling
                    if shock_active:
                        risk_pct = 0.035
                        scaled_msg = "5% -> 3.5% (Correlation Shock)"
                    else:
                        risk_pct = 0.05
                        scaled_msg = ""

                if verbose:
                    # Post-Shock Mode Status
                    if cooldown_active:
                        print(f"[POST-SHOCK MODE ACTIVE]")
                        print(f"    Cooldown Days Remaining: {cooldown_days_remaining}")

                    if vov_active:
                        if vov_enabled:
                            print("[VoV REGIME ACTIVE]")
                            print(f"    IV 5d Change:   {vc5:.1%}")
                            if not pd.isna(vov_90):
                                print(f"    VoV Value:      {vov_val:.4f}")
                                print(f"    90th Threshold: {vov_90:.4f}")
                                print(
                                    f"    History Length: {int(vov_count) if pd.notna(vov_count) else 0}"
                                )
                            else:
                                print(
                                    f"    VoV Check:      Skipped (Insufficient History: {int(vov_count) if pd.notna(vov_count) else 0})"
                                )

                            print(f"    Scaled: {scaled_msg}")
                    elif (
                        shock_active
                    ):  # Only print correlation if not already covered by dual shock
                        print(f"[Correlation Regime Shift]")
                        print(f"    Baseline Corr:   {c60:.2f}")
                        print(f"    Short-term Corr: {c10:.2f}")
                        print(f"    Scaled: {scaled_msg}")

                risk_budget = self.equity * risk_pct
                risk_per_lot = 2 * best["Prem"] * LOT_SIZE
                lots_risk = risk_budget // risk_per_lot if risk_per_lot > 0 else 0

                # Check Portfolio Margin Cap (65%)
                # Proposed Margin = Current + New
                # We need to know Lots first to compute New Margin.
                # But lots depends on Margin Budget?
                # "Recalculate projected portfolio margin ... NEVER allow > 65%"
                # So we calculate candidate lots based on RISK and generic margin budget first?
                # Then check the 65% hard cap.

                # Generic Margin Sizing (to ensure we have enough free cash to open)
                # We can stick to 60% sizing budget or use 65%?
                # If we use 65% for sizing, we might validly size to 64% and pass hard cap.
                # If we keep 60% sizing, we never hit 65% hard cap (unless existing positions expand? But here we use entry margin).
                # User said: "Recalculate projected ... Hard rule ... NEVER allow > 65%".
                # This implies checking the Resulting State.
                # Let's use 65% as the budget for sizing too, to allow the scaling logic to work up to the hard cap.

                margin_budget_total = self.equity * 0.65

                # Available for NEW trades
                margin_available = margin_budget_total - current_margin_used
                if margin_available <= 0:
                    # Already over cap
                    continue

                lots_margin = (
                    margin_available // margin_req_per_lot
                    if margin_req_per_lot > 0
                    else 0
                )

                candidate_lots = int(min(lots_risk, lots_margin))

                # RULE 4: Minimum Premium Safety
                if best["Prem"] < 0.005 * spot_entry:
                    continue

                original_lots = candidate_lots

                # RULE 1: Max 5 lots per symbol
                candidate_lots = min(candidate_lots, 5)

                # RULE 2: Single trade capital limit
                trade_premium = best["Prem"] * LOT_SIZE
                max_lots_by_trade_limit = (
                    int((0.03 * self.equity) / trade_premium)
                    if trade_premium > 0
                    else 0
                )
                candidate_lots = min(candidate_lots, max_lots_by_trade_limit)

                # RULE 5: Integer Safety
                if candidate_lots < 1:
                    continue

                if candidate_lots < original_lots:
                    print("[LOT SAFETY REDUCTION APPLIED]")

                # Final Hard Cap Check (Redundant if we sized with 65% budget, but specific blocking required)
                # "If exceeded: Skip... Print [Blocked: Portfolio Margin Cap]"
                # Sizing with 65% budget ensures we don't exceed 65%.
                # But maybe `lots_risk` is the constraint, and we need to check if that also fits margin?
                # min(lots_risk, lots_margin) guarantees it fits margin_available.
                # So we implicitly satisfy < 65%.
                # BUT, if we were rejected because `margin_available <= 0`, we should log "Blocked".
                # Wait, "If exceeded: Skip the trade. Print [Blocked...]"
                # The sizing logic `lots_margin` ensures we don't select a size that exceeds.
                # But if `margin_available` was effectively 0 (or close) such that lots < 1?
                # Let's implement the explicit check logic for the log.

                projected_margin_usage = current_margin_used + (
                    candidate_lots * margin_req_per_lot
                )
                projected_margin_pct = projected_margin_usage / self.equity

                if projected_margin_pct > 0.65:
                    # Should rarely happen with lots_margin logic, but floating point?
                    if verbose:
                        print("[Risk Block]")
                        print(
                            f"    Reason: Portfolio Margin > 65% ({projected_margin_pct:.1%})"
                        )
                    blocked_margin += 1
                    continue

                # If we are here, we are good.
                lots = candidate_lots

                # EXECUTE
                if shock_active or vov_active:
                    trades_scaled += 1

                margin_locked = lots * margin_req_per_lot
                current_margin_used += (
                    margin_locked  # Update for next symbol in same loop
                )

                pos_record = {
                    "Symbol": sym,
                    "Entry_Date": date_str,
                    "Expiry": expiry,
                    "Strike": best["Strike"],
                    "Premium": best["Prem"],
                    "Size": lots * LOT_SIZE,
                    "Appx_Spot": spot_entry,
                    "Margin_Locked": margin_locked,
                    "Margin_Pct": margin_factor,
                }
                open_positions.append(pos_record)

                if verbose:
                    print(f"[{sym}] EXECUTED STRADDLE")
                    print(f"    Date: {date_str} | Strike: {best['Strike']}")
                    print(
                        f"    Portfolio Margin Used: {current_margin_used / self.equity:.1%}"
                    )
                    print(f"    Margin Used:   {margin_factor:.0%}")

            # End of Day
            # Track Equity
            equity_curve.append({"Date": date_str, "Equity": self.equity})
            self.portfolio_returns.append(0.0)  # Placeholder, calc later or diff

        elapsed = time.perf_counter() - t0_start
        if verbose:
            print(f"[Backtest] Simulation Complete in {elapsed:.2f}s")
            print(f"[Safety] Trades Blocked By Regime Filter: {blocked_trades}")
            print(
                f"[Safety] Trades Blocked By Acceleration Filter: {blocked_acceleration}"
            )
            print(f"[Risk] Correlation Shock Days: {shock_days}")
            print(f"[Risk] Trades Scaled: {trades_scaled}")
            print(f"[Risk] Trades Blocked (Margin): {blocked_margin}")
            print(f"[Risk] VoV Trigger Count: {vov_triggers}")
            print(f"[Risk] Post-Shock Cooldown Activations: {cooldown_activations}")
            print(f"[Safety] Kill Switch Activations: {kill_switch_activations}")

        return (
            pd.DataFrame(closed_trades),
            pd.DataFrame(equity_curve),
            {
                "blocked_regime": blocked_trades,
                "blocked_accel": blocked_acceleration,
                "blocked_margin": blocked_margin,
                "trades_scaled": trades_scaled,
                "shock_days": shock_days,
                "vov_triggers": vov_triggers,
                "cooldown_activations": cooldown_activations,
                "kill_switch_activations": kill_switch_activations,
            },
        )

    def analyze_portfolio(self, trades, equity_df):
        if trades.empty:
            print("No trades.")
            return

        print("\n=== INSTITUTIONAL PORTFOLIO DASHBOARD ===")
        print(f"Ending Equity: {self.equity:,.2f} INR")

        # Calculates Returns based on Equity Curve
        equity_df["Prev"] = equity_df["Equity"].shift(1).fillna(CAPITAL)
        equity_df["Ret"] = equity_df["Equity"] / equity_df["Prev"] - 1

        # Portfolio Metrics
        mean_ret = equity_df["Ret"].mean()
        std_ret = equity_df["Ret"].std()
        ann_factor = np.sqrt(252)  # Daily returns

        sharpe = (mean_ret / std_ret) * ann_factor if std_ret > 0 else 0

        dd = equity_df["Equity"] / equity_df["Equity"].cummax() - 1
        max_dd = dd.min()

        downside = equity_df[equity_df["Ret"] < 0]["Ret"].std()
        sortino = (mean_ret / downside) * ann_factor if downside > 0 else 0

        print(f"{'Metric':<20} | {'Value':<15}")
        print("-" * 38)
        print(f"{'Sharpe Ratio':<20} | {sharpe:.2f}")
        print(f"{'Sortino Ratio':<20} | {sortino:.2f}")
        print(f"{'Max Drawdown':<20} | {max_dd:.2%}")
        print(f"{'Total Trades':<20} | {len(trades)}")

        # Asset Diversification
        print("\n[Asset Breakdown]")
        breakdown = trades.groupby("Symbol")["PnL_Net"].sum()
        for sym, pnl in breakdown.items():
            print(f"  {sym:<10}: {pnl:,.2f} INR")

        print("\n[Equity Curve]")
        if len(equity_df) > 0:
            min_e = equity_df["Equity"].min()
            max_e = equity_df["Equity"].max()
            rnge = max_e - min_e if max_e != min_e else 1
            step = max(1, len(equity_df) // 20)
            for _, row in equity_df.iloc[::step].iterrows():
                pos = int(((row["Equity"] - min_e) / rnge) * 50)
                print(f"{row['Date']}: {'#' * pos} ({row['Equity']:,.0f})")


def run_portfolio_backtest():
    # Silent run, manual output
    bt = PortfolioVolEngine()

    # We want standard 0.35 regime threshold, verbose=True to see log but NOT dashboard?
    # User said "Output ONLY..."
    # So run silent verbose=False to suppress trade logs?
    # "Log once at the end... No per-trade spam." -> verbose=False for sensitivity check?
    # But for THIS step user asked for specific logs "When scaling activates, print..."
    # So we need verbose=True (or custom verbose logic).
    # But user also said "No per-trade spam" in PREVIOUS task.
    # In Step 5 of THIS task: "When scaling activates, print... [Correlation Shock Active]..."
    # So we should run verbose=True? The previous "No per-trade spam" was for acceleration filter.
    # Let's run verbose=True to show the logs requested.
    trades, equity, stats = bt.simulate_portfolio(rv_threshold=0.35, verbose=True)

    # Compute Metrics
    end_eq = equity["Equity"].iloc[-1]

    equity["Prev"] = equity["Equity"].shift(1).fillna(CAPITAL)
    equity["Ret"] = equity["Equity"] / equity["Prev"] - 1
    mean_ret = equity["Ret"].mean()
    std_ret = equity["Ret"].std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

    downside = equity["Ret"][equity["Ret"] < 0].std()
    # Correct calculation: downside deviation of returns
    # Previous code: equity[equity["Ret"] < 0]["Ret"].std()
    # That works if "Ret" is column.

    sortino = (mean_ret / downside) * np.sqrt(252) if downside > 0 else 0

    dd = equity["Equity"] / equity["Equity"].cummax() - 1
    max_dd = dd.min()

    print("\n=== FINAL METRICS ===")
    print(f"Ending Equity:           {end_eq:,.2f}")

    total_return = (end_eq / CAPITAL) - 1.0

    crisis_trades = (
        trades[trades.get("Is_Crisis", False) == True]
        if "Is_Crisis" in trades.columns
        else pd.DataFrame()
    )
    crisis_count = len(crisis_trades)
    crisis_wins = (
        len(crisis_trades[crisis_trades["PnL_Net"] > 0]) if crisis_count > 0 else 0
    )
    crisis_win_rate = crisis_wins / crisis_count if crisis_count > 0 else 0.0

    print(f"Total Return:            {total_return:.2%}")
    print(f"Sharpe:                  {sharpe:.2f}")
    print(f"Max Drawdown:            {max_dd:.2%}")
    print(f"Sortino:                 {sortino:.2f}")
    print(f"Trades Taken:            {len(trades)}")
    print(f"Crisis Trades Count:     {crisis_count}")
    print(f"Crisis Win Rate:         {crisis_win_rate:.2%}")
    print(f"Trades Scaled:           {stats['trades_scaled']}")
    print(f"Trades Blocked (Margin): {stats['blocked_margin']}")
    print(f"VoV Trigger Count:       {stats['vov_triggers']}")
    print(f"Cooldown Activations:    {stats['cooldown_activations']}")
    print(f"Kill Switch Activations: {stats['kill_switch_activations']}")


if __name__ == "__main__":
    try:
        run_portfolio_backtest()
    except Exception:
        import traceback

        traceback.print_exc()
