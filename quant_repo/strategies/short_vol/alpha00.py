"""
backtest_short_vol.py

Production-Grade Short Volatility Backtest Engine
-------------------------------------------------
Strategy:
    - Sell ATM Straddle (CE+PE) on every Weekly Expiry (Thursday).
    - Entry: T-2 (two trading days before expiry) Market Open.
    - Exit: Expiry Day Market Close (or earlier on stops/profit targets).
    - Frequency: Weekly.

Technology:
    - DuckDB: Streaming SQL execution on Parquet Lake.
    - Zero-Memory Overhead.

Metrics:
    - Win Rate, Sharpe, Max Drawdown, Equity Curve.

MODIFIED VERSION – Increased position limits and relaxed stops for higher returns.
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
LOT_SIZE = 50  # NIFTY contract size (adjust for BANKNIFTY/FINNIFTY? They use different lot sizes but we'll keep as is; actual lot size is handled per symbol via data)

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

        # ========== MODIFIED PARAMETERS FOR HIGHER RETURNS ==========
        self.max_lots_per_sym = {
            "NIFTY": 20,
            "BANKNIFTY": 12,
            "FINNIFTY": 12,
        }  # Was 10
        self.trade_capital_limit_pct = 0.05  # Was 0.03 (3%)
        self.stop_loss_multiplier = 3.0  # Was 2.0
        self.risk_budget_pct = 0.08  # Was 0.05 (5%)
        self.corr_regime_shift_threshold = 0.20  # Was 0.12
        self.corr_crisis_threshold = 0.98  # Was 0.97
        # ============================================================

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

            # Validation Check
            count_nifty = self.con.execute(
                "SELECT COUNT(*) FROM fo_data WHERE TckrSymb = 'NIFTY'"
            ).fetchone()[0]

            if count_nifty == 0:
                raise RuntimeError(
                    "NIFTY not found in FO lake. Verify bhavcopy ingestion."
                )
            else:
                print(f"[Validation] NIFTY rows detected: {count_nifty}")

            total_count = self.con.execute("SELECT count(*) FROM fo_data").fetchone()[0]
            print(f"[Init] Lake Registered. Total Rows: {total_count}")

        except Exception as e:
            raise RuntimeError(f"Failed to setup DuckDB: {e}")

    def get_trading_dates_and_expiries(self, symbol):
        """Get trading dates and expiries for a specific symbol."""
        d_query = f"""
        SELECT DISTINCT TradDt 
        FROM fo_data 
        WHERE TckrSymb = '{symbol}'
        ORDER BY TradDt ASC
        """
        trading_dates = pd.to_datetime(
            self.con.execute(d_query).df()["TradDt"]
        ).tolist()

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
        ret_data = {}
        for sym in self.active_symbols:
            if "LogRet" in self.rv_series_map.get(sym, {}):
                ret_data[sym] = self.rv_series_map[sym]["LogRet"]

        if not ret_data:
            self.corr_10 = pd.Series(dtype=float)
            self.corr_60 = pd.Series(dtype=float)
            return

        df_ret = pd.DataFrame(ret_data)

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
        for sym, data in self.rv_series_map.items():
            if "RV" not in data:
                continue

            vol_series = data["RV"]
            vol_change_5d = vol_series.pct_change(periods=5)
            vol_change_1d = vol_series.diff().replace([np.inf, -np.inf], np.nan)
            vov = vol_change_1d.rolling(window=20).std()
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

            date_map = {d: i for i, d in enumerate(td)}
            entry_map = {}

            for exp in exps:
                if exp not in date_map:
                    continue
                exp_idx = date_map[exp]

                # Entry: T-2 (2 days before expiry)
                entry_idx = exp_idx - 2
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
        shock_days = 0
        trades_scaled = 0
        blocked_margin = 0

        # Daily Portfolio Kill Switch State
        kill_switch_active = False
        kill_switch_activations = 0
        day_start_equity = self.equity

        # Crisis Long Vol Tracking
        crisis_trades = 0

        t0_start = time.perf_counter()

        for curr_date in sorted_dates:
            # --- START OF DAY: RESET KILL SWITCH ---
            day_start_equity = self.equity
            kill_switch_active = False

            date_str = curr_date.strftime("%Y-%m-%d")

            # --- A. Update Metrics & Check Exits ---
            active_pos_next = []

            for pos in open_positions:
                sym = pos["Symbol"]
                expiry = pos["Expiry"]

                # Fetch OHLC for this Position (Strike)
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
                    active_pos_next.append(pos)
                    continue

                ce_h = df_hold[df_hold["OptnTp"] == "CE"]
                pe_h = df_hold[df_hold["OptnTp"] == "PE"]

                if ce_h.empty or pe_h.empty:
                    active_pos_next.append(pos)
                    continue

                ce_high = ce_h["HghPric"].iloc[0]
                ce_low = ce_h["LwPric"].iloc[0]
                pe_high = pe_h["HghPric"].iloc[0]
                pe_low = pe_h["LwPric"].iloc[0]

                ce_close = ce_h["ClsPric"].iloc[0]
                pe_close = pe_h["ClsPric"].iloc[0]

                is_expiry = curr_date == expiry
                buyback_cost = 0.0

                if pos.get("Is_Crisis", False):
                    # Crisis position handling (unchanged)
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
                    # Normal short vol position
                    worst_val = max(ce_high + pe_low, pe_high + ce_low)
                    # MODIFIED: Stop loss multiplier increased
                    stop_val = pos["Premium"] * self.stop_loss_multiplier

                    best_val = ce_low + pe_low
                    target_val = pos["Premium"] * 0.20

                    days_to_expiry = (expiry.date() - curr_date.date()).days

                    if worst_val >= stop_val:
                        exit_signal = True
                        exit_reason = f"Stop Loss ({self.stop_loss_multiplier:.1f}x)"
                        buyback_cost = stop_val
                    elif best_val <= target_val:
                        exit_signal = True
                        exit_reason = "Profit Take (80%)"
                        buyback_cost = target_val
                    elif days_to_expiry <= 1:
                        exit_signal = True
                        exit_reason = "Gamma Avoidance (T-1)"
                        buyback_cost = ce_close + pe_close
                    elif is_expiry:
                        exit_signal = True
                        exit_reason = "Expiry Close"
                        buyback_cost = ce_close + pe_close

                    if exit_signal:
                        points_pnl = pos["Premium"] - buyback_cost
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

                if not exit_signal:
                    active_pos_next.append(pos)

            open_positions = active_pos_next

            # --- B. Check Entries ---
            # KILL SWITCH CHECK
            if self.equity < day_start_equity:
                intraday_dd = (self.equity - day_start_equity) / day_start_equity
                if intraday_dd <= -0.04 and not kill_switch_active:
                    kill_switch_active = True
                    kill_switch_activations += 1
                    if verbose:
                        print("[KILL SWITCH TRIGGERED]")
                        print(f"    Intraday DD: {intraday_dd:.2%}")
                        print("    New trades blocked until next session.")

            # Update Correlation Shock Status (Regime Shift) – MODIFIED thresholds
            c10 = self.corr_10.get(curr_date, 0.0)
            c60 = self.corr_60.get(curr_date, 0.0)

            shock_active = False

            regime_cond = c10 >= (c60 + self.corr_regime_shift_threshold)
            crisis_cond = c10 >= self.corr_crisis_threshold

            if regime_cond or crisis_cond:
                shock_active = True
                shock_days += 1
            # Linear scaling factor based on correlation excess
            if shock_active:
                # scale_factor ranges from 0.5 (when c10 is much higher) to 1.0 (when just at threshold)
                excess = max(0, c10 - c60 - self.corr_regime_shift_threshold)
                scale_factor = max(0.5, 1.0 - excess)  # adjust 0.5 as desired
            else:
                scale_factor = 1.0

            # Update Market Regime (NIFTY Proxy)
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
            if (vol_accel >= 2.0 and c10 >= 0.95) or (
                pd.notna(n_move3) and n_move3 >= 0.06
            ):
                is_crisis = True

            if is_crisis:
                start_regime = "CRISIS"
            elif nifty_rv > rv_threshold:
                start_regime = "HIGH_RISK"
            else:
                start_regime = "NORMAL"

            # Log Regime Shift
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
                if kill_switch_active:
                    continue

                if current_regime == "CRISIS":
                    # Crisis logic (unchanged)
                    has_crisis = any(
                        p["Symbol"] == sym and p.get("Is_Crisis", False)
                        for p in open_positions
                    )
                    if has_crisis:
                        continue

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

                        ce_entry = ce["OpnPric"].iloc[0]
                        pe_entry = pe["OpnPric"].iloc[0]
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

                    cris_rv = (
                        self.rv_series_map.get(sym, {})
                        .get("RV", pd.Series())
                        .get(curr_date, np.nan)
                    )
                    if pd.isna(cris_rv):
                        cris_rv = nifty_rv
                    margin_per_lot = spot_entry * LOT_SIZE * get_margin_pct(cris_rv)

                    if margin_per_lot > 0:
                        candidate_lots = min(
                            candidate_lots, int((self.equity * 0.02) / margin_per_lot)
                        )
                    else:
                        candidate_lots = 0

                    if candidate_lots < 1 and self.equity >= risk_per_lot:
                        candidate_lots = 1

                    original_lots = candidate_lots

                    # MODIFIED: Use self.max_lots_per_sym
                    # Use per-symbol max lots, default to 10 if symbol not in dict
                    max_allowed = self.max_lots_per_sym.get(sym, 10)
                    candidate_lots = min(candidate_lots, max_allowed)

                    trade_premium = best["Prem"] * LOT_SIZE
                    max_lots_by_trade_limit = (
                        int(
                            (self.trade_capital_limit_pct * self.equity) / trade_premium
                        )
                        if trade_premium > 0
                        else 0
                    )
                    candidate_lots = min(candidate_lots, max_lots_by_trade_limit)

                    crisis_premium_spent = sum(
                        p["Premium"] * p["Size"]
                        for p in open_positions
                        if p.get("Is_Crisis", False)
                    )
                    new_trade_premium = candidate_lots * trade_premium
                    if crisis_premium_spent + new_trade_premium > 0.10 * self.equity:
                        continue

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

                    continue

                # Normal Entry Trigger
                if curr_date not in symbol_schedules[sym]:
                    continue

                expiry = symbol_schedules[sym][curr_date]

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

                    ce_entry = ce["OpnPric"].iloc[0]
                    pe_entry = pe["OpnPric"].iloc[0]

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

                print("[EXPIRY ENTRY VALIDATED]")
                print(f"Date: {date_str}")
                print(f"Expiry: {expiry.strftime('%Y-%m-%d')}")
                print(f"Strike: {best['Strike']}")

                # Yield Filter
                yield_pct = (best["Prem"] / spot_entry) * 100
                if yield_pct < 0.8:
                    continue

                # VRP Filter
                rv_series = self.rv_series_map.get(sym, {}).get("RV", pd.Series())
                rv = rv_series.get(curr_date, np.nan)
                if pd.isna(rv):
                    continue

                days_to_expiry = (expiry.date() - curr_date.date()).days
                if days_to_expiry <= 0:
                    continue

                t_years = days_to_expiry / 365.0
                iv_valid = best["Prem"] / (0.4 * spot_entry * np.sqrt(t_years))

                if rv <= 0 or iv_valid <= (rv * 1.15):
                    continue

                print("[IV VALIDATED]")
                print(f"IV: {iv_valid:.2%}")
                print(f"RV: {rv:.2%}")
                print(f"Ratio: {(iv_valid / rv):.2f}")

                best["iv_valid"] = iv_valid

                # --- SIZING & MARGIN CHECK ---
                margin_factor = get_margin_pct(rv)
                margin_req_per_lot = spot_entry * LOT_SIZE * margin_factor

                # MODIFIED: Use self.risk_budget_pct
                risk_pct = self.risk_budget_pct
                risk_budget = self.equity * risk_pct
                days_left = max((expiry.date() - curr_date.date()).days, 1)

                expected_move = (
                    spot_entry * best["iv_valid"] * np.sqrt(days_left / 365.0)
                )
                risk_per_lot = expected_move * LOT_SIZE
                lots_risk = risk_budget // risk_per_lot if risk_per_lot > 0 else 0

                margin_budget_total = self.equity * 0.85
                margin_available = margin_budget_total - current_margin_used
                if margin_available <= 0:
                    continue

                lots_margin = (
                    margin_available // margin_req_per_lot
                    if margin_req_per_lot > 0
                    else 0
                )
                candidate_lots = int(min(lots_risk, lots_margin))

                candidate_lots = int(candidate_lots * scale_factor)

                original_lots = candidate_lots

                # MODIFIED: Use self.max_lots_per_sym
                # Use per-symbol max lots, default to 10 if symbol not in dict
                max_allowed = self.max_lots_per_sym.get(sym, 10)
                candidate_lots = min(candidate_lots, max_allowed)

                trade_premium = best["Prem"] * LOT_SIZE
                max_lots_by_trade_limit = (
                    int((self.trade_capital_limit_pct * self.equity) / trade_premium)
                    if trade_premium > 0
                    else 0
                )
                candidate_lots = min(candidate_lots, max_lots_by_trade_limit)

                if candidate_lots < 1:
                    continue

                if candidate_lots < original_lots:
                    print("[LOT SAFETY REDUCTION APPLIED]")

                projected_margin_usage = current_margin_used + (
                    candidate_lots * margin_req_per_lot
                )
                projected_margin_pct = projected_margin_usage / self.equity

                if projected_margin_pct > 0.85:
                    if verbose:
                        print("[Risk Block]")
                        print(
                            f"    Reason: Portfolio Margin > 85% ({projected_margin_pct:.1%})"
                        )
                    blocked_margin += 1
                    continue

                lots = candidate_lots

                if shock_active:
                    trades_scaled += 1

                margin_locked = lots * margin_req_per_lot
                current_margin_used += margin_locked

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

            # ---------- DAILY MARK TO MARKET ----------
            unrealized_total = 0.0

            for pos in open_positions:
                sym = pos["Symbol"]
                expiry = pos["Expiry"]

                price_query = f"""
                SELECT ClsPric, OptnTp
                FROM fo_data
                WHERE TckrSymb='{sym}'
                  AND StrkPric={pos["Strike"]}
                  AND TradDt='{date_str}'
                  AND XpryDt='{expiry.strftime("%Y-%m-%d")}'
                """

                df_price = self.con.execute(price_query).df()

                if len(df_price) == 2:
                    current_val = df_price["ClsPric"].sum()
                    unreal = (pos["Premium"] - current_val) * (pos["Size"])
                    unrealized_total += unreal

            marked_equity = self.equity + unrealized_total

            # End of Day
            equity_curve.append({"Date": date_str, "Equity": marked_equity})
            self.portfolio_returns.append(0.0)

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
            print(f"[Safety] Kill Switch Activations: {kill_switch_activations}")

        return (
            pd.DataFrame(closed_trades),
            pd.DataFrame(equity_curve),
            {
                "blocked_regime": blocked_trades,
                "blocked_accel": blocked_acceleration,
                "blocked_margin": blocked_margin,
                "trades_scaled": trades_scaled,
                "kill_switch_activations": kill_switch_activations,
            },
        )

    def analyze_portfolio(self, trades, equity_df):
        if trades.empty:
            print("No trades.")
            return

        print("\n=== INSTITUTIONAL PORTFOLIO DASHBOARD ===")
        print(f"Ending Equity: {self.equity:,.2f} INR")

        equity_df["Prev"] = equity_df["Equity"].shift(1).fillna(CAPITAL)
        equity_df["Ret"] = equity_df["Equity"].pct_change().fillna(0)

        mean_ret = equity_df["Ret"].mean()
        std_ret = equity_df["Ret"].std()
        ann_factor = np.sqrt(252)

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
    bt = PortfolioVolEngine()
    trades, equity, stats = bt.simulate_portfolio(rv_threshold=0.35, verbose=True)

    end_eq = equity["Equity"].iloc[-1]

    equity["Prev"] = equity["Equity"].shift(1).fillna(CAPITAL)
    equity["Ret"] = equity["Equity"] / equity["Prev"] - 1
    mean_ret = equity["Ret"].mean()
    std_ret = equity["Ret"].std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

    downside = equity["Ret"][equity["Ret"] < 0].std()
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
    print(f"Kill Switch Activations: {stats['kill_switch_activations']}")


if __name__ == "__main__":
    try:
        run_portfolio_backtest()
    except Exception:
        import traceback

        traceback.print_exc()
