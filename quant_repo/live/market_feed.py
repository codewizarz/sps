#!/usr/bin/env python3
"""
=============================================================================
PAPER TRADING — MARKET FEED
=============================================================================
Connects to Angel One SmartAPI WebSocket V2 for live tick data.
Streams NIFTY, BANKNIFTY spot + ATM option chain.

Token references (exchange_type):
  1 = NSE (Cash/Index)
  2 = NFO (F&O)

Mode:
  1 = LTP only
  2 = Quote (LTP + OHLC + volume)
  3 = Snap Quote (full depth)

SmartWebSocketV2 key tokens:
  NIFTY 50       = 99926000 (NSE index)
  BANK NIFTY     = 99926009 (NSE index)
  Option tokens  = dynamic (looked up via REST API)
=============================================================================
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

try:
    from SmartApi import SmartConnect
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
    HAS_SMARTAPI = True
except ImportError:
    HAS_SMARTAPI = False


# ── Data structures ─────────────────────────────────────────────────────

@dataclass
class Tick:
    """Normalized tick payload for downstream consumption."""
    symbol: str                # e.g. "NIFTY", "BANKNIFTY"
    token: str                 # instrument token
    ltp: float                 # last traded price
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    oi: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    exchange_type: int = 1     # 1=NSE, 2=NFO
    is_option: bool = False
    option_type: str = ""      # "CE" or "PE"
    strike: float = 0.0
    expiry: str = ""


@dataclass
class MarketSnapshot:
    """Aggregated view of current market state."""
    nifty_spot: float = 0.0
    banknifty_spot: float = 0.0
    option_ticks: Dict[str, Tick] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def is_stale(self) -> bool:
        return (datetime.now() - self.last_update).seconds > 30


# ── Simulated Feed (for paper trading without live API) ─────────────────

class SimulatedFeed:
    """
    Generates synthetic ticks from the DuckDB data lake.
    Used when Angel One credentials are not available.
    Replays the most recent trading day's data at accelerated speed.
    """

    def __init__(
        self,
        data_lake_path: str,
        symbols: List[str] = None,
        replay_speed: float = 1.0,
    ):
        self.data_lake_path = data_lake_path
        self.symbols = symbols or ["NIFTY", "BANKNIFTY"]
        self.replay_speed = replay_speed
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        self._snapshot = MarketSnapshot()

    def register_callback(self, callback: Callable[[Tick], None]):
        self._callbacks.append(callback)

    @property
    def snapshot(self) -> MarketSnapshot:
        return self._snapshot

    def start(self):
        """Start replaying historical data as synthetic ticks."""
        self._running = True
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _replay_loop(self):
        """Replay most recent data lake entries as ticks."""
        try:
            import duckdb
            from pathlib import Path

            con = duckdb.connect(":memory:")
            pattern = str(Path(self.data_lake_path) / "**" / "*.parquet")
            con.execute("INSTALL parquet; LOAD parquet;")
            con.execute(
                f"CREATE OR REPLACE VIEW fo_data AS "
                f"SELECT * FROM read_parquet('{pattern}', hive_partitioning=true);"
            )

            # Get latest trading date
            latest_date = con.execute(
                "SELECT MAX(TradDt) FROM fo_data"
            ).fetchone()[0]

            for symbol in self.symbols:
                spot_row = con.execute(
                    f"SELECT AVG(UndrlygPric) AS spot FROM fo_data "
                    f"WHERE TckrSymb = '{symbol}' AND TradDt = '{latest_date}'"
                ).fetchone()
                if spot_row and spot_row[0]:
                    spot = float(spot_row[0])
                    tick = Tick(
                        symbol=symbol,
                        token=f"SIM_{symbol}",
                        ltp=spot,
                        close=spot,
                        exchange_type=1,
                    )
                    self._dispatch(tick)

                    # Also fetch ATM options for this symbol
                    self._replay_options(con, symbol, latest_date, spot)

            con.close()

            # Keep alive with heartbeat ticks
            while self._running:
                time.sleep(max(1.0 / self.replay_speed, 0.5))
                self._snapshot.last_update = datetime.now()

        except Exception as e:
            print(f"[SimulatedFeed] Error: {e}")
            import traceback
            traceback.print_exc()

    def _replay_options(self, con, symbol: str, date: str, spot: float):
        """Fetch ATM option pair from lake and emit ticks."""
        try:
            options = con.execute(f"""
                SELECT OptnTp, StrkPric, ClsPric, HghPric, LwPric,
                       OpnIntrst, TtlTradgVol, XpryDt, NewBrdLotQty
                FROM fo_data
                WHERE TckrSymb = '{symbol}'
                  AND TradDt = '{date}'
                  AND OptnTp IN ('CE','PE')
                ORDER BY ABS(StrkPric - {spot}) ASC
                LIMIT 20
            """).df()

            if options.empty:
                return

            for _, row in options.iterrows():
                tick = Tick(
                    symbol=symbol,
                    token=f"SIM_{symbol}_{row['OptnTp']}_{int(row['StrkPric'])}",
                    ltp=float(row["ClsPric"]),
                    high=float(row["HghPric"]),
                    low=float(row["LwPric"]),
                    close=float(row["ClsPric"]),
                    oi=int(row.get("OpnIntrst", 0) or 0),
                    volume=int(row.get("TtlTradgVol", 0) or 0),
                    exchange_type=2,
                    is_option=True,
                    option_type=str(row["OptnTp"]),
                    strike=float(row["StrkPric"]),
                    expiry=str(row["XpryDt"]),
                )
                self._dispatch(tick)
                key = f"{symbol}_{row['OptnTp']}_{int(row['StrkPric'])}"
                self._snapshot.option_ticks[key] = tick

        except Exception as e:
            print(f"[SimulatedFeed] Option replay error: {e}")

    def _dispatch(self, tick: Tick):
        """Send tick to all registered callbacks and update snapshot."""
        if tick.symbol == "NIFTY" and not tick.is_option:
            self._snapshot.nifty_spot = tick.ltp
        elif tick.symbol == "BANKNIFTY" and not tick.is_option:
            self._snapshot.banknifty_spot = tick.ltp
        self._snapshot.last_update = datetime.now()

        for cb in self._callbacks:
            try:
                cb(tick)
            except Exception:
                pass


# ── Live Angel One Feed ─────────────────────────────────────────────────

class AngelOneFeed:
    """
    Live market feed using Angel One SmartAPI WebSocket V2.
    Requires: pip install smartapi-python pyotp logzero websocket-client
    """

    # Well-known index tokens
    INDEX_TOKENS = {
        "NIFTY": "99926000",
        "BANKNIFTY": "99926009",
    }

    def __init__(
        self,
        api_key: str,
        client_code: str,
        password: str,
        totp_secret: str,
        symbols: List[str] = None,
    ):
        if not HAS_SMARTAPI:
            raise ImportError(
                "smartapi-python is not installed. "
                "Run: pip install smartapi-python pyotp logzero websocket-client"
            )

        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_secret = totp_secret
        self.symbols = symbols or ["NIFTY", "BANKNIFTY"]

        self._smart_api: Optional[SmartConnect] = None
        self._ws: Optional[SmartWebSocketV2] = None
        self._callbacks: List[Callable] = []
        self._snapshot = MarketSnapshot()
        self._auth_token = ""
        self._feed_token = ""
        self._running = False

    def register_callback(self, callback: Callable[[Tick], None]):
        self._callbacks.append(callback)

    @property
    def snapshot(self) -> MarketSnapshot:
        return self._snapshot

    def authenticate(self) -> bool:
        """Login to Angel One and get auth + feed tokens."""
        try:
            import pyotp

            self._smart_api = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now()

            session = self._smart_api.generateSession(
                self.client_code, self.password, totp
            )
            if not session.get("status"):
                print(f"[AngelOneFeed] Auth failed: {session}")
                return False

            self._auth_token = session["data"]["jwtToken"]
            self._feed_token = self._smart_api.getfeedToken()
            return True

        except Exception as e:
            print(f"[AngelOneFeed] Auth error: {e}")
            return False

    def start(self):
        """Connect WebSocket and subscribe to index + option tokens."""
        if not self._auth_token:
            if not self.authenticate():
                raise RuntimeError("Authentication failed")

        self._ws = SmartWebSocketV2(
            self._auth_token,
            self.api_key,
            self.client_code,
            self._feed_token,
        )

        self._ws.on_open = self._on_open
        self._ws.on_data = self._on_data
        self._ws.on_error = self._on_error
        self._ws.on_close = self._on_close

        self._running = True
        self._ws.connect()

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close_connection()
            except Exception:
                pass

    def _on_open(self, wsapp):
        """Subscribe to index tokens on connection."""
        tokens = [
            self.INDEX_TOKENS[s] for s in self.symbols
            if s in self.INDEX_TOKENS
        ]
        if tokens:
            self._ws.subscribe(
                "paper_trader",
                2,  # Quote mode
                [{"exchangeType": 1, "tokens": tokens}],
            )

    def _on_data(self, wsapp, message):
        """Process incoming tick data."""
        try:
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = message

            token = str(data.get("token", ""))
            ltp = float(data.get("last_traded_price", 0)) / 100.0  # API sends in paise

            # Map token back to symbol
            symbol = None
            for sym, tok in self.INDEX_TOKENS.items():
                if tok == token:
                    symbol = sym
                    break

            if symbol:
                tick = Tick(
                    symbol=symbol,
                    token=token,
                    ltp=ltp,
                    open=float(data.get("open_price_of_the_day", 0)) / 100.0,
                    high=float(data.get("high_price_of_the_day", 0)) / 100.0,
                    low=float(data.get("low_price_of_the_day", 0)) / 100.0,
                    close=float(data.get("closed_price", 0)) / 100.0,
                    volume=int(data.get("volume_trade_for_the_day", 0)),
                    exchange_type=1,
                )
                self._dispatch(tick)

        except Exception as e:
            print(f"[AngelOneFeed] Tick error: {e}")

    def _on_error(self, wsapp, error):
        print(f"[AngelOneFeed] WS Error: {error}")

    def _on_close(self, wsapp):
        print("[AngelOneFeed] WS Closed")
        self._running = False

    def _dispatch(self, tick: Tick):
        if tick.symbol == "NIFTY" and not tick.is_option:
            self._snapshot.nifty_spot = tick.ltp
        elif tick.symbol == "BANKNIFTY" and not tick.is_option:
            self._snapshot.banknifty_spot = tick.ltp
        self._snapshot.last_update = datetime.now()

        for cb in self._callbacks:
            try:
                cb(tick)
            except Exception:
                pass


# ── Factory ─────────────────────────────────────────────────────────────

def create_feed(
    mode: str = "simulated",
    data_lake_path: str = "",
    api_key: str = "",
    client_code: str = "",
    password: str = "",
    totp_secret: str = "",
    symbols: List[str] = None,
):
    """
    Factory to create the appropriate feed.
    mode = "simulated" | "live"
    """
    if mode == "live":
        return AngelOneFeed(
            api_key=api_key,
            client_code=client_code,
            password=password,
            totp_secret=totp_secret,
            symbols=symbols,
        )
    else:
        return SimulatedFeed(
            data_lake_path=data_lake_path,
            symbols=symbols or ["NIFTY", "BANKNIFTY"],
        )
