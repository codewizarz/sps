import shutil
from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader.backtest.node import BacktestDataConfig
from nautilus_trader.backtest.node import BacktestEngineConfig
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.backtest.node import BacktestRunConfig
from nautilus_trader.backtest.node import BacktestVenueConfig
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model import QuoteTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler
from nautilus_trader.test_kit.providers import CSVTickDataLoader
from nautilus_trader.test_kit.providers import TestInstrumentProvider


DATA_DIR = Path.cwd()
CSV_FILE = DATA_DIR / "DAT_ASCII_EURUSD_T_202601.csv"

# Generate synthetic data if not exists
if not CSV_FILE.exists():
    print("Generating synthetic data...")
    with open(CSV_FILE, "w") as f:
        # timestamp, bid, ask, vol
        # 20260101 000000000,1.10000,1.10010,1000000
        for i in range(100):
            ts = pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i)
            ts_str = ts.strftime("%Y%m%d %H%M%S%f")[
                :-3
            ]  # match format roughly or adjust parser
            # Parser expects %Y%m%d %H%M%S%f
            ts_str = ts.strftime("%Y%m%d %H%M%S") + "000"  # 20260101 000000000
            f.write(f"{ts_str},1.10000,1.10010,1000000\n")

raw_files = [CSV_FILE]


# Load the first CSV file into a pandas DataFrame
df = CSVTickDataLoader.load(
    file_path=raw_files[0],
    index_col=0,
    header=None,
    names=["timestamp", "bid_price", "ask_price", "volume"],
    usecols=["timestamp", "bid_price", "ask_price"],
    parse_dates=["timestamp"],
    date_format="%Y%m%d %H%M%S%f",
)

df = df.sort_index()
df.head(2)


# Process quotes using a wrangler
EURUSD = TestInstrumentProvider.default_fx_ccy("EUR/USD")
wrangler = QuoteTickDataWrangler(EURUSD)

ticks = wrangler.process(df)

# Preview: see first 2 ticks
ticks[0:2]

CATALOG_PATH = Path.cwd() / "catalog"

# Clear if it already exists, then create fresh
if CATALOG_PATH.exists():
    shutil.rmtree(CATALOG_PATH)
CATALOG_PATH.mkdir(parents=True)

# Create a catalog instance
catalog = ParquetDataCatalog(CATALOG_PATH)

# Write instrument to the catalog
catalog.write_data([EURUSD])

# Write ticks to catalog
catalog.write_data(ticks)

# Get list of all instruments in catalog
catalog.instruments()

# See 1st instrument from catalog
instrument = catalog.instruments()[0]
instrument


# Query quote ticks from catalog to determine the data range
all_ticks = catalog.quote_ticks(instrument_ids=[EURUSD.id.value])
print(f"Total ticks in catalog: {len(all_ticks)}")

if all_ticks:
    # Get timestamps from the data
    first_tick_time = pd.Timestamp(all_ticks[0].ts_init, unit="ns", tz="UTC")
    last_tick_time = pd.Timestamp(all_ticks[-1].ts_init, unit="ns", tz="UTC")
    print(f"Data range: {first_tick_time} to {last_tick_time}")

    # Set backtest range to first 2 weeks of data (as ISO strings for BacktestDataConfig)
    start_time = first_tick_time.isoformat()
    end_time = (first_tick_time + pd.Timedelta(days=14)).isoformat()
    print(f"Backtest range: {start_time} to {end_time}")

    # Preview selected data
    start_ns = all_ticks[0].ts_init
    end_ns = dt_to_unix_nanos(first_tick_time + pd.Timedelta(days=14))
    selected_quote_ticks = catalog.quote_ticks(
        instrument_ids=[EURUSD.id.value], start=start_ns, end=end_ns
    )
    print(f"Selected ticks for backtest: {len(selected_quote_ticks)}")
    selected_quote_ticks[:2]
venue_configs = [
    BacktestVenueConfig(
        name="SIM",
        oms_type="HEDGING",
        account_type="MARGIN",
        base_currency="USD",
        starting_balances=["1_000_000 USD"],
    ),
]

str(CATALOG_PATH)

data_configs = [
    BacktestDataConfig(
        catalog_path=str(CATALOG_PATH),
        data_cls=QuoteTick,
        instrument_id=instrument.id,
        start_time=start_time,
        end_time=end_time,
    ),
]


strategies = [
    ImportableStrategyConfig(
        strategy_path="nautilus_trader.examples.strategies.ema_cross:EMACross",
        config_path="nautilus_trader.examples.strategies.ema_cross:EMACrossConfig",
        config={
            "instrument_id": instrument.id,
            "bar_type": "EUR/USD.SIM-15-MINUTE-BID-INTERNAL",
            "fast_ema_period": 10,
            "slow_ema_period": 20,
            "trade_size": Decimal(1_000_000),
        },
    ),
]

config = BacktestRunConfig(
    engine=BacktestEngineConfig(strategies=strategies),
    data=data_configs,
    venues=venue_configs,
)

node = BacktestNode(configs=[config])

results = node.run()
print(results)
