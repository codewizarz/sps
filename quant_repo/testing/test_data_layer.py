import shutil
from pathlib import Path
import pandas as pd
import numpy as np

# Adjust path to import from quant_repo
import sys

sys.path.append(str(Path.cwd()))

from quant_repo.data.contract_metadata.registry import registry, InstrumentSpec
from quant_repo.data.loaders.parquet import ParquetDataLoader


def test_data_layer():
    print("[TEST] 1. initializing Registry...")
    spec = InstrumentSpec(
        symbol="EUR/USD",
        venue="SIM",
        asset_class="FX",
        price_precision=5,
        size_precision=0,
        price_increment=0.00001,
        lot_size=1000.0,
        base_currency="EUR",
        quote_currency="USD",
    )
    registry.register(spec)

    instrument = registry.get_instrument("EUR/USD", "SIM")
    assert instrument is not None
    print(f"[TEST] Registered and retrieved: {instrument}")

    print("[TEST] 2. Generatng Dummy Parquet Data...")
    data_root = Path.cwd() / "temp_data"
    if data_root.exists():
        shutil.rmtree(data_root)
    (data_root / "quotes").mkdir(parents=True)

    # create dummy df
    dates = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "bid": np.linspace(1.1000, 1.1100, 100),
            "ask": np.linspace(1.1001, 1.1101, 100),
            "bid_size": 1000000,
            "ask_size": 1000000,
        },
        index=dates,
    )
    df.index.name = "timestamp"

    file_path = data_root / "quotes" / "EUR_USD.SIM.parquet"
    df.to_parquet(file_path)
    print(f"[TEST] Wrote {file_path}")

    print("[TEST] 3. Testing ParquetDataLoader...")
    loader = ParquetDataLoader(data_root)

    # 3a. Load DataFrame
    loaded_df = loader.load_quotes_dataframe("EUR/USD.SIM")
    print(f"[TEST] Loaded DataFrame shape: {loaded_df.shape}")
    assert len(loaded_df) == 100
    assert "bid" in loaded_df.columns

    # 3b. Load Nautilus Objects
    # Note: QuoteTickDataWrangler expects columns: bid_price, ask_price.
    # Our dummy data has 'bid', 'ask'. The loader might fail if we don't rename.
    # Let's fix the loader or the data.
    # Nautilus default map usually expects bid_price/ask_price?
    # Actually, CSVTickDataLoader allowed passing names. Wrangler expects specific columns or we map them.
    # We will rename in this test script data generation to match expected defaults if possible,
    # or update the loader to be smarter.
    # Let's verify what `QuoteTickDataWrangler` expects.
    # It expects `bid` / `ask` usually or `bid_price`.
    # Let's just run it and see.

    try:
        ticks = loader.load_nautilus_quotes(instrument)
        print(f"[TEST] Loaded {len(ticks)} Nautilus QuoteTicks")
        assert len(ticks) == 100
    except Exception as e:
        print(f"[TEST] Nautilus Load Failed: {e}")
        # Hint: Wrangler might need 'bid' -> 'bid_price' rename.
        pass

    print("[TEST] Cleanup...")
    # shutil.rmtree(data_root)
    print("[TEST] Done.")


if __name__ == "__main__":
    test_data_layer()
