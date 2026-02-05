import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from quant_repo.testing.generate_mock_options import generate_mock_options_data
from quant_repo.data.contract_metadata.options_factory import IndianOptionsFactory
from quant_repo.data.loaders.options_loader import LazyOptionsOptionsLoader


def test_options_adapter():
    # 1. Generate Data
    dataset_path = generate_mock_options_data()

    # 2. Test Discovery
    print("[TEST] Discovering instruments...")
    factory = IndianOptionsFactory(dataset_path)
    options = factory.discover_instruments(
        symbol="NIFTY", start_date="2024-01-01", end_date="2024-01-31"
    )

    print(f"[TEST] Found {len(options)} options.")
    for o in options:
        print(f"  - {o.id} ({o.option_kind}) K={o.strike_price}")

    assert len(options) == 4  # 2 strikes * 2 rights

    # 3. Test Loading
    print("[TEST] Streaming ticks...")
    loader = LazyOptionsOptionsLoader(dataset_path)

    # Pick one option to load
    target_opt = options[0]
    print(f"[TEST] Loading for {target_opt.id}")

    ticks = list(loader.load_ticks(target_opt, "2024-01-01", "2024-01-02"))
    print(f"[TEST] Loaded {len(ticks)} ticks.")

    assert len(ticks) > 0
    first = ticks[0]
    print(f"  First Tick: {first}")
    assert first.instrument_id == target_opt.id

    print("[TEST] SUCCESS")


if __name__ == "__main__":
    test_options_adapter()
