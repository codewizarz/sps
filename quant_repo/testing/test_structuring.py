import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from quant_repo.portfolio.selection import StrikeSelector, OptionChain, OptionInfo
from quant_repo.portfolio.structuring import StructureFactory
from quant_repo.portfolio.risk import (
    RiskFilter,
    RiskException,
    TradeStructure,
    TradeLeg,
)


def test_strategy_construction():
    print("[TEST] Strategy Construction Layer...")

    # 1. Setup Mock Option Chain
    # ATM = 100
    chain_data = [
        # Puts
        OptionInfo("P-90", 90, "PUT", -0.15, 0.1, 500),
        OptionInfo("P-95", 95, "PUT", -0.30, 0.1, 1000),
        OptionInfo("P-100", 100, "PUT", -0.50, 0.1, 2000),
        # Calls
        OptionInfo("C-100", 100, "CALL", 0.50, 0.1, 2000),
        OptionInfo("C-105", 105, "CALL", 0.30, 0.1, 1000),
        OptionInfo("C-110", 110, "CALL", 0.15, 0.1, 500),
    ]
    chain = OptionChain(chain_data)

    selector = StrikeSelector(min_oi=100)
    factory = StructureFactory(selector=selector)
    risk_filter = RiskFilter()

    # 2. Test Iron Condor Generation
    # Short Delta 30 (Sell C-105, Sell P-95)
    # Wing Width 15 Delta -> Long 15 Delta (Buy C-110, Buy P-90)

    print("\n[TEST] Generating Iron Condor...")
    ic = factory.create_iron_condor(chain, short_delta=0.30, wing_width_deltas=0.15)

    print(f"  Structure: {ic.strategy_type}")
    for leg in ic.legs:
        print(f"    {leg.ratio}x {leg.instrument_id} ({leg.option_kind})")

    # Verify Legs
    # Short P-95 (-1), Long P-90 (+1)
    # Short C-105 (-1), Long C-110 (+1)

    ids = [l.instrument_id for l in ic.legs]
    assert "P-95" in ids
    assert "P-90" in ids
    assert "C-105" in ids
    assert "C-110" in ids

    # 3. Test Risk Filtering
    print("\n[TEST] Risk Filtering...")

    # Valid IC check
    try:
        risk_filter.validate(ic)
        print("  Iron Condor: PASS (Expected)")
    except RiskException as e:
        print(f"  Iron Condor: FAIL ({e})")
        raise

    # Naked Short Call Check
    naked_call = TradeStructure(
        legs=[TradeLeg("C-100", -1, "CALL")], strategy_type="NAKED_SHORT"
    )

    try:
        risk_filter.validate(naked_call)
        print("  Naked Call: PASS (Unexpected!)")
        assert False, "Should have rejected Naked Call"
    except RiskException as e:
        print(f"  Naked Call: REJECTED (Expected: {e})")

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_strategy_construction()
