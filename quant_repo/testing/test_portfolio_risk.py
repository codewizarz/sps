import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from quant_repo.portfolio.risk_manager import PortfolioRiskManager
from quant_repo.portfolio.kill_switch import KillSwitch, AccountState
from quant_repo.portfolio.sizing import VolTargetSizer
from quant_repo.portfolio.risk import TradeStructure  # Mock structure


def test_portfolio_risk():
    print("[TEST] Portfolio Risk Layer...")

    # 1. Setup
    kill_switch = KillSwitch(max_drawdown_pct=0.10)
    sizer = VolTargetSizer(target_vol_pct_per_trade=0.001)  # 0.1% per trade
    manager = PortfolioRiskManager(
        kill_switch=kill_switch,
        sizer=sizer,
        max_net_vega_pct=0.01,  # 1% Max Net Vega
    )

    # Mock Account: 1M Equity
    # Target Trade Vega = 1M * 0.001 = 1000 Vega
    account = AccountState(
        equity=1_000_000,
        margin_used=100_000,
        peak_equity=1_050_000,  # Small Drawdown
        net_vega=5000,  # 0.5% current exposure
        net_gamma=0,
    )

    # Mock Trade: Iron Condor with Unit Vega = 50
    # Expected Size = 1000 / 50 = 20 lots
    structure_unit_vega = 50.0
    structure_unit_margin = 2000.0  # 20 * 2000 = 40k Margin

    # Trade Mock
    trade = TradeStructure(legs=[], strategy_type="IRON_CONDOR")

    # 2. Test Normal Sizing
    print("\n[TEST] Normal Sizing...")
    qty = manager.check_and_size(
        trade, account, structure_unit_vega, structure_unit_margin
    )
    print(f"  Qty: {qty}")
    assert qty == 20

    # 3. Test Vega Limit Breach
    print("\n[TEST] Vega Limit Breach...")
    # Current Vega = 9500 (Limit is 10,000)
    # Trade adds 20 * 50 = 1000. New Total = 10,500 > Limit.
    # Should reject or resize.

    account_risky = AccountState(
        equity=1_000_000,
        margin_used=100_000,
        peak_equity=1_000_000,
        net_vega=9500,  # Near limit
        net_gamma=0,
    )

    qty_risky = manager.check_and_size(
        trade, account_risky, structure_unit_vega, structure_unit_margin
    )
    print(f"  Qty Risky: {qty_risky}")
    # Implementation currently rejects (returns 0) if limit is hit pro-forma
    assert qty_risky == 0

    # 4. Test Kill Switch
    print("\n[TEST] Kill Switch...")
    # 1M Peak, 800k Current -> 20% Drawdown > 10% Limit
    account_dead = AccountState(
        equity=800_000,
        margin_used=100_000,
        peak_equity=1_000_000,
        net_vega=0,
        net_gamma=0,
    )

    qty_dead = manager.check_and_size(
        trade, account_dead, structure_unit_vega, structure_unit_margin
    )
    print(f"  Qty Dead: {qty_dead}")
    assert qty_dead == 0
    assert manager.kill_switch._triggered == True

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_portfolio_risk()
