import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import pytest
from quant_repo.portfolio.smart_kill_switch import (
    SmartKillSwitch,
    KillStatus,
    KillSwitchConfig,
)
from quant_repo.analytics.gamma_regime import GammaRegime


def test_smart_kill_switch():
    print("[TEST] Smart Kill Switch...")

    config = KillSwitchConfig(
        max_drawdown_multiplier=1.5,
        liquidity_spread_threshold=0.01,
        strategy_type="SHORT_VOL",
    )
    ks = SmartKillSwitch(config)

    # 1. Normal State
    pnl = {"current_drawdown_pct": 0.05, "expected_drawdown_pct": 0.10}
    regime = GammaRegime.LONG_GAMMA
    liq = {"bid_ask_spread_pct": 0.001}
    exec_metrics = {"realized_slippage_pct": 0.0001, "modeled_slippage_pct": 0.0001}

    print("\n--- Checking Normal State ---")
    status = ks.check_status(pnl, regime, liq, exec_metrics)
    print(f"Status: {status}")
    assert status == KillStatus.ACTIVE

    # 2. Regime Trigger (Short Gamma)
    print("\n--- Checking Regime Trigger (Short Gamma) ---")
    status = ks.check_status(pnl, GammaRegime.SHORT_GAMMA, liq, exec_metrics)
    print(f"Status: {status} | Reason: {ks.reason}")
    assert status == KillStatus.PAUSED
    assert "Regime Mismatch" in ks.reason

    # 3. Liquidity Trigger
    print("\n--- Checking Liquidity Trigger ---")
    liq_bad = {"bid_ask_spread_pct": 0.02}  # 2% spread
    status = ks.check_status(pnl, GammaRegime.LONG_GAMMA, liq_bad, exec_metrics)
    print(f"Status: {status} | Reason: {ks.reason}")
    assert status == KillStatus.PAUSED
    assert "Liquidity" in ks.reason

    # 4. Drawdown Kill
    print("\n--- Checking Drawdown Kill ---")
    pnl_bad = {"current_drawdown_pct": 0.16, "expected_drawdown_pct": 0.10}
    # Limit = 0.10 * 1.5 = 0.15. 0.16 > 0.15.

    status = ks.check_status(pnl_bad, GammaRegime.LONG_GAMMA, liq, exec_metrics)
    print(f"Status: {status} | Reason: {ks.reason}")
    assert status == KillStatus.KILLED
    assert "Drawdown" in ks.reason

    # 5. Verify Persistence of Kill
    # Even if PnL improves immediately (e.g. data glitch fixed?), it should remain KILLED until reset.
    print("\n--- Checking Kill Persistence ---")
    status_retry = ks.check_status(
        pnl, GammaRegime.LONG_GAMMA, liq, exec_metrics
    )  # Inputting normal pnl
    print(f"Status (Retry): {status_retry}")
    assert status_retry == KillStatus.KILLED

    # 6. Manual Reset
    print("\n--- Checking Manual Reset ---")
    ks.reset()
    status_reset = ks.check_status(pnl, GammaRegime.LONG_GAMMA, liq, exec_metrics)
    print(f"Status (Reset): {status_reset}")
    assert status_reset == KillStatus.ACTIVE

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_smart_kill_switch()
