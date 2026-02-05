from dataclasses import dataclass


@dataclass
class AccountState:
    equity: float
    margin_used: float
    peak_equity: float  # Tracked externally or here
    net_vega: float
    net_gamma: float


class KillSwitch:
    """
    Global circuit breaker for trading operations.
    """

    def __init__(self, max_drawdown_pct: float = 0.10):
        self.max_drawdown_pct = max_drawdown_pct
        self._triggered = False

    def check(self, state: AccountState) -> bool:
        """
        Returns True if trading should STOP.
        """
        if self._triggered:
            return True

        return self._check_drawdown(state)

    def _check_drawdown(self, state: AccountState) -> bool:
        if state.peak_equity <= 0:
            return False

        dd = (state.peak_equity - state.equity) / state.peak_equity
        if dd > self.max_drawdown_pct:
            print(
                f"[KillSwitch] TRIGGERED: Drawdown {dd:.2%} exceeds limit {self.max_drawdown_pct:.2%}"
            )
            self._triggered = True
            return True

        return False

    def reset(self):
        self._triggered = False
