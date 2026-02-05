import polars as pl
from quant_repo.validation.definitions import ValidationResult, ValidationStatus


class BiasDetector:
    """
    Checks for structural biases in the trade log.
    """

    def check_lookahead(self, df: pl.DataFrame) -> ValidationResult:
        """
        Asserts that Entry Time >= Signal Info Check ?
        Or checks if Fill Price equals Next Open (valid) vs High/Low of *same* bar (potential lookahead).
        Since we don't have bar data here, we check implicit timestamps if available.
        For now, let's check basic timestamp ordering.
        Assume 'entry_time' and 'exit_time'.
        """
        if "entry_time" not in df.columns or "exit_time" not in df.columns:
            return ValidationResult(
                "Lookahead Check", ValidationStatus.WARNING, 0, "Missing timestamps"
            )

        invalid_trades = df.filter(pl.col("exit_time") <= pl.col("entry_time"))
        count = len(invalid_trades)

        if count > 0:
            return ValidationResult(
                "Lookahead Check",
                ValidationStatus.FAIL,
                0,
                f"Found {count} trades closing before/at open.",
            )

        return ValidationResult(
            "Lookahead Check", ValidationStatus.PASS, 100, "Timestamps valid."
        )

    def check_liquidity_illusion(self, df_trades: pl.DataFrame) -> ValidationResult:
        """
        Checks if fills are realistic.
        Since we simulated execution, we expect slippage > 0 often.
        If 'cost_slippage' column exists and is all 0, it's suspicious for Limit Orders or Mid fills.
        """
        if "cost_slippage" not in df_trades.columns:
            return ValidationResult(
                "Liquidity Check", ValidationStatus.WARNING, 0, "No slippage data"
            )

        # Check if 100% of trades have 0 slippage (Mid price fills?)
        zero_slip = df_trades.filter(pl.col("cost_slippage") == 0)
        pct_perfect = len(zero_slip) / len(df_trades) if len(df_trades) > 0 else 0

        if pct_perfect > 0.95:
            return ValidationResult(
                "Liquidity Illusion",
                ValidationStatus.FAIL,
                0,
                f"{pct_perfect:.1%} of trades have ZERO slippage. Unrealistic.",
            )

        return ValidationResult(
            "Liquidity Illusion",
            ValidationStatus.PASS,
            100,
            f"Perfect fills: {pct_perfect:.1%}",
        )
