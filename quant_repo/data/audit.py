import polars as pl
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AuditReport:
    total_rows: int
    clean_rows: int
    health_score: float  # 0-100
    issues: Dict[str, int]
    bad_rows: pl.DataFrame


class DataAuditor:
    """
    Automated quality control for options data.
    Detects integrity issues before research begins.
    """

    def __init__(self):
        pass

    def audit_dataframe(self, df: pl.DataFrame) -> AuditReport:
        """
        Expects columns: [date, symbol, expiry, strike, type, bid, ask, close, volume]
        """
        total_rows = len(df)
        if total_rows == 0:
            return AuditReport(0, 0, 100.0, {}, pl.DataFrame())

        issues = {}

        # 1. Duplicate Check
        # Key: date, symbol, expiry, strike, type
        # We can't easily flag specific rows as 'bad' without a unique ID,
        # but we can count duplicates.
        n_unique = df.n_unique(subset=["date", "symbol", "expiry", "strike", "type"])
        n_duplicates = total_rows - n_unique
        if n_duplicates > 0:
            issues["duplicates"] = n_duplicates

        # 2. Logic Checks (Vectorized)

        # Crossed Markets
        crossed_expr = pl.col("bid") > pl.col("ask")

        # Zero/Negative Prices (Assuming 'close' is the main price field)
        zero_price_expr = pl.col("close") <= 0

        # Broken Expiries (Expiry < Date)
        # Ensure date columns are date type
        broken_expiry_expr = pl.col("expiry") < pl.col("date")

        # Create a boolean mask for bad rows (excluding duplicates which are harder to mask 1-to-1 here)
        bad_mask = crossed_expr | zero_price_expr | broken_expiry_expr

        bad_rows = df.filter(bad_mask)

        # Count specific issues
        issues["crossed_markets"] = df.filter(crossed_expr).height
        issues["zero_prices"] = df.filter(zero_price_expr).height
        issues["broken_expiries"] = df.filter(broken_expiry_expr).height

        # 3. Strike Gaps (Structural)
        # For each date/expiry/type, strikes should be equidistant (mostly)
        # This is expensive to check on full history.
        # Simplified Check: Check for massive gaps in sorted strikes?
        # Let's skip complex structural checks for the 'fast' audit and focus on row-level integrity.

        # 4. Returns/Spikes (Time series)
        # Requires sorting.
        # df = df.sort(["date"])
        # returns = df["close"].pct_change()
        # threshold = returns.std() * 50
        # spikes = (returns.abs() > threshold).sum()
        # issues["extreme_spikes"] = spikes

        # Calculate Score
        total_issues = sum(issues.values())
        # Penalize health.
        # Duplicates count as 1 issue per row.

        clean_rows = total_rows - bad_rows.height - n_duplicates

        # Score: Clean / Total * 100
        # But if total is huge, 99.9% might still hide critical errors.
        # Let's use strictly Row %
        health_score = (clean_rows / total_rows) * 100.0

        return AuditReport(
            total_rows=total_rows,
            clean_rows=clean_rows,
            health_score=health_score,
            issues=issues,
            bad_rows=bad_rows,
        )
