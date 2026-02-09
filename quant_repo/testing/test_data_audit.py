import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import polars as pl
from datetime import date
from quant_repo.data.audit import DataAuditor


def test_data_auditor():
    print("[TEST] Trading Data Auditor...")

    # Create Mock Data with Known Issues
    data = pl.DataFrame(
        {
            "date": [
                date(2023, 1, 1),
                date(2023, 1, 1),
                date(2023, 1, 1),
                date(2023, 1, 2),
            ],
            "symbol": ["NIFTY", "NIFTY", "NIFTY", "NIFTY"],
            "expiry": [
                date(2023, 1, 5),
                date(2023, 1, 5),
                date(2022, 12, 31),
                date(2023, 1, 5),
            ],  # Row 3: Broken Expiry (2022 < 2023)
            "strike": [18000, 18000, 18100, 18000],
            "type": ["CE", "CE", "CE", "CE"],
            "bid": [100.0, 105.0, 90.0, 100.0],
            "ask": [102.0, 95.0, 92.0, 102.0],  # Row 2: Crossed (Bid 105 > Ask 95)
            "close": [101.0, 100.0, 91.0, 0.0],  # Row 4: Zero Price
            "volume": [100, 100, 100, 100],
        }
    )

    # Row 1 and Row 2 have same key (date, sym, exp, strike, type) -> Duplicate!

    auditor = DataAuditor()
    report = auditor.audit_dataframe(data)

    print(f"Health Score: {report.health_score:.2f}%")
    print("Issues Found:", report.issues)

    print("\nBad Rows:")
    print(report.bad_rows)

    # Verification
    # 1. Duplicates: Row 1 & 2 are duplicates of each other in terms of key.
    # n_unique = 3 (Key 1, Key 3, Key 4). Total 4. Duplicates = 1.
    assert report.issues.get("duplicates", 0) == 1

    # 2. Crossed: Row 2
    assert report.issues.get("crossed_markets", 0) == 1

    # 3. Broken Expiry: Row 3
    assert report.issues.get("broken_expiries", 0) == 1

    # 4. Zero Price: Row 4
    assert report.issues.get("zero_prices", 0) == 1

    # Total badness
    assert report.health_score < 100.0

    print("\n[TEST] SUCCESS")


if __name__ == "__main__":
    test_data_auditor()
