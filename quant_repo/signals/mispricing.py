import polars as pl
import numpy as np
from typing import List
from quant_repo.signals.definitions import Signal, SignalType, Direction
from quant_repo.features.volatility import (
    compute_realized_vol,
    compute_zscores,
    compute_iv_rv_spread,
)


class MispricingDetector:
    def __init__(self):
        pass

    def detect_vol_arb(
        self,
        df: pl.DataFrame,
        underlying_col: str = "spot_price",
        iv_col: str = "iv",
        rv_window: int = 30,
        z_window: int = 90,
        threshold: float = 2.0,
    ) -> List[Signal]:
        """
        Detects Volatility Arbitrage opportunities where IV diverges significantly from RV.

        Logic:
        1. Calculate RV from underlying price history.
        2. Calculate Spread = IV - RV.
        3. Calculate Z-Score of Spread.
        4. Signal if |Z-Score| > threshold.
        """
        # 1. Compute RV
        # Ensure rows are sorted by time per instrument if mixed (handling grouping outside or assuming single instrument df)
        # Here assuming strict time ordering for the passed DataFrame

        # Calculate features
        df_feats = (
            df.pipe(compute_realized_vol, price_col=underlying_col, window=rv_window)
            .pipe(compute_iv_rv_spread, iv_col=iv_col, rv_col="realized_vol")
            .pipe(compute_zscores, col_name="vol_spread", window=z_window)
        )

        # Filter for signals
        # We need to collect to iterate and create signal objects, or we can use polars selection
        # and then iterate the reduced set.

        # Long Vol Signal: Spread is LOW (IV < RV), Z-Score < -Threshold -> Expect IV to rise or RV to fall (?)
        # Wait, Spread = IV - RV.
        # If Spread is Negative (IV < RV), IV is "cheap". Buy Vol. (Long)
        # If Spread is Positive (IV > RV), IV is "expensive". Sell Vol. (Short)

        # Z-Score logic:
        # Z < -2.0: Spread is abnormally low. IV is cheap. -> LONG VOL
        # Z > +2.0: Spread is abnormally high. IV is expensive. -> SHORT VOL

        signals_df = (
            df_feats.filter(pl.col("vol_spread_zscore").abs() > threshold)
            .select(
                [
                    "timestamp",
                    "instrument_id",
                    "vol_spread_zscore",
                    "realized_vol",
                    iv_col,
                ]
            )
            .collect()
        )  # Materialize

        results = []
        for row in signals_df.iter_rows(named=True):
            z = row["vol_spread_zscore"]
            if z > threshold:
                direction = Direction.SHORT  # Sell expensive vol
            else:
                direction = Direction.LONG  # Buy cheap vol

            sig = Signal(
                timestamp=row["timestamp"],
                instrument_id=row["instrument_id"],
                signal_type=SignalType.VOL_ARBITRAGE,
                direction=direction,
                strength=abs(z),
                confidence=0.8,  # Placeholder model confidence
                metadata={"iv": row[iv_col], "rv": row["realized_vol"], "z_score": z},
            )
            results.append(sig)

        return results

    def detect_parity_violations(
        self,
        calls: pl.DataFrame,
        puts: pl.DataFrame,
        spots: pl.DataFrame,
        r: float = 0.05,
        cost_threshold: float = 0.5,
    ) -> List[Signal]:
        """
        Detects Put-Call Parity violations.
        C - P = S - K * exp(-rT)
        """
        # 1. Join Calls and Puts on (expiry, strike, timestamp)
        # 2. Join with Spot
        # 3. Calculate deviation

        # simplified join keys
        # Assuming DataFrames have: timestamp, expiry, strike, close (price)

        base_df = calls.join(puts, on=["timestamp", "expiry", "strike"], suffix="_put")
        # calls has 'close', puts has 'close_put'

        # Join Spot
        full_df = base_df.join(
            spots, on="timestamp", suffix="_spot"
        )  # Assuming spot has 'close' -> 'close_spot' implies spot df has 'close'

        # Calculate Time to Expiry T (years)
        # T = (expiry - timestamp) / (365 * 24 * 3600 * 1e9)
        # Assuming timestamps are ns

        full_df = full_df.with_columns(
            [
                (
                    (pl.col("expiry") - pl.col("timestamp")) / (365 * 24 * 3600 * 1e9)
                ).alias("T")
            ]
        )

        # Theoretical Diff: S - K * exp(-rT)
        # Actual Diff: C - P

        full_df = full_df.with_columns(
            [
                (pl.col("close") - pl.col("close_put")).alias("market_diff"),
                (
                    pl.col("close_spot") - pl.col("strike") * np.exp(-r * pl.col("T"))
                ).alias("theo_diff"),
            ]
        )

        full_df = full_df.with_columns(
            [(pl.col("market_diff") - pl.col("theo_diff")).alias("parity_error")]
        )

        # Filter
        violations = full_df.filter(
            pl.col("parity_error").abs() > cost_threshold
        ).collect()

        results = []
        for row in violations.iter_rows(named=True):
            err = row["parity_error"]
            # If Market Diff > Theo Diff -> (C - P) > (S - K) -> C is expensive or P is cheap
            # Box Arb: Sell C, Buy P, Buy S?
            # Direction here refers to the structure or the "Synthetic" vs "Spot"?
            # Let's say SHORT means "Short the Synthetic" (Sell C, Buy P) because it's expensive

            direction = Direction.SHORT if err > 0 else Direction.LONG

            sig = Signal(
                timestamp=row["timestamp"],
                instrument_id=f"SYNTH-{row['expiry']}-{row['strike']}",  # Synthetic ID
                signal_type=SignalType.PARITY_VIOLATION,
                direction=direction,
                strength=abs(err),
                confidence=0.95,
                metadata={"error": err},
            )
            results.append(sig)

        return results
