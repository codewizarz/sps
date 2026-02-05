from pathlib import Path
from typing import List
import polars as pl
from decimal import Decimal
import pandas as pd

from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.instruments import OptionContract
from nautilus_trader.model.enums import AssetClass, OptionKind
from nautilus_trader.model.objects import Currency, Price, Quantity
from nautilus_trader.core.datetime import dt_to_unix_nanos


class IndianOptionsFactory:
    """
    Scans a Hive-partitioned Parquet dataset to generate Nautilus OptionContract instruments.
    Expected structure: root / symbol=X / expiry=YYYY-MM-DD / data.parquet
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def discover_instruments(
        self, symbol: str, start_date: str, end_date: str
    ) -> List[OptionContract]:
        """
        Scans headers/partitions to find unique contracts active in the period.
        """
        q = pl.scan_parquet(self.dataset_path / f"symbol={symbol}" / "**" / "*.parquet")

        unique_contracts = q.select(["strike", "right", "expiry"]).unique().collect()

        instruments = []
        for row in unique_contracts.iter_rows(named=True):
            inst = self._create_option(
                symbol=symbol,
                strike=row["strike"],
                right=row["right"],
                expiry_str=row["expiry"],
            )
            instruments.append(inst)

        return instruments

    def _create_option(
        self, symbol: str, strike: float, right: str, expiry_str: str
    ) -> OptionContract:
        # ID Format: NIFTY-20240125-21000-CE.NSE

        if right.upper() in ["C", "CE", "CALL"]:
            opt_kind = OptionKind.CALL
            right_suffix = "CE"
        else:
            opt_kind = OptionKind.PUT
            right_suffix = "PE"

        expiry_dt = pd.Timestamp(expiry_str)
        expiry_fmt = expiry_dt.strftime("%Y%m%d")

        # Construct ID
        id_str = f"{symbol}-{expiry_fmt}-{int(strike)}-{right_suffix}.NSE"
        expiration_ns = dt_to_unix_nanos(expiry_dt)

        return OptionContract(
            instrument_id=InstrumentId.from_str(id_str),
            raw_symbol=Symbol(id_str),
            asset_class=AssetClass.INDEX,  # NIFTY is an Index
            currency=Currency.from_str("INR"),
            price_precision=2,
            price_increment=Price.from_str("0.05"),
            multiplier=Quantity.from_int(
                1
            ),  # Multiplier is usually 1 for Index Options in Nautilus perspective (lot size handles the size)
            lot_size=Quantity.from_int(50),
            underlying=symbol,
            option_kind=opt_kind,
            strike_price=Price.from_str(str(strike)),
            activation_ns=0,
            expiration_ns=expiration_ns,
            ts_event=0,
            ts_init=0,
        )
