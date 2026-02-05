from typing import Dict, Optional, List
from decimal import Decimal
from dataclasses import dataclass

from nautilus_trader.model.identifiers import InstrumentId, Venue, Symbol
from nautilus_trader.model.instruments import Instrument, CurrencyPair, Equity
from nautilus_trader.model.objects import Currency, Price, Quantity


@dataclass
class InstrumentSpec:
    symbol: str
    venue: str
    asset_class: str  # "FX", "EQUITY", "FUTURE"
    price_precision: int
    size_precision: int
    price_increment: float
    lot_size: float
    multiplier: float = 1.0
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None


class ContractMetadataRegistry:
    """
    Central registry for instrument definitions.
    """

    def __init__(self):
        self._specs: Dict[str, InstrumentSpec] = {}
        self._instruments: Dict[str, Instrument] = {}

    def register(self, spec: InstrumentSpec):
        """Register a new instrument specification."""
        instrument_id_str = f"{spec.symbol}.{spec.venue}"
        self._specs[instrument_id_str] = spec
        self._instruments[instrument_id_str] = self._create_nautilus_instrument(spec)

    def get_instrument(self, symbol: str, venue: str) -> Optional[Instrument]:
        """Retrieve a Nautilus Instrument object."""
        key = f"{symbol}.{venue}"
        return self._instruments.get(key)

    def get_spec(self, symbol: str, venue: str) -> Optional[InstrumentSpec]:
        """Retrieve the raw specification."""
        key = f"{symbol}.{venue}"
        return self._specs.get(key)

    def _create_nautilus_instrument(self, spec: InstrumentSpec) -> Instrument:
        instrument_id = InstrumentId.from_str(f"{spec.symbol}.{spec.venue}")

        if spec.asset_class == "FX":
            return CurrencyPair(
                instrument_id=instrument_id,
                raw_symbol=Symbol(spec.symbol),
                base_currency=Currency.from_str(spec.base_currency),
                quote_currency=Currency.from_str(spec.quote_currency),
                price_precision=spec.price_precision,
                size_precision=spec.size_precision,
                price_increment=Price.from_str(str(spec.price_increment)),
                size_increment=Quantity.from_int(1),
                lot_size=Quantity.from_int(int(spec.lot_size)),
                ts_event=0,
                ts_init=0,
            )
        elif spec.asset_class == "EQUITY":
            return Equity(
                instrument_id=instrument_id,
                raw_symbol=spec.symbol,
                currency=Currency.from_str(spec.quote_currency),
                price_precision=spec.price_precision,
                size_precision=spec.size_precision,
                price_increment=Price.from_str(str(spec.price_increment)),
                size_increment=Quantity.from_int(1),
                lot_size=Quantity.from_int(int(spec.lot_size)),
                ts_event=0,
                ts_init=0,
            )
        else:
            raise NotImplementedError(
                f"Asset class {spec.asset_class} not supported yet."
            )


# Global instance
registry = ContractMetadataRegistry()
