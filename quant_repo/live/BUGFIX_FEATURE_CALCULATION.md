# Feature Calculation Bug Fix — Symbol Isolation

## Problem
Both NIFTY and BANKNIFTY prices were being appended to a single `FeatureEngine` price buffer. Since the price ranges differ significantly (~22,000 vs ~51,000), this caused:
- Artificial price jumps when switching between symbols
- Inflated RV20 (Realized Volatility 20-period) calculations
- Regime detection always showing HIGH volatility
- Strategy never entering trades

## Root Cause
In `PaperTrader.__init__` and `_on_tick`, all ticks (regardless of symbol) were routed to a single shared feature engine:

```python
self.feature_engine = FeatureEngine(maxlen=500)  # SHARED!
# ... later in _on_tick:
self.feature_engine.update(price, timestamp)  # ALL symbols mixed
```

## Solution
Created **separate FeatureEngine instances per symbol** to maintain independent price series:

### Changes in `paper_trader.py`

#### 1. **__init__** — Create per-symbol engines
```python
# OLD: Single shared engine
self.feature_engine = FeatureEngine(maxlen=500)
self._feature_ready_logged = False

# NEW: Per-symbol engines
self.feature_engines: Dict[str, FeatureEngine] = {
    symbol: FeatureEngine(maxlen=500)
    for symbol in self.strategy.config.symbols
}
self._feature_ready_logged: Dict[str, bool] = {
    symbol: False for symbol in self.strategy.config.symbols
}
```

#### 2. **_on_tick** — Symbol-aware routing
- Extract `symbol` from `Tick` object
- Route price ONLY to `self.feature_engines[symbol]`
- Track ready state per symbol
- Pass `symbol` parameter to strategy's `on_tick()`

```python
# OLD: No symbol awareness
self.feature_engine.update(price, timestamp)
self.strategy.on_tick(price=price, features=features, timestamp=timestamp)

# NEW: Symbol-specific routing
feature_engine = self.feature_engines[symbol]
feature_engine.update(price, timestamp)
self.strategy.on_tick(symbol=symbol, price=price, features=features, timestamp=timestamp)
```

#### 3. **_check_entries** — Per-symbol logging
```python
# Now computes features using symbol-specific engine:
features = self.feature_engines[symbol].compute_features()
self.logger.info(f"[FEATURE] {symbol} Buffer={len(self.feature_engines[symbol].prices)} RV20={features['rv20']:.4f}")
```

#### 4. **StrategyWrapper.on_tick** — Accept symbol parameter
```python
# OLD: Ignored which symbol the tick was for
def on_tick(self, price: float, features: Dict, timestamp: datetime):

# NEW: Explicitly passes symbol through to strategy
def on_tick(self, symbol: str, price: float, features: Dict, timestamp: datetime):
```

## Verification
- ✅ NIFTY prices no longer contaminate BANKNIFTY volatility calculations
- ✅ Each symbol has independent RV20 computation
- ✅ Regime detection per symbol is now reliable
- ✅ Strategy receives symbol context with each tick/feature update
- ✅ Logging clearly shows which symbol's data is being processed

## Impact
- **Before**: RV20 inflated, always HIGH regime → no trades
- **After**: Accurate per-symbol volatility → regime detection works → strategy can enter trades

## Files Modified
- `/Users/Balu/SPS/quant_repo/live/paper_trader.py` (PaperTrader + StrategyWrapper)

## Notes
- FeatureEngine itself did not need changes — issue was at the orchestration layer
- Strategy's own `_price_buffers` continue to work per-symbol (already implemented correctly)
- No changes needed to market_feed.py — it already identifies symbol in Tick object
