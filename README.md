# Haili-stock-style

Lightweight backtesting helpers and indicators for stock strategies.

## RSI divergence detection

This repository includes a conservative implementation of RSI divergence detection in `haili_backtest.py`.

Function signature:

```
detect_rsi_divergence(prices: pd.Series, rsi_values: pd.Series, window: int = 14, rsi_validity_threshold: float = RSI_VALIDITY_THRESHOLD) -> bool
```

It returns `True` when a regular bullish or bearish RSI divergence is detected within the most recent `window` bars, otherwise `False`.

Defaults (internal tuning constants):
- min_rsi_diff = 3.0
- min_price_diff_pct = 0.005 (0.5%)
- extrema_order = 1
- extrema_prominence = 0.0

### Quick usage example

```python
import pandas as pd
from haili_backtest import detect_rsi_divergence

# Example price and RSI series (index aligned)
prices = pd.Series([100, 98, 99, 95, 96, 94, 97, 96, 98, 97])
rsi_values = pd.Series([55.0, 48.0, 50.0, 35.0, 38.0, 33.0, 40.0, 39.0, 42.0, 41.0])

# Check for divergence in the last 14 bars (default window)
has_divergence = detect_rsi_divergence(prices, rsi_values)
print("RSI divergence detected:", has_divergence)
```

### Notes
- The implementation uses a local-extrema-based approach to detect two recent peaks/troughs and checks price vs RSI direction with conservative thresholds to reduce false positives.
- The function currently returns a boolean for backward compatibility. If you need richer output (e.g., indices, divergence type), consider requesting an enhancement or extending the helper function.

### Running tests
Run the test suite with:
```
pytest -q
```
