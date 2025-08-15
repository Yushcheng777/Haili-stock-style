# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Implement conservative RSI divergence detection based on local extrema in `haili_backtest.py`.
  - Detects regular bullish and bearish divergences using the two most recent local extrema within the recent window.
  - Conservative thresholds: minimum RSI diff = 3.0, minimum price change = 0.5%.
  - Retains existing RSI validity (non-NaN fraction) checks.
- Add unit tests for bullish and bearish divergence detection and for insufficient RSI validity.

### Example
See README.md for a short usage example.