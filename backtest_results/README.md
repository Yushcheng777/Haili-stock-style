# Backtest Results Directory

This directory contains the output files from the automated backtest report generation system.

## Files Structure

- `equity_curve.csv` - Input file containing the equity curve data with columns:
  - First column: date/time
  - Value column: one of `nav`, `net_value`, `equity`, `value`, or `close`

- `metrics.json` - Generated performance metrics including:
  - Sharpe ratio
  - Sortino ratio
  - Annualized return
  - Maximum drawdown
  - Configuration parameters

- `report.md` - Generated markdown report with performance analysis

- `report.html` - Generated HTML report (optional, if dependencies available)

- `charts/` - Directory containing generated charts:
  - `rolling_sharpe.png` - Rolling Sharpe ratio visualization

## Usage

The `generate_backtest_report.py` script automatically processes the equity curve data and generates all output files. It handles missing dependencies gracefully and provides meaningful fallbacks.

## Configuration

Use environment variables to customize the analysis:

- `ROLLING_WINDOW_DAYS` - Rolling window for Sharpe calculation (default: 60, minimum: 30)
- `RISK_FREE_RATE` - Annual risk-free rate (default: 0.0)

Example:
```bash
ROLLING_WINDOW_DAYS=45 RISK_FREE_RATE=0.02 python generate_backtest_report.py
```