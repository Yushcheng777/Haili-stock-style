# Backtest Report Enhancement

This directory contains enhanced backtest reporting functionality with Sortino ratio calculation and Rolling Sharpe chart generation.

## New Files

### 1. `generate_backtest_report.py`
Enhanced backtest report generator that adds:
- **Sortino ratio calculation**: Risk-adjusted return metric that only considers downside volatility
- **Rolling Sharpe ratio charts**: Visualizes performance consistency over time
- **Environment variable configuration**: Flexible parameter adjustment
- **Robust error handling**: Graceful handling of missing data and edge cases

### 2. `templates/report_template.md.j2`
Jinja2 template for generating Markdown reports that includes:
- Core metrics summary table with Sortino ratio
- Rolling Sharpe chart section
- Detailed analysis sections
- Proper conditional rendering for missing charts

## Usage

### Basic Usage
```bash
python generate_backtest_report.py [backtest_results_directory]
```

### With Environment Variables
```bash
# Use 90-day rolling window and 2% risk-free rate
ROLLING_WINDOW_DAYS=90 RISK_FREE_RATE=0.02 python generate_backtest_report.py

# Use custom window size
ROLLING_WINDOW_DAYS=120 python generate_backtest_report.py backtest_results/
```

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ROLLING_WINDOW_DAYS` | int | 60 | Rolling window for Sharpe calculation (minimum 30) |
| `RISK_FREE_RATE` | float | 0.0 | Annual risk-free rate for excess return calculations |

## Input Requirements

### Required Files
- `equity_curve.csv`: Contains NAV/equity data over time
- `metrics.json`: Contains existing performance metrics (created if missing)

### Expected Data Format
```csv
date,nav
2023-01-01,10000
2023-01-02,10050
...
```

**Supported column names:**
- NAV columns: `nav`, `NAV`, `equity`, `cumulative_return`, `portfolio_value`
- Date columns: `date`, `Date`, `timestamp`, `index`

### Minimum Data Requirements
- At least 30 rows of equity curve data
- Valid numeric NAV values
- Parseable date column

## Output

### Updated `metrics.json`
```json
{
  "performance": {
    "sharpe": 1.45,
    "sortino": 2.31,
    "annualized_return": 0.1425,
    ...
  },
  "charts": {
    "rolling_sharpe_png": "charts/rolling_sharpe.png"
  }
}
```

### Generated Charts
- `charts/rolling_sharpe.png`: Rolling Sharpe ratio visualization (if matplotlib available)

## Key Features

### Sortino Ratio Calculation
- Measures risk-adjusted returns using only downside volatility
- Formula: `(Annualized Excess Return) / (Annualized Downside Deviation)`
- Returns `None` if no negative returns exist
- Uses configurable risk-free rate

### Rolling Sharpe Analysis
- Calculates Sharpe ratio over rolling time windows
- Default 60-day window (configurable)
- Generates matplotlib chart with reference lines
- Skips windows with zero standard deviation

### Error Handling
- **Insufficient data** (< 30 rows): Exits with error message
- **Missing files**: Graceful error with informative message
- **No negative returns**: Sets Sortino to None with explanation
- **Constant NAV**: Handles zero-variance cases properly
- **Missing dependencies**: Fallback implementations for core functionality

## Dependencies

### Required (Built-in)
- `os`, `sys`, `json`, `csv`, `math`, `pathlib`, `datetime`

### Optional (Enhanced functionality)
- `pandas`: For robust CSV parsing and data manipulation
- `numpy`: For efficient numerical calculations
- `matplotlib`: For chart generation
- `jinja2`: For template rendering

### Fallback Behavior
The script works without optional dependencies:
- Uses basic CSV parsing instead of pandas
- Uses built-in math functions instead of numpy  
- Skips chart generation without matplotlib
- All core calculations still work

## Testing

Run the included test scripts to verify functionality:

```bash
# Test basic functionality
python test_report.py

# Test template rendering
python test_template.py

# Test edge cases
python test_edge_cases.py
```

## Integration

This enhancement is designed to work with existing backtest workflows:

1. **Existing code** continues to work unchanged
2. **New metrics** are added to existing `metrics.json` structure
3. **Charts** are saved to standard `charts/` directory
4. **Template** follows existing report format conventions

## Example Output

### Core Metrics Table
| 指标 | 数值 |
|------|------|
| 年化收益率 | 14.25% |
| 夏普比率 | 1.450 |
| **Sortino比率** | **2.310** |
| 最大回撤 | -4.50% |

### Rolling Sharpe Chart
The generated chart shows how the Sharpe ratio evolves over time, helping identify periods of consistent vs. inconsistent performance.

## Technical Notes

### Calculation Details
- **Daily returns**: Calculated as `(NAV[t] - NAV[t-1]) / NAV[t-1]`
- **Annualization**: Uses 252 trading days per year
- **Risk-free rate**: Converted from annual to daily: `(1 + annual_rate)^(1/252) - 1`
- **Downside deviation**: Standard deviation of negative excess returns only

### Performance Considerations
- Memory efficient: Processes data incrementally
- No external API calls: Works with local data only
- Configurable precision: Handles floating-point edge cases