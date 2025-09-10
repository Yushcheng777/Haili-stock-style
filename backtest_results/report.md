# Backtest Performance Report

*Generated on: 2025-09-10T10:23:06.167938*

## Executive Summary

This report provides a comprehensive analysis of the backtesting performance, including key risk-adjusted metrics and visual charts.


## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Sharpe Ratio** | 9.135730600237308 | Risk-adjusted return relative to volatility |
| **Sortino Ratio** | 12.71610006145079 | Risk-adjusted return relative to downside volatility |
| **Annualized Return** | 0.38707733537302214 | Compound annual growth rate |
| **Maximum Drawdown** | -0.0024875621890547706 | Largest peak-to-trough decline |

### Interpretation Guide

- **Sharpe Ratio**: 
  - \> 2.0: Excellent
  - 1.0 - 2.0: Good  
  - 0.5 - 1.0: Acceptable
  - < 0.5: Poor

- **Sortino Ratio**: 
  - Higher values indicate better risk-adjusted returns
  - Focuses only on downside risk (negative volatility)

- **Maximum Drawdown**: 
  - Lower absolute values are better
  - Represents the worst-case scenario during the backtest period





## Technical Details


### Configuration

- **Rolling Window**: 60 days
- **Risk-Free Rate**: 0.0 annual
- **Trading Days per Year**: 252




### Data Summary

- **Total Data Points**: 90

- **Return Periods**: 89
- **Analysis Period**: Approximately 89 days



## Methodology

### Metrics Calculation

1. **Daily Returns**: Calculated as percentage change between consecutive periods
2. **Sharpe Ratio**: (Mean Excess Return / Standard Deviation) × √252
3. **Sortino Ratio**: Annualized Return / Downside Deviation × √252
4. **Annualized Return**: (End Value / Start Value)^(252/periods) - 1
5. **Maximum Drawdown**: Minimum of (Current Value / Running Maximum - 1)

### Risk-Free Rate

The risk-free rate is converted from annual to daily using: (1 + annual_rate)^(1/252) - 1

### Rolling Analysis

Rolling metrics use a 60-day window to assess performance consistency over time.

---

*This report was generated automatically using the Haili Stock Style backtesting framework.*