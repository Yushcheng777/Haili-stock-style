# Backtest Report Generation

This directory contains the automated report generation system for backtest results.

## Overview

The report generation system automatically creates comprehensive Markdown and HTML reports from backtest output files. It integrates seamlessly with the existing workflow infrastructure to provide detailed analysis of backtest performance.

## Files

### Core Components
- `generate_backtest_report.py` - Main report generation script
- `report_template.md.jinja2` - Jinja2 template for Markdown reports
- Updated workflow files with report generation steps

### Dependencies
- `jinja2` - Template engine for report generation
- `markdown` - HTML conversion from Markdown
- `matplotlib` - Chart generation (optional, graceful fallback)
- `pandas` - Data processing (optional, graceful fallback)

## Input Files

The report generator expects the following files in the input directory (typically `backtest_results/` or `results/{timestamp}/`):

### Required Files
- `summary.csv` - Overall backtest performance metrics
- `metrics.json` - Trading statistics and additional metrics

### Optional Files
- `equity_curve.csv` - Time series of portfolio equity for chart generation
- `trades.csv` - Individual trade details
- `positions_end_of_day.csv` - Daily position snapshots
- `factor_exposures.csv` - Factor exposure analysis
- `logs.txt` - Backtest execution logs

## Output Files

The generator creates the following output files:

- `report.md` - Comprehensive Markdown report
- `report.html` - Styled HTML version of the report
- `charts/equity_curve.png` - Equity curve visualization (if matplotlib available)
- `charts/drawdown.png` - Drawdown analysis chart (if matplotlib available)

## Usage

### Manual Execution
```bash
python generate_backtest_report.py --input-dir backtest_results
```

### Command Line Options
- `--input-dir`: Directory containing backtest result files (default: `backtest_results`)
- `--template`: Path to Jinja2 template file (default: `report_template.md.jinja2`)

### Workflow Integration

The report generation is automatically integrated into backtest workflows:

1. **generate-report.yml** - Artifacts-only report generation (no Git operations)
2. **scheduled-backtest.yml** - Daily scheduled backtests
3. **daily_haili_backtest.yml** - Haili strategy backtests
4. **auto_backtest.yml** - Automated strategy backtests

Note: The `generate-report.yml` workflow specifically operates in artifacts-only mode for clean report generation without repository modifications.

## Report Content

The generated reports include:

### 1. Executive Summary
- Run metadata (timestamp, period, initial/final capital)
- Key performance metrics (returns, Sharpe ratio, max drawdown)

### 2. Performance Overview
- Detailed metrics table
- Performance statistics

### 3. Equity Curve & Drawdown Analysis
- Visual charts (when matplotlib available)
- Drawdown periods and recovery analysis

### 4. Trading Statistics
- Trade summary statistics
- Top winning and losing trades
- Trade distribution analysis

### 5. Factor/Sector Exposures
- Factor exposure analysis (if available)
- Sector allocation details

### 6. Positions Analysis
- End-of-day position snapshots
- Position sizing and allocation

### 7. Execution Logs
- Backtest execution details
- Error logs and warnings

### 8. Metadata
- Runtime information
- Script versions and environment details

## Error Handling

The system includes robust error handling:

- **Missing Dependencies**: Graceful fallback when matplotlib or pandas unavailable
- **Missing Files**: Reports partial data when some input files missing
- **Data Parsing Errors**: Continues processing with available data
- **Template Errors**: Falls back to basic report generation

## Template Customization

The report template (`report_template.md.jinja2`) can be customized:

- Modify sections, add new metrics
- Change formatting and styling
- Add or remove data visualizations
- Translate to other languages

## Integration with Workflows

### Automatic Execution
Reports are automatically generated after each backtest run in the CI/CD workflows.

### Artifact Upload
Reports are included in GitHub Actions artifacts for download and review.

### Artifacts-Only Mode
The `generate-report.yml` workflow operates in artifacts-only mode - it generates reports and uploads them as GitHub Actions artifacts without committing changes to the repository or creating pull requests. This provides a clean separation between report generation and repository modifications.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Install required packages: `pip install jinja2 markdown matplotlib pandas`
   - System works with graceful fallbacks if packages unavailable

2. **Template Errors**
   - Check Jinja2 syntax in template file
   - Verify data structure matches template expectations

3. **Chart Generation Fails**
   - Ensure matplotlib is installed
   - Check equity_curve.csv format and data

4. **Empty Reports**
   - Verify input directory contains expected files
   - Check file formats match expected structure

### Debug Mode
Run with Python in verbose mode to see detailed error messages:
```bash
python -v generate_backtest_report.py --input-dir your_directory
```

## Future Enhancements

Potential improvements:
- Interactive charts with plotly
- PDF report generation
- Email notification integration
- Performance benchmarking
- Risk attribution analysis
- Multi-strategy comparison reports