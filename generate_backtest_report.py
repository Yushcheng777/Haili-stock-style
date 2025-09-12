#!/usr/bin/env python3
"""
generate_backtest_report.py

Generates automated Markdown & HTML backtest reports from existing backtest output files.

Expected input files in backtest_results directory:
- summary.csv
- metrics.json
- equity_curve.csv
- trades.csv
- positions_end_of_day.csv
- factor_exposures.csv
- logs.txt (optional)

Outputs:
- backtest_results/report.md
- backtest_results/report.html
- backtest_results/charts/equity_curve.png
- backtest_results/charts/drawdown.png
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
import platform

# Try to import required libraries
try:
    import jinja2
except ImportError:
    print("Warning: jinja2 not available, using basic template substitution")
    jinja2 = None

# Try to import matplotlib for charts
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not available, skipping chart generation")
    HAS_MATPLOTLIB = False

# Check for pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("Warning: pandas not available, using basic CSV parsing")
    HAS_PANDAS = False

# Aliases for flexible column handling
DATE_ALIASES = ['date', 'Date', 'trade_date', 'TradeDate', 'datetime', 'Datetime', 'timestamp', 'Timestamp', 'time', 'Time']
EQUITY_ALIASES = ['equity', 'Equity', 'portfolio_value', 'PortfolioValue', 'total_value', 'TotalValue', 'nav', 'NAV', 'value', 'Value']

# Basic CSV reader when pandas is not available
def read_csv_basic(filepath):
    """Basic CSV reader that returns list of dictionaries"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return data

def read_csv_data(filepath):
    """Read CSV data using pandas if available, otherwise basic reader"""
    if HAS_PANDAS:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {filepath} with pandas: {e}")
            return read_csv_basic(filepath)
    else:
        return read_csv_basic(filepath)

def safe_float(value, default=None):
    """Safely convert value to float"""
    if value is None or value == '' or value == 'N/A':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=None):
    """Safely convert value to int"""
    if value is None or value == '' or value == 'N/A':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def find_column_by_aliases(columns, aliases):
    """Find column name by checking against aliases"""
    columns_lower = [col.lower() for col in columns]
    for alias in aliases:
        alias_lower = alias.lower()
        if alias_lower in columns_lower:
            # Return the original column name (not lowercased)
            return columns[columns_lower.index(alias_lower)]
    return None

def _normalize_equity_dataframe(df):
    """Normalize DataFrame to have columns: date (datetime64), equity (float)"""
    if not HAS_PANDAS:
        return None
        
    if df is None:
        return None
        
    if hasattr(df, 'empty') and df.empty:
        return None
        
    if not hasattr(df, 'columns'):
        return None
    
    columns = list(df.columns)
    
    # Find date column
    date_col = find_column_by_aliases(columns, DATE_ALIASES)
    
    # Find equity/value column
    equity_col = find_column_by_aliases(columns, EQUITY_ALIASES)
    
    if date_col is None:
        print("Warning: Could not identify date column in equity_curve.csv; skipping chart generation.")
        return None
    if equity_col is None:
        print("Warning: Could not identify equity/value column in equity_curve.csv; skipping chart generation.")
        return None

    # Rename to canonical names
    if date_col != 'date':
        df = df.rename(columns={date_col: 'date'})
    if equity_col != 'equity':
        df = df.rename(columns={equity_col: 'equity'})

    # Coerce types and clean
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['equity'] = pd.to_numeric(df['equity'], errors='coerce')
    df = df.dropna(subset=['date', 'equity']).sort_values('date')

    if len(df) < 2:
        print("Warning: Not enough valid data points in equity curve")
        return None
        
    return df

def load_backtest_data(base_dir):
    """Load all backtest data files"""
    data = {
        'summary': None,
        'metrics': None,
        'equity_curve': None,
        'trades': None,
        'positions': None,
        'factor_exposures': None,
        'logs': None,
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'base_dir': str(base_dir)
    }
    
    # Load summary.csv
    summary_path = base_dir / 'summary.csv'
    if summary_path.exists():
        try:
            data['summary'] = read_csv_data(summary_path)
            print(f"Loaded summary data from {summary_path}")
        except Exception as e:
            print(f"Warning: Failed to load summary.csv: {e}")
    
    # Load metrics.json
    metrics_path = base_dir / 'metrics.json'
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                data['metrics'] = json.load(f)
            print(f"Loaded metrics from {metrics_path}")
        except Exception as e:
            print(f"Warning: Failed to load metrics.json: {e}")
    
    # Load equity_curve.csv
    equity_path = base_dir / 'equity_curve.csv'
    if equity_path.exists():
        try:
            data['equity_curve'] = read_csv_data(equity_path)
            print(f"Loaded equity curve from {equity_path}")
        except Exception as e:
            print(f"Warning: Failed to load equity_curve.csv: {e}")
    
    # Load trades.csv
    trades_path = base_dir / 'trades.csv'
    if trades_path.exists():
        try:
            data['trades'] = read_csv_data(trades_path)
            print(f"Loaded trades from {trades_path}")
        except Exception as e:
            print(f"Warning: Failed to load trades.csv: {e}")
    
    # Load positions_end_of_day.csv
    positions_path = base_dir / 'positions_end_of_day.csv'
    if positions_path.exists():
        try:
            data['positions'] = read_csv_data(positions_path)
            print(f"Loaded positions from {positions_path}")
        except Exception as e:
            print(f"Warning: Failed to load positions_end_of_day.csv: {e}")
    
    # Load factor_exposures.csv
    exposures_path = base_dir / 'factor_exposures.csv'
    if exposures_path.exists():
        try:
            data['factor_exposures'] = read_csv_data(exposures_path)
            print(f"Loaded factor exposures from {exposures_path}")
        except Exception as e:
            print(f"Warning: Failed to load factor_exposures.csv: {e}")
    
    # Load logs.txt
    logs_path = base_dir / 'logs.txt'
    if logs_path.exists():
        try:
            with open(logs_path, 'r') as f:
                data['logs'] = f.read()
            print(f"Loaded logs from {logs_path}")
        except Exception as e:
            print(f"Warning: Failed to load logs.txt: {e}")
    
    return data

def generate_charts(base_dir, data):
    """Generate equity curve and drawdown charts"""
    chart_paths = {}
    
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping chart generation")
        return chart_paths
    
    equity_curve = data.get('equity_curve')
    if equity_curve is None or (HAS_PANDAS and hasattr(equity_curve, 'empty') and equity_curve.empty) or (not HAS_PANDAS and len(equity_curve) == 0):
        print("No equity curve data available, skipping chart generation")
        return chart_paths
    
    try:
        # Create charts directory
        charts_dir = base_dir / 'charts'
        charts_dir.mkdir(exist_ok=True)
        
        # Normalize equity data
        df = _normalize_equity_dataframe(data['equity_curve'])
        if df is None:
            print("Could not normalize equity curve data, skipping chart generation")
            return chart_paths
        
        # Generate equity curve chart
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df['date'], df['equity'], linewidth=2, color='blue')
            plt.title('Portfolio Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            equity_chart_path = charts_dir / 'equity_curve.png'
            plt.savefig(equity_chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            chart_paths['equity_curve'] = str(equity_chart_path)
            print(f"Generated equity curve chart: {equity_chart_path}")
        except Exception as e:
            print(f"Warning: Failed to generate equity curve chart: {e}")
        
        # Generate drawdown chart
        try:
            # Calculate drawdown
            df['cummax'] = df['equity'].cummax()
            df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax'] * 100
            
            plt.figure(figsize=(12, 4))
            plt.fill_between(df['date'], df['drawdown'], 0, color='red', alpha=0.3)
            plt.plot(df['date'], df['drawdown'], color='red', linewidth=1)
            plt.title('Portfolio Drawdown')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            drawdown_chart_path = charts_dir / 'drawdown.png'
            plt.savefig(drawdown_chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            chart_paths['drawdown'] = str(drawdown_chart_path)
            print(f"Generated drawdown chart: {drawdown_chart_path}")
        except Exception as e:
            print(f"Warning: Failed to generate drawdown chart: {e}")
            
    except Exception as e:
        print(f"Error generating charts: {e}")

    return chart_paths

def generate_markdown_report(data, template_path):
    """Generate Markdown report using template if available, otherwise basic report"""
    
    # Try to use Jinja2 template if available
    if jinja2 and template_path.exists():
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            template = jinja2.Template(template_content)
            return template.render(**data)
        except Exception as e:
            print(f"Warning: Template rendering failed ({e}), falling back to basic report")
    
    # Fallback to basic report generation
    report = []
    report.append("# Backtest Report")
    report.append("")
    report.append(f"Generated on: {data.get('generation_time', 'Unknown')}")
    report.append("")
    
    # Summary section
    if data.get('summary'):
        report.append("## Summary")
        if HAS_PANDAS and hasattr(data['summary'], 'to_string'):
            report.append("```")
            report.append(data['summary'].to_string(index=False))
            report.append("```")
        else:
            report.append("Summary data available but could not format")
        report.append("")
    
    # Metrics section
    if data.get('metrics'):
        report.append("## Key Metrics")
        for key, value in data['metrics'].items():
            report.append(f"- **{key}**: {value}")
        report.append("")
    
    # Charts section
    if data.get('charts'):
        report.append("## Charts")
        for chart_name, chart_path in data['charts'].items():
            if chart_path:
                report.append(f"### {chart_name.replace('_', ' ').title()}")
                report.append(f"![{chart_name}]({os.path.relpath(chart_path, data['base_dir'])})")
                report.append("")
    
    # Trades section
    if data.get('trades'):
        report.append("## Trade Summary")
        if HAS_PANDAS and hasattr(data['trades'], 'shape'):
            report.append(f"Total trades: {len(data['trades'])}")
        else:
            report.append(f"Total trades: {len(data['trades']) if data['trades'] else 0}")
        report.append("")
    
    # Logs section
    if data.get('logs'):
        report.append("## Logs")
        report.append("```")
        # Limit logs to last 50 lines to avoid huge reports
        log_lines = data['logs'].split('\n')[-50:]
        report.append('\n'.join(log_lines))
        report.append("```")
        report.append("")
    
    return '\n'.join(report)

def convert_md_to_html(markdown_content, output_path):
    """Convert Markdown to HTML"""
    try:
        import markdown
        html = markdown.markdown(markdown_content)
        
        # Wrap in basic HTML structure
        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        return True
    except ImportError:
        print("Warning: markdown library not available, skipping HTML conversion")
        return False
    except Exception as e:
        print(f"Warning: HTML conversion failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate backtest reports')
    parser.add_argument('--input-dir', default='backtest_results',
                       help='Directory containing backtest results')
    parser.add_argument('--template', default='report_template.md.jinja2',
                       help='Path to Jinja2 template file')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist")
        sys.exit(1)

    template_path = Path(args.template)

    print(f"Loading backtest data from {input_dir}...")
    data = load_backtest_data(input_dir)

    print("Generating charts...")
    charts = generate_charts(input_dir, data)
    data['charts'] = charts

    print("Generating Markdown report...")
    # If template exists and jinja2 available, use it; otherwise fall back to basic
    markdown_content = generate_markdown_report(data, template_path)

    # Save Markdown report
    md_output = input_dir / 'report.md'
    try:
        with open(md_output, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Markdown report saved to {md_output}")
    except Exception as e:
        print(f"Error saving Markdown report: {e}")

    # Convert to HTML
    print("Converting to HTML...")
    html_output = input_dir / 'report.html'
    if convert_md_to_html(markdown_content, html_output):
        print(f"HTML report saved to {html_output}")
    else:
        print("HTML conversion failed")

    print("Report generation complete!")

if __name__ == '__main__':
    main()