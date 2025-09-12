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

# Column name aliases for robust CSV parsing
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

def load_backtest_data(base_dir):
    """Load all backtest data from the specified directory"""
    base_path = Path(base_dir)
    data = {
        'run_id': base_path.name,
        'base_dir': str(base_path),
        'generated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': platform.python_version(),
        'script_version': '1.0.0'
    }
    
    # Load summary.csv
    summary_file = base_path / 'summary.csv'
    if summary_file.exists():
        summary_data = read_csv_data(summary_file)
        if summary_data:
            if HAS_PANDAS:
                row = summary_data.iloc[0] if len(summary_data) > 0 else {}
            else:
                row = summary_data[0] if summary_data else {}
            
            data['capital'] = {
                'initial': safe_float(row.get('initial_capital')),
                'final': safe_float(row.get('final_capital'))
            }
            data['performance'] = {
                'total_return': safe_float(row.get('total_return')),
                'annualized_return': safe_float(row.get('annualized_return')),
                'annualized_vol': safe_float(row.get('annualized_volatility')),
                'sharpe': safe_float(row.get('sharpe_ratio')),
                'max_drawdown': safe_float(row.get('max_drawdown'))
            }
            data['period'] = {
                'start': row.get('start_date', ''),
                'end': row.get('end_date', '')
            }
    else:
        data['capital'] = {'initial': None, 'final': None}
        data['performance'] = {
            'total_return': None, 'annualized_return': None,
            'annualized_vol': None, 'sharpe': None, 'max_drawdown': None
        }
        data['period'] = {'start': '', 'end': ''}
    
    # Load metrics.json
    metrics_file = base_path / 'metrics.json'
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                if 'trading' in metrics:
                    data['trading'] = metrics['trading']
                else:
                    # Extract trading metrics from the JSON structure
                    data['trading'] = {
                        'trade_count': safe_int(metrics.get('trade_count')),
                        'win_rate': safe_float(metrics.get('win_rate')),
                        'turnover': safe_float(metrics.get('turnover'))
                    }
        except Exception as e:
            print(f"Error loading metrics.json: {e}")
            data['trading'] = {'trade_count': None, 'win_rate': None, 'turnover': None}
    else:
        data['trading'] = {'trade_count': None, 'win_rate': None, 'turnover': None}
    
    # Load trades.csv
    trades_file = base_path / 'trades.csv'
    if trades_file.exists():
        trades_data = read_csv_data(trades_file)
        if trades_data:
            if HAS_PANDAS:
                # Calculate trade statistics
                trades_data['pnl'] = pd.to_numeric(trades_data.get('pnl', []), errors='coerce')
                trades_data['pnl_pct'] = pd.to_numeric(trades_data.get('pnl_pct', []), errors='coerce')
                
                data['trades_summary'] = {
                    'count': len(trades_data),
                    'avg_holding_days': safe_float(trades_data.get('holding_period_days', pd.Series()).mean()),
                    'avg_return_pct': safe_float(trades_data['pnl_pct'].mean() * 100),
                    'max_return_pct': safe_float(trades_data['pnl_pct'].max() * 100),
                    'min_return_pct': safe_float(trades_data['pnl_pct'].min() * 100)
                }
                
                # Top winners and losers
                top_winners = trades_data.nlargest(5, 'pnl').to_dict('records')
                top_losers = trades_data.nsmallest(5, 'pnl').to_dict('records')
            else:
                # Basic processing without pandas
                pnl_values = [safe_float(row.get('pnl', 0)) for row in trades_data]
                pnl_pct_values = [safe_float(row.get('pnl_pct', 0)) for row in trades_data]
                
                data['trades_summary'] = {
                    'count': len(trades_data),
                    'avg_holding_days': None,  # Would need more complex calculation
                    'avg_return_pct': sum(pnl_pct_values) / len(pnl_pct_values) * 100 if pnl_pct_values else None,
                    'max_return_pct': max(pnl_pct_values) * 100 if pnl_pct_values else None,
                    'min_return_pct': min(pnl_pct_values) * 100 if pnl_pct_values else None
                }
                
                # Sort trades by PnL for top winners/losers
                trades_with_pnl = [(row, safe_float(row.get('pnl', 0))) for row in trades_data]
                trades_with_pnl.sort(key=lambda x: x[1], reverse=True)
                
                # Convert string fields to proper types for template
                top_winners = []
                top_losers = []
                
                for row, _ in trades_with_pnl[:5]:
                    processed_row = dict(row)
                    processed_row['pnl'] = safe_float(row.get('pnl'))
                    processed_row['pnl_pct'] = safe_float(row.get('pnl_pct'))
                    top_winners.append(processed_row)
                
                for row, _ in trades_with_pnl[-5:]:
                    processed_row = dict(row)
                    processed_row['pnl'] = safe_float(row.get('pnl'))
                    processed_row['pnl_pct'] = safe_float(row.get('pnl_pct'))
                    top_losers.append(processed_row)
            
            data['top_winners'] = top_winners
            data['top_losers'] = top_losers
            data['top_trades_limit'] = 5
        else:
            data['trades_summary'] = None
            data['top_winners'] = []
            data['top_losers'] = []
    else:
        data['trades_summary'] = None
        data['top_winners'] = []
        data['top_losers'] = []
    
    # Load positions_end_of_day.csv
    positions_file = base_path / 'positions_end_of_day.csv'
    if positions_file.exists():
        positions_data = read_csv_data(positions_file)
        if HAS_PANDAS and hasattr(positions_data, 'to_dict'):
            data['positions'] = positions_data.head(30).to_dict('records')
        else:
            data['positions'] = positions_data[:30] if positions_data else []
    else:
        data['positions'] = []
    
    # Load factor_exposures.csv
    factor_file = base_path / 'factor_exposures.csv'
    if factor_file.exists():
        factor_data = read_csv_data(factor_file)
        if HAS_PANDAS and hasattr(factor_data, 'to_dict'):
            data['factor_exposures'] = factor_data.head(50).to_dict('records')
        else:
            data['factor_exposures'] = factor_data[:50] if factor_data else []
    else:
        data['factor_exposures'] = []
    
    # Load logs.txt
    logs_file = base_path / 'logs.txt'
    if logs_file.exists():
        try:
            with open(logs_file, 'r', encoding='utf-8') as f:
                logs_content = f.read()
                # Truncate logs if too long
                if len(logs_content) > 5000:
                    logs_content = logs_content[-5000:] + "\n\n... (truncated)"
                data['logs'] = logs_content
        except Exception as e:
            print(f"Error reading logs.txt: {e}")
            data['logs'] = None
    else:
        data['logs'] = None
    
    return data

def generate_charts(base_dir, data):
    """Generate equity curve and drawdown charts with robust column name handling"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping chart generation")
        return {'equity_curve_png': None, 'drawdown_png': None}
    
    base_path = Path(base_dir)
    charts_dir = base_path / 'charts'
    charts_dir.mkdir(exist_ok=True)
    
    equity_file = base_path / 'equity_curve.csv'
    chart_paths = {'equity_curve_png': None, 'drawdown_png': None}
    
    if not equity_file.exists():
        print(f"Equity curve file not found: {equity_file}")
        return chart_paths
    
    try:
        equity_data = read_csv_data(equity_file)
        
        if HAS_PANDAS:
            # Robust column name normalization for pandas DataFrame
            if not hasattr(equity_data, 'columns'):
                print("Error: Expected pandas DataFrame but got different type")
                return chart_paths
            
            original_columns = list(equity_data.columns)
            print(f"Original CSV columns: {original_columns}")
            
            # 1. Try to find and rename date column
            date_col_found = None
            for alias in DATE_ALIASES:
                if alias in equity_data.columns:
                    date_col_found = alias
                    break
            
            if date_col_found:
                equity_data = equity_data.rename(columns={date_col_found: 'date'})
                print(f"Found date column: '{date_col_found}' -> 'date'")
            else:
                # Auto-detect datetime column
                print("No date alias found, attempting auto-detection...")
                for col in equity_data.columns:
                    try:
                        test_dates = pd.to_datetime(equity_data[col], errors='coerce')
                        non_nan_ratio = test_dates.notna().sum() / len(test_dates)
                        if non_nan_ratio >= 0.5:  # At least 50% parseable as datetime
                            equity_data = equity_data.rename(columns={col: 'date'})
                            date_col_found = col
                            print(f"Auto-detected date column: '{col}' -> 'date' ({non_nan_ratio:.1%} valid dates)")
                            break
                    except Exception:
                        continue
                
                if not date_col_found:
                    print("Warning: No suitable date column found. Skipping chart generation.")
                    return chart_paths
            
            # 2. Try to find and rename equity/value column
            equity_col_found = None
            for alias in EQUITY_ALIASES:
                if alias in equity_data.columns:
                    equity_col_found = alias
                    break
            
            if equity_col_found:
                equity_data = equity_data.rename(columns={equity_col_found: 'equity'})
                print(f"Found equity column: '{equity_col_found}' -> 'equity'")
            else:
                print("Warning: No suitable equity/value column found. Skipping chart generation.")
                return chart_paths
            
            # 3. Clean and coerce data types
            equity_data['date'] = pd.to_datetime(equity_data['date'], errors='coerce')
            equity_data['equity'] = pd.to_numeric(equity_data['equity'], errors='coerce')
            
            # Drop rows with NaN in either date or equity
            before_count = len(equity_data)
            equity_data = equity_data.dropna(subset=['date', 'equity'])
            after_count = len(equity_data)
            
            if before_count != after_count:
                print(f"Dropped {before_count - after_count} rows with invalid date/equity values")
            
            if after_count < 2:
                print(f"Warning: Only {after_count} valid rows remaining after cleaning. Need at least 2 rows for charts.")
                return chart_paths
            
            # Sort by date
            equity_data = equity_data.sort_values('date')
            
            dates = equity_data['date']
            values = equity_data['equity']
            
            # Calculate drawdown
            peak = values.expanding().max()
            drawdown = (values - peak) / peak
            
        else:
            # Fallback for non-pandas case
            if not equity_data:
                print("Error: No data loaded from CSV")
                return chart_paths
            
            # For basic CSV reading, look for date/equity keys in the first row
            sample_row = equity_data[0] if equity_data else {}
            if 'date' not in sample_row or 'equity' not in sample_row:
                print("Warning: Basic CSV fallback requires 'date' and 'equity' columns exactly. Skipping chart generation.")
                return chart_paths
            
            dates = [row.get('date', '') for row in equity_data]
            values = [safe_float(row.get('equity', 0)) for row in equity_data]
            
            # Simple drawdown calculation
            drawdown = []
            peak = 0
            for val in values:
                if val > peak:
                    peak = val
                drawdown.append((val - peak) / peak if peak > 0 else 0)
        
        # Wrap chart generation in try/except for non-fatal errors
        try:
            # Create equity curve chart
            plt.figure(figsize=(12, 6))
            plt.plot(dates, values, linewidth=2, color='blue')
            plt.title('资金曲线 (Equity Curve)', fontsize=14, pad=20)
            plt.xlabel('日期 (Date)')
            plt.ylabel('资金 (Equity)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            equity_chart_path = charts_dir / 'equity_curve.png'
            plt.savefig(equity_chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            chart_paths['equity_curve_png'] = 'charts/equity_curve.png'
            print(f"Generated equity curve chart: {equity_chart_path}")
            
        except Exception as e:
            print(f"Error generating equity curve chart: {e}")
        
        try:
            # Create drawdown chart
            plt.figure(figsize=(12, 6))
            plt.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
            plt.plot(dates, drawdown, linewidth=2, color='darkred')
            plt.title('回撤曲线 (Drawdown)', fontsize=14, pad=20)
            plt.xlabel('日期 (Date)')
            plt.ylabel('回撤 (Drawdown)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            drawdown_chart_path = charts_dir / 'drawdown.png'
            plt.savefig(drawdown_chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            chart_paths['drawdown_png'] = 'charts/drawdown.png'
            print(f"Generated drawdown chart: {drawdown_chart_path}")
            
        except Exception as e:
            print(f"Error generating drawdown chart: {e}")
        
    except Exception as e:
        print(f"Error generating charts: {e}")
    
    return chart_paths

def generate_markdown_report(data, template_path):
    """Generate Markdown report using Jinja2 template"""
    if jinja2:
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            template = jinja2.Template(template_content)
            return template.render(**data)
        except Exception as e:
            print(f"Error with Jinja2 template: {e}")
            return generate_basic_markdown_report(data)
    else:
        return generate_basic_markdown_report(data)

def generate_basic_markdown_report(data):
    """Generate basic Markdown report without Jinja2"""
    report = f"""# 回测报告（{data['run_id']}）

生成时间：{data['generated_at']} UTC  
回测区间：{data['period']['start']} → {data['period']['end']}  
初始资金：{data['capital']['initial'] or 'N/A'}  
期末资金：{data['capital']['final'] or 'N/A'}  
总收益率：{(data['performance']['total_return'] * 100) if data['performance']['total_return'] else 'N/A'} %  
年化收益：{(data['performance']['annualized_return'] * 100) if data['performance']['annualized_return'] else 'N/A'} %  
年化波动：{(data['performance']['annualized_vol'] * 100) if data['performance']['annualized_vol'] else 'N/A'} %  
Sharpe：{data['performance']['sharpe'] or 'N/A'}  
最大回撤：{(data['performance']['max_drawdown'] * 100) if data['performance']['max_drawdown'] else 'N/A'} %  

---

## 1. 核心指标概览

| 指标 | 值 |
|------|----|
| 总收益率 | {(data['performance']['total_return'] * 100) if data['performance']['total_return'] else 'N/A'} % |
| 年化收益 | {(data['performance']['annualized_return'] * 100) if data['performance']['annualized_return'] else 'N/A'} % |
| 年化波动 | {(data['performance']['annualized_vol'] * 100) if data['performance']['annualized_vol'] else 'N/A'} % |
| Sharpe | {data['performance']['sharpe'] or 'N/A'} |
| 最大回撤 | {(data['performance']['max_drawdown'] * 100) if data['performance']['max_drawdown'] else 'N/A'} % |
| 交易次数 | {data['trading']['trade_count'] or 'N/A'} |
| 胜率 | {(data['trading']['win_rate'] * 100) if data['trading']['win_rate'] else 'N/A'} % |
| 换手率 | {data['trading']['turnover'] or 'N/A'} |

---

## 2. 资金曲线与回撤

{'![Equity Curve](' + data['charts']['equity_curve_png'] + ')' if data['charts']['equity_curve_png'] else '（未生成资金曲线图）'}

{'![Drawdown Curve](' + data['charts']['drawdown_png'] + ')' if data['charts']['drawdown_png'] else ''}

---

## 7. 运行元信息

| 字段 | 值 |
|------|----|
| run_id | {data['run_id']} |
| 生成时间(UTC) | {data['generated_at']} |
| 脚本版本 | {data['script_version']} |
| Python 版本 | {data['python_version']} |
| 文件来源目录 | {data['base_dir']} |

---

（报告自动生成；若需 HTML 交互版，请查看同目录的 report.html）
"""
    return report

def convert_md_to_html(markdown_content, output_path):
    """Convert Markdown to HTML"""
    try:
        import markdown
        html_content = markdown.markdown(markdown_content, extensions=['tables'])
        
        # Wrap in basic HTML structure
        full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回测报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }}
        details {{ margin: 10px 0; }}
        summary {{ cursor: pointer; font-weight: bold; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        return True
    except ImportError:
        # Fallback: create basic HTML without markdown processing
        simple_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回测报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; white-space: pre-wrap; }}
    </style>
</head>
<body>
<pre>{markdown_content}</pre>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(simple_html)
        return True
    except Exception as e:
        print(f"Error converting to HTML: {e}")
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
    if not template_path.exists():
        print(f"Template file {template_path} does not exist")
        sys.exit(1)
    
    print(f"Loading backtest data from {input_dir}...")
    data = load_backtest_data(input_dir)
    
    print("Generating charts...")
    charts = generate_charts(input_dir, data)
    data['charts'] = charts
    
    print("Generating Markdown report...")
    markdown_content = generate_markdown_report(data, template_path)
    
    # Save Markdown report
    md_output = input_dir / 'report.md'
    with open(md_output, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print(f"Markdown report saved to {md_output}")
    
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