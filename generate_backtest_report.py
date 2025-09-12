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

def load_backtest_data(base_dir):
    """Load all backtest data from the specified directory"""
    base_path = Path(base_dir)
    data = {
        'run_id': base_path.name,
        'base_dir': str(base_path),
        'generated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': platform.python_version(),
        'script_version': '1.1.0'
    }

    # Load summary.csv
    summary_file = base_path / 'summary.csv'
    if summary_file.exists():
        summary_data = read_csv_data(summary_file)
        if summary_data is not None:
            if HAS_PANDAS and hasattr(summary_data, 'iloc'):
                row = summary_data.iloc[0] if len(summary_data) > 0 else {}
                getter = row.get if hasattr(row, 'get') else row.__getitem__
            else:
                row = summary_data[0] if summary_data else {}
                getter = row.get if isinstance(row, dict) else (lambda k, d=None: d)

            data['capital'] = {
                'initial': safe_float(getter('initial_capital')),
                'final': safe_float(getter('final_capital'))
            }
            data['performance'] = {
                'total_return': safe_float(getter('total_return')),
                'annualized_return': safe_float(getter('annualized_return')),
                'annualized_vol': safe_float(getter('annualized_volatility')),
                'sharpe': safe_float(getter('sharpe_ratio')),
                'max_drawdown': safe_float(getter('max_drawdown'))
            }
            data['period'] = {
                'start': getter('start_date', ''),
                'end': getter('end_date', '')
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
            if HAS_PANDAS and hasattr(trades_data, 'to_dict'):
                trades_data = trades_data.copy()
                trades_data['pnl'] = pd.to_numeric(trades_data.get('pnl', []), errors='coerce')
                trades_data['pnl_pct'] = pd.to_numeric(trades_data.get('pnl_pct', []), errors='coerce')

                data['trades_summary'] = {
                    'count': int(len(trades_data)),
                    'avg_holding_days': safe_float(trades_data.get('holding_period_days', pd.Series(dtype=float)).mean()),
                    'avg_return_pct': safe_float((trades_data['pnl_pct'].mean() * 100) if len(trades_data['pnl_pct']) else None),
                    'max_return_pct': safe_float((trades_data['pnl_pct'].max() * 100) if len(trades_data['pnl_pct']) else None),
                    'min_return_pct': safe_float((trades_data['pnl_pct'].min() * 100) if len(trades_data['pnl_pct']) else None)
                }

                top_winners = trades_data.nlargest(5, 'pnl', keep='all').head(5).to_dict('records')
                top_losers = trades_data.nsmallest(5, 'pnl', keep='all').head(5).to_dict('records')
            else:
                pnl_values = [safe_float(row.get('pnl', 0)) for row in trades_data]
                pnl_pct_values = [safe_float(row.get('pnl_pct', 0)) for row in trades_data]

                data['trades_summary'] = {
                    'count': len(trades_data),
                    'avg_holding_days': None,
                    'avg_return_pct': (sum(v for v in pnl_pct_values if v is not None) / len([v for v in pnl_pct_values if v is not None]) * 100) if any(v is not None for v in pnl_pct_values) else None,
                    'max_return_pct': (max(v for v in pnl_pct_values if v is not None) * 100) if any(v is not None for v in pnl_pct_values) else None,
                    'min_return_pct': (min(v for v in pnl_pct_values if v is not None) * 100) if any(v is not None for v in pnl_pct_values) else None
                }

                trades_with_pnl = [(dict(row), safe_float(row.get('pnl', 0))) for row in trades_data]
                trades_with_pnl.sort(key=lambda x: (x[1] is None, x[1]), reverse=True)

                top_winners = []
                top_losers = []
                for row, _ in trades_with_pnl[:5]:
                    row['pnl'] = safe_float(row.get('pnl'))
                    row['pnl_pct'] = safe_float(row.get('pnl_pct'))
                    top_winners.append(row)
                for row, _ in trades_with_pnl[-5:]:
                    row['pnl'] = safe_float(row.get('pnl'))
                    row['pnl_pct'] = safe_float(row.get('pnl_pct'))
                    top_losers.append(row)

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

def _normalize_equity_dataframe(df):
    """Normalize DataFrame to have columns: date (datetime64), equity (float)"""
    if not HAS_PANDAS or df is None or not hasattr(df, 'columns'):
        return None

    df = df.copy()
    cols = list(df.columns)

    # 1) Try alias matching
    date_col = next((c for c in DATE_ALIASES if c in cols), None)
    equity_col = next((c for c in EQUITY_ALIASES if c in cols), None)

    # 2) Auto-detect date if not found: choose the first column with >=50% parseable datetimes
    if date_col is None:
        best = None
        best_ratio = 0.0
        for c in cols:
            try:
                parsed = pd.to_datetime(df[c], errors='coerce', utc=False)
                ratio = parsed.notna().mean()
                if ratio >= 0.5 and ratio > best_ratio:
                    best = c
                    best_ratio = ratio
            except Exception:
                continue
        date_col = best

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
        print("Warning: Not enough rows after cleaning equity data; skipping chart generation.")
        return None

    return df[['date', 'equity']]

def generate_charts(base_dir, data):
    """Generate equity curve and drawdown charts"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping chart generation")
        return {'equity_curve_png': None, 'drawdown_png': None}

    base_path = Path(base_dir)
    charts_dir = base_path / 'charts'
    charts_dir.mkdir(exist_ok=True)

    equity_file = base_path / 'equity_curve.csv'
    chart_paths = {'equity_curve_png': None, 'drawdown_png': None}

    if not equity_file.exists():
        print(f"No equity curve file found at {equity_file}; skipping charts.")
        return chart_paths

    try:
        equity_data = read_csv_data(equity_file)

        if HAS_PANDAS and hasattr(equity_data, 'columns'):
            df = _normalize_equity_dataframe(equity_data)
            if df is None:
                return chart_paths

            dates = df['date']
            values = df['equity']

            # Drawdown
            rolling_peak = values.cummax()
            drawdown = (values - rolling_peak) / rolling_peak.replace(0, pd.NA)
            drawdown = drawdown.fillna(0.0)

        else:
            # Basic implementation without pandas using aliases if possible
            rows = equity_data or []
            # Try to pick alias keys present in rows
            keys = rows[0].keys() if rows else []
            date_key = next((k for k in DATE_ALIASES if k in keys), 'date')
            eq_key = next((k for k in EQUITY_ALIASES if k in keys), 'equity')

            dates = [r.get(date_key, '') for r in rows]
            values = [safe_float(r.get(eq_key, None)) for r in rows]
            values = [v for v in values if v is not None]

            drawdown = []
            peak = 0.0
            for v in values:
                if v > peak:
                    peak = v
                drawdown.append((v - peak) / peak if peak else 0.0)

        # Plot equity curve
        try:
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
        except Exception as e:
            print(f"Error generating equity curve chart: {e}")

        # Plot drawdown
        try:
            plt.figure(figsize=(12, 6))
            # fill_between may require numeric x; for pandas datetimes it works; otherwise it still draws
            plt.fill_between(dates, drawdown, 0, alpha=0.3, color='red', step=None)
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
        except Exception as e:
            print(f"Error generating drawdown chart: {e}")

    except Exception as e:
        print(f"Error generating charts: {e}")

    return chart_paths

def generate_markdown_report(data, template_path):
    """Generate Markdown report using Jinja2 template"""
    if jinja2 and template_path.exists():
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()

            template = jinja2.Template(template_content)
            return template.render(**data)
        except Exception as e:
            print(f"Error with Jinja2 template: {e}")
            return generate_basic_markdown_report(data)
    else:
        if not template_path.exists():
            print(f"Template not found at {template_path}, using basic Markdown output.")
        return generate_basic_markdown_report(data)

def generate_basic_markdown_report(data):
    """Generate basic Markdown report without Jinja2"""
    report = f"""# 回测报告（{data['run_id']}）

生成时间：{data['generated_at']} UTC  
回测区间：{data['period']['start']} → {data['period']['end']}  
初始资金：{data['capital']['initial'] or 'N/A'}  
期末资金：{data['capital']['final'] or 'N/A'}  
总收益率：{(data['performance']['total_return'] * 100) if data['performance']['total_return'] is not None else 'N/A'} %  
年化收益：{(data['performance']['annualized_return'] * 100) if data['performance']['annualized_return'] is not None else 'N/A'} %  
年化波动：{(data['performance']['annualized_vol'] * 100) if data['performance']['annualized_vol'] is not None else 'N/A'} %  
Sharpe：{data['performance']['sharpe'] if data['performance']['sharpe'] is not None else 'N/A'}  
最大回撤：{(data['performance']['max_drawdown'] * 100) if data['performance']['max_drawdown'] is not None else 'N/A'} %  

---

## 1. 核心指标概览

| 指标 | 值 |
|------|----|
| 总收益率 | {(data['performance']['total_return'] * 100) if data['performance']['total_return'] is not None else 'N/A'} % |
| 年化收益 | {(data['performance']['annualized_return'] * 100) if data['performance']['annualized_return'] is not None else 'N/A'} % |
| 年化波动 | {(data['performance']['annualized_vol'] * 100) if data['performance']['annualized_vol'] is not None else 'N/A'} % |
| Sharpe | {data['performance']['sharpe'] if data['performance']['sharpe'] is not None else 'N/A'} |
| 最大回撤 | {(data['performance']['max_drawdown'] * 100) if data['performance']['max_drawdown'] is not None else 'N/A'} % |
| 交易次数 | {data['trading']['trade_count'] if data['trading']['trade_count'] is not None else 'N/A'} |
| 胜率 | {(data['trading']['win_rate'] * 100) if data['trading']['win_rate'] is not None else 'N/A'} % |
| 换手率 | {data['trading']['turnover'] if data['trading']['turnover'] is not None else 'N/A'} |

---

## 2. 资金曲线与回撤

{'![Equity Curve](' + data['charts']['equity_curve_png'] + ')' if data['charts'].get('equity_curve_png') else '（未生成资金曲线图）'}

{'![Drawdown Curve](' + data['charts']['drawdown_png'] + ')' if data['charts'].get('drawdown_png') else ''}

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
