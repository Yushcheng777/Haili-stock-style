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

def normalize_date_column(data):
    """Normalize date column names and format"""
    if HAS_PANDAS and hasattr(data, 'columns'):
        # Look for date column variations
        date_aliases = ['date', 'Date', 'DATE', 'datetime', 'Datetime', 'timestamp', 'time']
        date_col = None
        for alias in date_aliases:
            if alias in data.columns:
                date_col = alias
                break
        
        if date_col and date_col != 'date':
            data = data.rename(columns={date_col: 'date'})
        
        # Convert to datetime if not already
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            
    return data

def normalize_equity_column(data):
    """Detect and normalize equity/value column names"""
    equity_aliases = ['equity', 'Equity', 'portfolio_value', 'PortfolioValue', 
                     'total_value', 'TotalValue', 'nav', 'NAV', 'value', 'Value']
    
    if HAS_PANDAS and hasattr(data, 'columns'):
        equity_col = None
        for alias in equity_aliases:
            if alias in data.columns:
                equity_col = alias
                break
        
        if equity_col:
            if equity_col != 'equity':
                data = data.rename(columns={equity_col: 'equity'})
            
            # Convert to numeric and drop NaNs
            data['equity'] = pd.to_numeric(data['equity'], errors='coerce')
            data = data.dropna(subset=['equity'])
            
    else:
        # For basic CSV reader (list of dicts)
        if isinstance(data, list) and data:
            equity_col = None
            for alias in equity_aliases:
                if alias in data[0]:
                    equity_col = alias
                    break
            
            if equity_col and equity_col != 'equity':
                for row in data:
                    if equity_col in row:
                        row['equity'] = row.pop(equity_col)
            
            # Convert to numeric and filter out invalid rows
            filtered_data = []
            for row in data:
                if 'equity' in row:
                    equity_val = safe_float(row['equity'])
                    if equity_val is not None:
                        row['equity'] = equity_val
                        filtered_data.append(row)
            data = filtered_data
            
    return data

def compute_performance_metrics(equity_data):
    """Auto-compute performance metrics from equity curve data"""
    # Check if data is empty
    is_empty = False
    if HAS_PANDAS and hasattr(equity_data, 'empty'):
        is_empty = equity_data.empty
    elif isinstance(equity_data, list):
        is_empty = not equity_data
    else:
        is_empty = equity_data is None
        
    if is_empty:
        return {
            'initial_capital': None,
            'final_value': None,
            'total_return': None,
            'annualized_return': None,
            'annualized_vol': None,
            'sharpe': None,
            'max_drawdown': None
        }
    
    try:
        if HAS_PANDAS and hasattr(equity_data, 'equity'):
            equity_series = equity_data['equity'].dropna()
            if len(equity_series) < 2:
                return {
                    'initial_capital': equity_series.iloc[0] if len(equity_series) > 0 else None,
                    'final_value': equity_series.iloc[-1] if len(equity_series) > 0 else None,
                    'total_return': None,
                    'annualized_return': None,
                    'annualized_vol': None,
                    'sharpe': None,
                    'max_drawdown': None
                }
            
            initial_capital = equity_series.iloc[0]
            final_value = equity_series.iloc[-1]
            
            # Total return
            total_return = (final_value / initial_capital - 1) if initial_capital > 0 else 0
            
            # Daily returns
            daily_returns = equity_series.pct_change().dropna()
            
            # Annual return (CAGR)
            n_trading_days = len(equity_series) - 1
            annualized_return = None
            if n_trading_days >= 2 and initial_capital > 0:
                annualized_return = (final_value / initial_capital) ** (252 / n_trading_days) - 1
            
            # Annual volatility
            annualized_vol = None
            if len(daily_returns) >= 2:
                annualized_vol = daily_returns.std() * (252 ** 0.5)
            
            # Sharpe ratio
            sharpe = None
            if len(daily_returns) >= 2 and daily_returns.std() > 0:
                sharpe = daily_returns.mean() / daily_returns.std() * (252 ** 0.5)
            
            # Max drawdown
            peak = equity_series.expanding().max()
            drawdown = (equity_series / peak - 1).min()
            max_drawdown = drawdown if pd.notna(drawdown) else None
            
        else:
            # Basic implementation without pandas
            equity_values = [safe_float(row.get('equity', 0)) for row in equity_data if safe_float(row.get('equity', 0)) is not None]
            
            if len(equity_values) < 2:
                return {
                    'initial_capital': equity_values[0] if equity_values else None,
                    'final_value': equity_values[-1] if equity_values else None,
                    'total_return': None,
                    'annualized_return': None,
                    'annualized_vol': None,
                    'sharpe': None,
                    'max_drawdown': None
                }
            
            initial_capital = equity_values[0]
            final_value = equity_values[-1]
            
            # Total return
            total_return = (final_value / initial_capital - 1) if initial_capital > 0 else 0
            
            # Daily returns calculation
            daily_returns = []
            for i in range(1, len(equity_values)):
                if equity_values[i-1] > 0:
                    daily_returns.append(equity_values[i] / equity_values[i-1] - 1)
            
            # Annual return (CAGR)
            n_trading_days = len(equity_values) - 1
            annualized_return = None
            if n_trading_days >= 2 and initial_capital > 0:
                annualized_return = (final_value / initial_capital) ** (252 / n_trading_days) - 1
            
            # Annual volatility
            annualized_vol = None
            if len(daily_returns) >= 2:
                mean_return = sum(daily_returns) / len(daily_returns)
                variance = sum((r - mean_return) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
                annualized_vol = (variance ** 0.5) * (252 ** 0.5)
            
            # Sharpe ratio
            sharpe = None
            if len(daily_returns) >= 2 and annualized_vol and annualized_vol > 0:
                mean_return = sum(daily_returns) / len(daily_returns)
                sharpe = mean_return / (annualized_vol / (252 ** 0.5))
            
            # Max drawdown
            max_drawdown = None
            peak = 0
            min_drawdown = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                if peak > 0:
                    drawdown = (value / peak - 1)
                    if drawdown < min_drawdown:
                        min_drawdown = drawdown
            max_drawdown = min_drawdown if min_drawdown < 0 else None
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_vol': annualized_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }
        
    except Exception as e:
        print(f"Error computing performance metrics: {e}")
        return {
            'initial_capital': None,
            'final_value': None,
            'total_return': None,
            'annualized_return': None,
            'annualized_vol': None,
            'sharpe': None,
            'max_drawdown': None
        }

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
    
    # Initialize empty defaults
    data['capital'] = {'initial': None, 'final': None}
    data['performance'] = {
        'total_return': None, 'annualized_return': None,
        'annualized_vol': None, 'sharpe': None, 'max_drawdown': None
    }
    data['period'] = {'start': '', 'end': ''}
    
    # Try to load equity curve first for auto-computation
    equity_file = base_path / 'equity_curve.csv'
    equity_data = None
    auto_computed_metrics = {}
    
    if equity_file.exists():
        try:
            equity_data = read_csv_data(equity_file)
            equity_data = normalize_date_column(equity_data)
            equity_data = normalize_equity_column(equity_data)
            auto_computed_metrics = compute_performance_metrics(equity_data)
            
            # Set period from equity data if available
            has_valid_data = False
            if equity_data is not None and auto_computed_metrics.get('initial_capital') is not None:
                has_valid_data = True
                if HAS_PANDAS and hasattr(equity_data, 'date'):
                    if not equity_data['date'].isna().all():
                        data['period']['start'] = str(equity_data['date'].min().date())
                        data['period']['end'] = str(equity_data['date'].max().date())
                elif isinstance(equity_data, list) and equity_data:
                    dates = [row.get('date', '') for row in equity_data if row.get('date')]
                    if dates:
                        data['period']['start'] = str(min(dates))
                        data['period']['end'] = str(max(dates))
                        
        except Exception as e:
            print(f"Error loading equity_curve.csv: {e}")
    
    # Load summary.csv (existing metrics take precedence over auto-computed)
    summary_file = base_path / 'summary.csv'
    if summary_file.exists():
        try:
            summary_data = read_csv_data(summary_file)
            if summary_data:
                if HAS_PANDAS:
                    row = summary_data.iloc[0] if len(summary_data) > 0 else {}
                else:
                    row = summary_data[0] if summary_data else {}
                
                data['capital'] = {
                    'initial': safe_float(row.get('initial_capital')) or auto_computed_metrics.get('initial_capital'),
                    'final': safe_float(row.get('final_capital')) or auto_computed_metrics.get('final_value')
                }
                data['performance'] = {
                    'total_return': safe_float(row.get('total_return')) or auto_computed_metrics.get('total_return'),
                    'annualized_return': safe_float(row.get('annualized_return')) or auto_computed_metrics.get('annualized_return'),
                    'annualized_vol': safe_float(row.get('annualized_volatility')) or auto_computed_metrics.get('annualized_vol'),
                    'sharpe': safe_float(row.get('sharpe_ratio')) or auto_computed_metrics.get('sharpe'),
                    'max_drawdown': safe_float(row.get('max_drawdown')) or auto_computed_metrics.get('max_drawdown')
                }
                
                # Use summary dates if available, otherwise use auto-computed
                summary_start = row.get('start_date', '')
                summary_end = row.get('end_date', '')
                if summary_start:
                    data['period']['start'] = summary_start
                if summary_end:
                    data['period']['end'] = summary_end
                    
        except Exception as e:
            print(f"Error loading summary.csv: {e}")
    else:
        # Use auto-computed metrics if no summary.csv
        data['capital'] = {
            'initial': auto_computed_metrics.get('initial_capital'),
            'final': auto_computed_metrics.get('final_value')
        }
        data['performance'] = {
            'total_return': auto_computed_metrics.get('total_return'),
            'annualized_return': auto_computed_metrics.get('annualized_return'),
            'annualized_vol': auto_computed_metrics.get('annualized_vol'),
            'sharpe': auto_computed_metrics.get('sharpe'),
            'max_drawdown': auto_computed_metrics.get('max_drawdown')
        }
    
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
    
    # Load trades.csv with enhanced parsing
    trades_file = base_path / 'trades.csv'
    auto_trades_count = None
    auto_win_rate = None
    
    if trades_file.exists():
        try:
            trades_data = read_csv_data(trades_file)
            has_trades_data = False
            if HAS_PANDAS and hasattr(trades_data, 'empty'):
                has_trades_data = not trades_data.empty
            elif isinstance(trades_data, list):
                has_trades_data = bool(trades_data)
            else:
                has_trades_data = trades_data is not None
                
            if has_trades_data:
                if HAS_PANDAS:
                    auto_trades_count = len(trades_data)
                    
                    # Look for PnL column variations
                    pnl_col = None
                    for col_name in ['pnl', 'PnL', 'PNL', 'profit_loss', 'profit', 'return']:
                        if col_name in trades_data.columns:
                            pnl_col = col_name
                            break
                    
                    if pnl_col:
                        pnl_values = pd.to_numeric(trades_data[pnl_col], errors='coerce').dropna()
                        if len(pnl_values) > 0:
                            auto_win_rate = (pnl_values > 0).mean()
                    
                    # Calculate trade statistics
                    trades_data['pnl'] = pd.to_numeric(trades_data.get('pnl', []), errors='coerce')
                    trades_data['pnl_pct'] = pd.to_numeric(trades_data.get('pnl_pct', []), errors='coerce')
                    
                    data['trades_summary'] = {
                        'count': auto_trades_count,
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
                    auto_trades_count = len(trades_data)
                    
                    # Look for PnL column variations
                    pnl_col = None
                    if trades_data:
                        for col_name in ['pnl', 'PnL', 'PNL', 'profit_loss', 'profit', 'return']:
                            if col_name in trades_data[0]:
                                pnl_col = col_name
                                break
                    
                    if pnl_col:
                        pnl_values = [safe_float(row.get(pnl_col)) for row in trades_data]
                        pnl_values = [v for v in pnl_values if v is not None]
                        if pnl_values:
                            auto_win_rate = sum(1 for v in pnl_values if v > 0) / len(pnl_values)
                    
                    pnl_values = [safe_float(row.get('pnl', 0)) for row in trades_data]
                    pnl_pct_values = [safe_float(row.get('pnl_pct', 0)) for row in trades_data]
                    
                    data['trades_summary'] = {
                        'count': auto_trades_count,
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
        except Exception as e:
            print(f"Error loading trades.csv: {e}")
            data['trades_summary'] = None
            data['top_winners'] = []
            data['top_losers'] = []
    else:
        data['trades_summary'] = None
        data['top_winners'] = []
        data['top_losers'] = []
    
    # Update trading metrics with auto-computed values if not available
    if data['trading']['trade_count'] is None and auto_trades_count is not None:
        data['trading']['trade_count'] = auto_trades_count
    if data['trading']['win_rate'] is None and auto_win_rate is not None:
        data['trading']['win_rate'] = auto_win_rate
    
    # Load positions_end_of_day.csv, positions.csv, or holdings.csv
    positions_files = ['positions_end_of_day.csv', 'positions.csv', 'holdings.csv']
    positions_data = None
    
    for filename in positions_files:
        positions_file = base_path / filename
        if positions_file.exists():
            try:
                positions_data = read_csv_data(positions_file)
                break
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    # Check if we have any positions data
    has_any_positions_data = False
    if HAS_PANDAS and hasattr(positions_data, 'empty'):
        has_any_positions_data = not positions_data.empty
    elif isinstance(positions_data, list):
        has_any_positions_data = bool(positions_data)
    else:
        has_any_positions_data = positions_data is not None
    
    if has_any_positions_data:
        try:
            # Check if positions data is valid
            has_positions_data = False
            if HAS_PANDAS and hasattr(positions_data, 'empty'):
                has_positions_data = not positions_data.empty
            elif isinstance(positions_data, list):
                has_positions_data = bool(positions_data)
            else:
                has_positions_data = positions_data is not None
                
            if has_positions_data:
                # Normalize column names for positions
                if HAS_PANDAS and hasattr(positions_data, 'columns'):
                    # Map common quantity aliases
                    qty_aliases = ['quantity', 'qty', 'shares', 'Quantity', 'Qty', 'Shares']
                    for alias in qty_aliases:
                        if alias in positions_data.columns and alias != 'quantity':
                            positions_data = positions_data.rename(columns={alias: 'quantity'})
                            break
                    
                    # Map common value/weight aliases
                    value_aliases = ['value', 'weight', 'market_value', 'Value', 'Weight', 'MarketValue']
                    for alias in value_aliases:
                        if alias in positions_data.columns and alias != 'value':
                            positions_data = positions_data.rename(columns={alias: 'value'})
                            break
                    
                    data['positions'] = positions_data.head(30).to_dict('records')
                else:
                    # Basic processing for positions
                    processed_positions = []
                    for row in positions_data[:30]:
                        processed_row = dict(row)
                        # Normalize quantity column
                        for alias in ['quantity', 'qty', 'shares', 'Quantity', 'Qty', 'Shares']:
                            if alias in row and alias != 'quantity':
                                processed_row['quantity'] = row[alias]
                                break
                        # Normalize value column  
                        for alias in ['value', 'weight', 'market_value', 'Value', 'Weight', 'MarketValue']:
                            if alias in row and alias != 'value':
                                processed_row['value'] = row[alias]
                                break
                        processed_positions.append(processed_row)
                    data['positions'] = processed_positions
        except Exception as e:
            print(f"Error processing positions data: {e}")
            data['positions'] = []
    else:
        data['positions'] = []
    
    # Load factor_exposures.csv, industry_exposure.csv, or factor_exposure.csv
    exposure_files = ['factor_exposures.csv', 'industry_exposure.csv', 'factor_exposure.csv']
    exposure_data = None
    
    for filename in exposure_files:
        exposure_file = base_path / filename
        if exposure_file.exists():
            try:
                exposure_data = read_csv_data(exposure_file)
                break
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    # Check if we have any exposure data
    has_any_exposure_data = False
    if HAS_PANDAS and hasattr(exposure_data, 'empty'):
        has_any_exposure_data = not exposure_data.empty
    elif isinstance(exposure_data, list):
        has_any_exposure_data = bool(exposure_data)
    else:
        has_any_exposure_data = exposure_data is not None
    
    if has_any_exposure_data:
        try:
            # Check if exposure data is valid
            has_exposure_data = False
            if HAS_PANDAS and hasattr(exposure_data, 'empty'):
                has_exposure_data = not exposure_data.empty
            elif isinstance(exposure_data, list):
                has_exposure_data = bool(exposure_data)
            else:
                has_exposure_data = exposure_data is not None
                
            if has_exposure_data:
                if HAS_PANDAS and hasattr(exposure_data, 'to_dict'):
                    data['factor_exposures'] = exposure_data.head(50).to_dict('records')
                else:
                    data['factor_exposures'] = exposure_data[:50] if exposure_data else []
        except Exception as e:
            print(f"Error processing exposure data: {e}")
            data['factor_exposures'] = []
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
    """Generate equity curve and drawdown charts"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping chart generation")
        return {'equity_curve_png': None, 'drawdown_png': None}
    
    base_path = Path(base_dir)
    charts_dir = base_path / 'charts'
    charts_dir.mkdir(exist_ok=True)
    
    equity_file = base_path / 'equity_curve.csv'
    chart_paths = {'equity_curve_png': None, 'drawdown_png': None}
    
    if equity_file.exists():
        try:
            equity_data = read_csv_data(equity_file)
            equity_data = normalize_date_column(equity_data)
            equity_data = normalize_equity_column(equity_data)
            
            if HAS_PANDAS:
                if equity_data is not None and not equity_data.empty and len(equity_data) > 0:
                    equity_data = equity_data.sort_values('date')
                    
                    dates = equity_data['date']
                    values = equity_data['equity']
                    
                    # Calculate drawdown
                    peak = values.expanding().max()
                    drawdown = (values - peak) / peak
                else:
                    return chart_paths
            else:
                # Basic implementation without pandas
                if not equity_data:
                    return chart_paths
                    
                dates = [row.get('date', '') for row in equity_data]
                values = [row.get('equity', 0) for row in equity_data]
                
                # Simple drawdown calculation
                drawdown = []
                peak = 0
                for val in values:
                    if val > peak:
                        peak = val
                    drawdown.append((val - peak) / peak if peak > 0 else 0)
            
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