#!/usr/bin/env python3
"""
generate_backtest_report.py

Automated backtest report generation with Sortino ratio and Rolling Sharpe chart.
Handles missing dependencies gracefully and creates comprehensive performance reports.

Usage:
    python generate_backtest_report.py
    
Environment Variables:
    ROLLING_WINDOW_DAYS: Rolling window for Sharpe calculation (default: 60, min: 30)
    RISK_FREE_RATE: Annual risk-free rate (default: 0.0)
"""

import os
import sys
import json
import csv
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Try to import optional dependencies
HAS_PANDAS = False
HAS_NUMPY = False
HAS_MATPLOTLIB = False
HAS_JINJA2 = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    pass

try:
    from jinja2 import Template
    HAS_JINJA2 = True
except ImportError:
    pass


# Configuration constants
TRADING_DAYS_PER_YEAR = 252
DEFAULT_ROLLING_WINDOW = 60
MIN_ROLLING_WINDOW = 30
DEFAULT_RISK_FREE_RATE = 0.0

# File paths
EQUITY_CURVE_FILE = "backtest_results/equity_curve.csv"
METRICS_FILE = "backtest_results/metrics.json"
REPORT_FILE = "backtest_results/report.md"
REPORT_HTML_FILE = "backtest_results/report.html"
ROLLING_SHARPE_CHART = "backtest_results/charts/rolling_sharpe.png"
TEMPLATE_FILE = "templates/report_template.md.j2"


def log_message(message: str, level: str = "INFO") -> None:
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def get_config() -> Dict[str, Any]:
    """Get configuration from environment variables."""
    rolling_window = int(os.getenv("ROLLING_WINDOW_DAYS", DEFAULT_ROLLING_WINDOW))
    rolling_window = max(rolling_window, MIN_ROLLING_WINDOW)
    
    risk_free_rate = float(os.getenv("RISK_FREE_RATE", DEFAULT_RISK_FREE_RATE))
    
    return {
        "rolling_window_days": rolling_window,
        "risk_free_rate_annual": risk_free_rate,
        "risk_free_rate_daily": (1 + risk_free_rate) ** (1/TRADING_DAYS_PER_YEAR) - 1
    }


def read_equity_curve_csv(file_path: str) -> Optional[Tuple[List[str], List[float]]]:
    """
    Read equity curve CSV file without pandas.
    Returns (dates, values) tuple or None if file not found/invalid.
    """
    if not os.path.exists(file_path):
        log_message(f"Equity curve file not found: {file_path}", "WARNING")
        return None
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
            # Find date column (first column) and value column
            date_col = 0
            value_col = None
            
            # Look for value column names
            value_candidates = ['nav', 'net_value', 'equity', 'value', 'close']
            for i, header in enumerate(headers):
                if header.lower() in value_candidates:
                    value_col = i
                    break
            
            if value_col is None:
                # Find numeric column with highest variance
                rows = list(reader)
                if not rows:
                    log_message("Empty CSV file", "ERROR")
                    return None
                
                # Reset reader
                f.seek(0)
                next(csv.reader(f))  # Skip header again
                
                numeric_cols = []
                for i in range(1, len(headers)):  # Skip first column (date)
                    try:
                        values = [float(row[i]) for row in rows if len(row) > i and row[i].strip()]
                        if values:
                            variance = sum((x - sum(values)/len(values))**2 for x in values) / len(values)
                            numeric_cols.append((i, variance))
                    except (ValueError, IndexError):
                        continue
                
                if numeric_cols:
                    value_col = max(numeric_cols, key=lambda x: x[1])[0]
                else:
                    log_message("No numeric columns found in CSV", "ERROR")
                    return None
            
            # Read data
            f.seek(0)
            next(csv.reader(f))  # Skip header
            
            dates = []
            values = []
            
            for row in csv.reader(f):
                if len(row) > max(date_col, value_col):
                    try:
                        date_str = row[date_col].strip()
                        value_str = row[value_col].strip()
                        
                        if date_str and value_str:
                            dates.append(date_str)
                            values.append(float(value_str))
                    except (ValueError, IndexError):
                        continue
            
            if not dates or not values:
                log_message("No valid data found in CSV", "ERROR")
                return None
                
            log_message(f"Successfully read {len(dates)} data points from {file_path}")
            log_message(f"Using column '{headers[value_col]}' as value column")
            return dates, values
            
    except Exception as e:
        log_message(f"Error reading CSV file: {e}", "ERROR")
        return None


def calculate_returns(values: List[float]) -> List[float]:
    """Calculate daily returns from value series."""
    if len(values) < 2:
        return []
    
    returns = []
    for i in range(1, len(values)):
        if values[i-1] != 0:
            ret = (values[i] / values[i-1]) - 1
            returns.append(ret)
        else:
            returns.append(0.0)
    
    return returns


def calculate_sharpe_ratio(returns: List[float], risk_free_rate_daily: float) -> Optional[float]:
    """Calculate annualized Sharpe ratio."""
    if not returns:
        return None
    
    excess_returns = [r - risk_free_rate_daily for r in returns]
    mean_excess = sum(excess_returns) / len(excess_returns)
    
    # Calculate standard deviation
    if len(excess_returns) < 2:
        return None
    
    variance = sum((r - mean_excess) ** 2 for r in excess_returns) / (len(excess_returns) - 1)
    std_excess = variance ** 0.5
    
    if std_excess == 0:
        return None
    
    sharpe = (mean_excess / std_excess) * (TRADING_DAYS_PER_YEAR ** 0.5)
    return sharpe


def calculate_sortino_ratio(returns: List[float], risk_free_rate_daily: float) -> Optional[float]:
    """Calculate Sortino ratio using downside deviation."""
    if not returns:
        return None
    
    excess_returns = [r - risk_free_rate_daily for r in returns]
    mean_excess = sum(excess_returns) / len(excess_returns)
    
    # Calculate downside deviation (only negative excess returns)
    negative_excess = [r for r in excess_returns if r < 0]
    
    if not negative_excess:
        return None  # No downside risk
    
    downside_variance = sum(r ** 2 for r in negative_excess) / len(negative_excess)
    downside_std = downside_variance ** 0.5
    
    if downside_std == 0:
        return None
    
    # Annualized return
    annualized_return = mean_excess * TRADING_DAYS_PER_YEAR
    
    # Annualized downside deviation
    downside_std_annualized = downside_std * (TRADING_DAYS_PER_YEAR ** 0.5)
    
    sortino = annualized_return / downside_std_annualized
    return sortino


def calculate_annualized_return(values: List[float]) -> Optional[float]:
    """Calculate annualized return."""
    if len(values) < 2:
        return None
    
    start_value = values[0]
    end_value = values[-1]
    
    if start_value <= 0:
        return None
    
    total_return = (end_value / start_value) - 1
    num_periods = len(values) - 1  # Number of return periods
    
    if num_periods == 0:
        return None
    
    annualized = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / num_periods) - 1
    return annualized


def calculate_max_drawdown(values: List[float]) -> Optional[float]:
    """Calculate maximum drawdown."""
    if len(values) < 2:
        return None
    
    running_peak = values[0]
    max_drawdown = 0.0
    
    for value in values[1:]:
        if value > running_peak:
            running_peak = value
        
        drawdown = (value / running_peak) - 1
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    
    return max_drawdown


def calculate_rolling_sharpe(returns: List[float], window: int, risk_free_rate_daily: float) -> List[float]:
    """Calculate rolling Sharpe ratio series."""
    if len(returns) < window:
        return []
    
    rolling_sharpe = []
    
    for i in range(window - 1, len(returns)):
        window_returns = returns[i - window + 1:i + 1]
        sharpe = calculate_sharpe_ratio(window_returns, risk_free_rate_daily)
        rolling_sharpe.append(sharpe if sharpe is not None else 0.0)
    
    return rolling_sharpe


def create_rolling_sharpe_chart(dates: List[str], rolling_sharpe: List[float], output_path: str) -> bool:
    """Create rolling Sharpe ratio chart."""
    if not HAS_MATPLOTLIB:
        log_message("Matplotlib not available, skipping chart generation", "WARNING")
        return False
    
    if not rolling_sharpe:
        log_message("No rolling Sharpe data to plot", "WARNING")
        return False
    
    try:
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Use dates corresponding to rolling sharpe (offset by window)
        window_offset = len(dates) - len(rolling_sharpe)
        chart_dates = dates[window_offset:]
        
        # Plot rolling Sharpe
        plt.plot(range(len(rolling_sharpe)), rolling_sharpe, 'b-', linewidth=1.5, label='Rolling Sharpe Ratio')
        
        # Add reference lines
        plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
        plt.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='Sharpe = 2.0')
        plt.axhline(y=0.0, color='k', linestyle='-', alpha=0.3)
        
        plt.title('Rolling Sharpe Ratio Over Time')
        plt.xlabel('Time Period')
        plt.ylabel('Rolling Sharpe Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log_message(f"Rolling Sharpe chart saved to {output_path}")
        return True
        
    except Exception as e:
        log_message(f"Error creating chart: {e}", "ERROR")
        return False


def load_existing_metrics() -> Dict[str, Any]:
    """Load existing metrics.json if it exists."""
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            log_message(f"Error loading existing metrics: {e}", "WARNING")
    
    return {}


def update_metrics(metrics: Dict[str, Any]) -> bool:
    """Update metrics.json file."""
    try:
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
        
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        log_message(f"Metrics updated in {METRICS_FILE}")
        return True
        
    except Exception as e:
        log_message(f"Error updating metrics: {e}", "ERROR")
        return False


def generate_report(metrics: Dict[str, Any], template_path: str, output_path: str) -> bool:
    """Generate markdown report using template."""
    if not os.path.exists(template_path):
        log_message(f"Template file not found: {template_path}", "WARNING")
        # Create a simple report without template
        return create_simple_report(metrics, output_path)
    
    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        if HAS_JINJA2:
            template = Template(template_content)
            report_content = template.render(**metrics)
        else:
            # Simple string replacement fallback
            report_content = template_content
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        placeholder = f"{{{{ {key}.{subkey} }}}}"
                        if subvalue is not None:
                            if isinstance(subvalue, float):
                                if subkey in ['annualized_return', 'max_drawdown']:
                                    # Convert to percentage for these metrics
                                    report_content = report_content.replace(placeholder, f"{subvalue * 100:.2f}%")
                                else:
                                    report_content = report_content.replace(placeholder, f"{subvalue:.4f}")
                            else:
                                report_content = report_content.replace(placeholder, str(subvalue))
                        else:
                            report_content = report_content.replace(placeholder, "N/A")
                else:
                    placeholder = f"{{{{ {key} }}}}"
                    if value is not None:
                        if isinstance(value, float):
                            if key == 'risk_free_rate_annual':
                                report_content = report_content.replace(placeholder, f"{value * 100:.2f}%")
                            else:
                                report_content = report_content.replace(placeholder, f"{value:.4f}")
                        else:
                            report_content = report_content.replace(placeholder, str(value))
                    else:
                        report_content = report_content.replace(placeholder, "N/A")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        log_message(f"Report generated: {output_path}")
        return True
        
    except Exception as e:
        log_message(f"Error generating report: {e}", "ERROR")
        return create_simple_report(metrics, output_path)


def create_simple_report(metrics: Dict[str, Any], output_path: str) -> bool:
    """Create a simple report without template."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# Backtest Performance Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Performance Metrics\n\n")
            
            if 'performance' in metrics:
                perf = metrics['performance']
                sharpe_str = f"{perf['sharpe']:.4f}" if perf.get('sharpe') is not None else "N/A"
                sortino_str = f"{perf['sortino']:.4f}" if perf.get('sortino') is not None else "N/A"
                ann_ret_str = f"{perf['annualized_return']*100:.2f}%" if perf.get('annualized_return') is not None else "N/A"
                max_dd_str = f"{perf['max_drawdown']*100:.2f}%" if perf.get('max_drawdown') is not None else "N/A"
                
                f.write(f"- **Sharpe Ratio**: {sharpe_str}\n")
                f.write(f"- **Sortino Ratio**: {sortino_str}\n")
                f.write(f"- **Annualized Return**: {ann_ret_str}\n")
                f.write(f"- **Maximum Drawdown**: {max_dd_str}\n\n")
            
            if 'charts' in metrics:
                charts = metrics['charts']
                f.write("## Charts\n\n")
                if 'rolling_sharpe' in charts:
                    f.write(f"- Rolling Sharpe Ratio: ![Rolling Sharpe]({charts['rolling_sharpe']})\n")
        
        log_message(f"Simple report generated: {output_path}")
        return True
        
    except Exception as e:
        log_message(f"Error creating simple report: {e}", "ERROR")
        return False


def main():
    """Main function to generate backtest report."""
    log_message("Starting backtest report generation")
    
    # Check dependency status
    log_message(f"Dependencies: pandas={HAS_PANDAS}, numpy={HAS_NUMPY}, matplotlib={HAS_MATPLOTLIB}, jinja2={HAS_JINJA2}")
    
    # Get configuration
    config = get_config()
    log_message(f"Configuration: {config}")
    
    # Read equity curve data
    equity_data = read_equity_curve_csv(EQUITY_CURVE_FILE)
    if equity_data is None:
        log_message("Failed to read equity curve data. Creating empty report.", "WARNING")
        # Create empty metrics and report
        empty_metrics = {
            'performance': {
                'sharpe': None,
                'sortino': None,
                'annualized_return': None,
                'max_drawdown': None
            },
            'charts': {},
            'timestamp': datetime.now().isoformat(),
            'config': config
        }
        update_metrics(empty_metrics)
        generate_report(empty_metrics, TEMPLATE_FILE, REPORT_FILE)
        return
    
    dates, values = equity_data
    
    # Calculate returns
    returns = calculate_returns(values)
    log_message(f"Calculated {len(returns)} daily returns")
    
    # Calculate performance metrics
    sharpe = calculate_sharpe_ratio(returns, config['risk_free_rate_daily'])
    sortino = calculate_sortino_ratio(returns, config['risk_free_rate_daily'])
    annualized_return = calculate_annualized_return(values)
    max_drawdown = calculate_max_drawdown(values)
    
    log_message(f"Performance metrics calculated:")
    log_message(f"  Sharpe Ratio: {sharpe}")
    log_message(f"  Sortino Ratio: {sortino}")
    log_message(f"  Annualized Return: {annualized_return}")
    log_message(f"  Max Drawdown: {max_drawdown}")
    
    # Calculate rolling Sharpe
    rolling_sharpe = calculate_rolling_sharpe(returns, config['rolling_window_days'], config['risk_free_rate_daily'])
    log_message(f"Calculated {len(rolling_sharpe)} rolling Sharpe values")
    
    # Create rolling Sharpe chart
    chart_created = create_rolling_sharpe_chart(dates, rolling_sharpe, ROLLING_SHARPE_CHART)
    
    # Load existing metrics and update
    existing_metrics = load_existing_metrics()
    
    # Update performance metrics
    existing_metrics['performance'] = existing_metrics.get('performance', {})
    existing_metrics['performance'].update({
        'sharpe': sharpe,
        'sortino': sortino,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown
    })
    
    # Update charts
    existing_metrics['charts'] = existing_metrics.get('charts', {})
    if chart_created:
        existing_metrics['charts']['rolling_sharpe'] = ROLLING_SHARPE_CHART
    
    # Add metadata
    existing_metrics['timestamp'] = datetime.now().isoformat()
    existing_metrics['config'] = config
    existing_metrics['data_points'] = len(values)
    existing_metrics['return_periods'] = len(returns)
    
    # Update metrics file
    update_metrics(existing_metrics)
    
    # Generate report
    generate_report(existing_metrics, TEMPLATE_FILE, REPORT_FILE)
    
    log_message("Backtest report generation completed")


if __name__ == "__main__":
    main()