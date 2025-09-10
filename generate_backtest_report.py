#!/usr/bin/env python3
"""
generate_backtest_report.py

Enhanced backtest report generator with Sortino ratio calculation and Rolling Sharpe chart generation.

This script enhances backtest reporting by adding:
1. Sortino ratio calculation - measures risk-adjusted returns considering only downside volatility
2. Rolling Sharpe ratio chart generation - shows how risk-adjusted performance changes over time
3. Environment variable configuration for flexibility
4. Robust error handling for missing data and edge cases

Environment Variables:
    ROLLING_WINDOW_DAYS (int): Rolling window for Sharpe calculation (default: 60)
        - Minimum recommended: 30 days
        - Typical values: 60, 90, 120 days
    RISK_FREE_RATE (float): Annual risk-free rate (default: 0.0)
        - Used for excess return calculations in Sortino and Sharpe ratios
        - Example: 0.02 for 2% annual risk-free rate

Usage:
    python generate_backtest_report.py [backtest_results_directory]
    
    # With environment variables:
    ROLLING_WINDOW_DAYS=90 RISK_FREE_RATE=0.02 python generate_backtest_report.py

Requirements:
    - Minimum data: equity_curve.csv with >= 30 rows
    - Expected columns: date, nav (or variations like equity, portfolio_value)
    - Optional: pandas, numpy, matplotlib for enhanced functionality

Output:
    - Updates metrics.json with Sortino ratio and annualized return
    - Creates rolling_sharpe.png chart (if matplotlib available)
    - Handles missing data gracefully with appropriate fallbacks
"""

import os
import sys
import json
import csv
import math
from pathlib import Path
from datetime import datetime

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("Warning: pandas not available, using basic CSV parsing")
    HAS_PANDAS = False
    pd = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("Warning: numpy not available, using basic math functions")
    HAS_NUMPY = False
    np = None

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not available, skipping chart generation")
    HAS_MATPLOTLIB = False
    plt = None

try:
    from jinja2 import Template
    HAS_JINJA2 = True
except ImportError:
    print("Warning: jinja2 not available, using basic string templating")
    HAS_JINJA2 = False
    Template = None


def get_env_var(var_name, default_value, var_type=str):
    """Get environment variable with type conversion and default."""
    value = os.environ.get(var_name)
    if value is None:
        return default_value
    try:
        if var_type == int:
            return int(value)
        elif var_type == float:
            return float(value)
        else:
            return str(value)
    except ValueError:
        print(f"Warning: Invalid {var_name} value '{value}', using default {default_value}")
        return default_value


def load_equity_curve(equity_path):
    """Load equity curve data from CSV file."""
    try:
        if HAS_PANDAS:
            # Use pandas if available
            df = pd.read_csv(equity_path)
            # Handle different possible column names for NAV/equity
            nav_col = None
            for col in ['nav', 'NAV', 'equity', 'cumulative_return', 'portfolio_value']:
                if col in df.columns:
                    nav_col = col
                    break
            
            if nav_col is None:
                print(f"Error: No NAV column found in {equity_path}")
                return None
                
            # Ensure we have a date column
            date_col = None
            for col in ['date', 'Date', 'timestamp', 'index']:
                if col in df.columns:
                    date_col = col
                    break
                    
            if date_col is None:
                print(f"Error: No date column found in {equity_path}")
                return None
                
            df['date'] = pd.to_datetime(df[date_col])
            df['nav'] = pd.to_numeric(df[nav_col], errors='coerce')
            df = df.dropna(subset=['nav'])
            df = df.sort_values('date').reset_index(drop=True)
            
            return df[['date', 'nav']]
        else:
            # Use basic CSV parsing
            data = []
            with open(equity_path, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                # Find NAV and date columns
                nav_col = None
                date_col = None
                
                for col in headers:
                    if col.lower() in ['nav', 'equity', 'cumulative_return', 'portfolio_value']:
                        nav_col = col
                    if col.lower() in ['date', 'timestamp']:
                        date_col = col
                
                if nav_col is None or date_col is None:
                    print(f"Error: Required columns not found in {equity_path}")
                    return None
                
                for row in reader:
                    try:
                        nav_value = float(row[nav_col])
                        date_value = row[date_col]
                        data.append({'date': date_value, 'nav': nav_value})
                    except ValueError:
                        continue  # Skip invalid rows
            
            if len(data) == 0:
                print(f"Error: No valid data found in {equity_path}")
                return None
                
            return data
    except Exception as e:
        print(f"Error loading equity curve from {equity_path}: {e}")
        return None


def calculate_daily_returns(equity_data, risk_free_rate_annual=0.0):
    """Calculate daily returns from NAV data."""
    if HAS_PANDAS and hasattr(equity_data, 'iloc'):
        # Use pandas Series
        nav_series = equity_data['nav']
        daily_returns = nav_series.pct_change().dropna()
    else:
        # Use basic list operations
        nav_values = [row['nav'] for row in equity_data]
        daily_returns = []
        for i in range(1, len(nav_values)):
            daily_ret = (nav_values[i] - nav_values[i-1]) / nav_values[i-1]
            daily_returns.append(daily_ret)
    
    # Convert annual risk-free rate to daily
    daily_rf_rate = (1 + risk_free_rate_annual) ** (1/252) - 1
    
    # Calculate excess returns
    if HAS_PANDAS and hasattr(daily_returns, 'values'):
        excess_returns = daily_returns - daily_rf_rate
    else:
        excess_returns = [r - daily_rf_rate for r in daily_returns]
    
    return daily_returns, excess_returns


def calculate_sortino_ratio(daily_returns, risk_free_rate_annual=0.0):
    """Calculate Sortino ratio."""
    if HAS_PANDAS and hasattr(daily_returns, '__len__'):
        returns_len = len(daily_returns)
    elif isinstance(daily_returns, list):
        returns_len = len(daily_returns)
    else:
        returns_len = 0
        
    if returns_len == 0:
        return None
        
    # Convert annual risk-free rate to daily
    daily_rf_rate = (1 + risk_free_rate_annual) ** (1/252) - 1
    
    if HAS_PANDAS and hasattr(daily_returns, 'values'):
        excess_returns = daily_returns - daily_rf_rate
        negative_returns = excess_returns[excess_returns < 0]
    else:
        excess_returns = [r - daily_rf_rate for r in daily_returns]
        negative_returns = [r for r in excess_returns if r < 0]
    
    if len(negative_returns) == 0:
        # No negative returns - return None
        return None
    
    # Annualized return
    if HAS_PANDAS and hasattr(excess_returns, 'mean'):
        mean_daily_return = excess_returns.mean()
        downside_std = negative_returns.std()
    else:
        mean_daily_return = sum(excess_returns) / len(excess_returns)
        # Calculate standard deviation manually
        neg_mean = sum(negative_returns) / len(negative_returns)
        variance = sum((r - neg_mean)**2 for r in negative_returns) / len(negative_returns)
        downside_std = math.sqrt(variance)
    
    annualized_return = mean_daily_return * 252
    
    # Downside deviation (annualized)
    downside_deviation = downside_std * math.sqrt(252)
    
    if downside_deviation == 0:
        return None
        
    sortino_ratio = annualized_return / downside_deviation
    return sortino_ratio


def calculate_rolling_sharpe(daily_returns, window_days=60):
    """Calculate rolling Sharpe ratio."""
    if HAS_PANDAS and hasattr(daily_returns, '__len__'):
        returns_len = len(daily_returns)
    elif isinstance(daily_returns, list):
        returns_len = len(daily_returns)
    else:
        returns_len = 0
        
    if returns_len < window_days:
        print(f"Insufficient data for rolling Sharpe (need >= {window_days} days, have {returns_len})")
        return None, None
    
    if HAS_PANDAS and hasattr(daily_returns, 'rolling'):
        # Use pandas rolling functions
        rolling_mean = daily_returns.rolling(window=window_days).mean()
        rolling_std = daily_returns.rolling(window=window_days).std()
        
        # Skip windows with std=0
        valid_mask = (rolling_std > 0) & rolling_mean.notna() & rolling_std.notna()
        
        rolling_sharpe = np.where(valid_mask, 
                                 (rolling_mean * np.sqrt(252)) / (rolling_std * np.sqrt(252)),
                                 np.nan)
    else:
        # Use basic list operations for rolling calculation
        rolling_sharpe = []
        valid_mask = []
        
        for i in range(len(daily_returns)):
            if i < window_days - 1:
                rolling_sharpe.append(float('nan'))
                valid_mask.append(False)
            else:
                window_data = daily_returns[i - window_days + 1:i + 1]
                window_mean = sum(window_data) / len(window_data)
                
                # Calculate standard deviation
                variance = sum((r - window_mean)**2 for r in window_data) / len(window_data)
                window_std = math.sqrt(variance)
                
                if window_std > 0:
                    sharpe = (window_mean * math.sqrt(252)) / (window_std * math.sqrt(252))
                    rolling_sharpe.append(sharpe)
                    valid_mask.append(True)
                else:
                    rolling_sharpe.append(float('nan'))
                    valid_mask.append(False)
    
    return rolling_sharpe, valid_mask


def create_rolling_sharpe_chart(dates, rolling_sharpe, save_path, window_days=60):
    """Create and save rolling Sharpe ratio chart."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping chart creation")
        return False
        
    try:
        plt.figure(figsize=(12, 6))
        
        # Filter out NaN values for plotting
        if HAS_NUMPY:
            valid_data = ~np.isnan(rolling_sharpe)
        else:
            valid_data = [not (math.isnan(val) if isinstance(val, float) else False) for val in rolling_sharpe]
        
        if HAS_PANDAS and hasattr(dates, 'iloc'):
            plot_dates = dates.iloc[valid_data] if hasattr(dates, 'iloc') else [dates[i] for i, v in enumerate(valid_data) if v]
        else:
            plot_dates = [dates[i] for i, v in enumerate(valid_data) if v]
            
        plot_sharpe = [rolling_sharpe[i] for i, v in enumerate(valid_data) if v]
        
        if len(plot_dates) == 0:
            print("No valid rolling Sharpe data to plot")
            return False
            
        plt.plot(plot_dates, plot_sharpe, linewidth=1.5, color='blue', alpha=0.8)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        plt.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Sharpe = 2')
        
        plt.title(f'Rolling Sharpe Ratio ({window_days}-day window)')
        plt.xlabel('Date')
        plt.ylabel('Rolling Sharpe Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error creating rolling Sharpe chart: {e}")
        return False


def load_or_create_metrics(metrics_path):
    """Load existing metrics.json or create new structure."""
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Error loading metrics from {metrics_path}: {e}")
            metrics = {}
    else:
        metrics = {}
    
    # Ensure performance dict exists
    if 'performance' not in metrics:
        metrics['performance'] = {}
        
    return metrics


def save_metrics(metrics, metrics_path):
    """Save metrics to JSON file."""
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving metrics to {metrics_path}: {e}")
        return False


def calculate_annualized_return(daily_returns):
    """Calculate annualized return from daily returns."""
    if HAS_PANDAS and hasattr(daily_returns, '__len__'):
        returns_len = len(daily_returns)
    elif isinstance(daily_returns, list):
        returns_len = len(daily_returns)
    else:
        returns_len = 0
        
    if returns_len == 0:
        return 0.0
    
    # Calculate total return
    if HAS_PANDAS and hasattr(daily_returns, 'values'):
        total_return = (1 + daily_returns).prod() - 1
    else:
        total_return = 1.0
        for ret in daily_returns:
            total_return *= (1 + ret)
        total_return -= 1
    
    num_days = returns_len
    
    if num_days < 252:
        # Annualize based on actual number of days
        annualized_return = (1 + total_return) ** (252 / num_days) - 1
    else:
        annualized_return = (1 + total_return) ** (252 / num_days) - 1
        
    return annualized_return


def generate_backtest_report(backtest_dir=None):
    """Main function to generate enhanced backtest report."""
    
    # Get environment variables
    rolling_window_days = get_env_var('ROLLING_WINDOW_DAYS', 60, int)
    risk_free_rate = get_env_var('RISK_FREE_RATE', 0.0, float)
    
    print(f"Rolling window: {rolling_window_days} days")
    print(f"Risk-free rate: {risk_free_rate:.3f}")
    
    # Determine backtest directory
    if backtest_dir is None:
        backtest_dir = "backtest_results"
    
    backtest_path = Path(backtest_dir)
    if not backtest_path.exists():
        print(f"Backtest directory {backtest_path} does not exist")
        return False
    
    # File paths
    equity_path = backtest_path / "equity_curve.csv"
    metrics_path = backtest_path / "metrics.json"
    charts_dir = backtest_path / "charts"
    rolling_sharpe_chart = charts_dir / "rolling_sharpe.png"
    
    # Load equity curve
    if not equity_path.exists():
        print(f"Equity curve file {equity_path} not found, skipping computations")
        return False
        
    equity_df = load_equity_curve(equity_path)
    if equity_df is None or len(equity_df) < 30:
        print("Insufficient equity curve data (need >= 30 rows)")
        return False
    
    print(f"Loaded equity curve with {len(equity_df)} data points")
    
    # Calculate daily returns
    if HAS_PANDAS and hasattr(equity_df, 'iloc'):
        daily_returns, excess_returns = calculate_daily_returns(equity_df, risk_free_rate)
    else:
        daily_returns, excess_returns = calculate_daily_returns(equity_df, risk_free_rate)
    
    if len(daily_returns) == 0:
        print("No valid daily returns calculated")
        return False
    
    # Load/create metrics
    metrics = load_or_create_metrics(metrics_path)
    
    # Calculate and inject Sortino ratio
    sortino_ratio = calculate_sortino_ratio(daily_returns, risk_free_rate)
    if sortino_ratio is not None:
        metrics['performance']['sortino'] = float(sortino_ratio)
        print(f"Calculated Sortino ratio: {sortino_ratio:.4f}")
    else:
        metrics['performance']['sortino'] = None
        print("Sortino ratio: N/A (no negative returns)")
    
    # Calculate annualized return if not present
    if 'annualized_return' not in metrics['performance']:
        ann_return = calculate_annualized_return(daily_returns)
        metrics['performance']['annualized_return'] = float(ann_return)
        print(f"Calculated annualized return: {ann_return:.4f}")
    
    # Calculate Rolling Sharpe
    rolling_sharpe, valid_mask = calculate_rolling_sharpe(daily_returns, rolling_window_days)
    
    chart_created = False
    if rolling_sharpe is not None:
        # Get dates for charting
        if HAS_PANDAS and hasattr(equity_df, 'iloc'):
            dates_for_chart = equity_df['date'].iloc[rolling_window_days-1:]
        else:
            dates_for_chart = [row['date'] for row in equity_df[rolling_window_days-1:]]
            
        chart_created = create_rolling_sharpe_chart(
            dates_for_chart,  # Align with rolling window
            rolling_sharpe[rolling_window_days-1:],  # Skip initial NaN values
            rolling_sharpe_chart,
            rolling_window_days
        )
        
        if chart_created:
            print(f"Created rolling Sharpe chart: {rolling_sharpe_chart}")
        else:
            print("Failed to create rolling Sharpe chart")
    
    # Update metrics with chart info
    if 'charts' not in metrics:
        metrics['charts'] = {}
    
    if chart_created:
        # Store relative path for template
        metrics['charts']['rolling_sharpe_png'] = f"charts/rolling_sharpe.png"
    else:
        metrics['charts']['rolling_sharpe_png'] = None
    
    # Save updated metrics
    if save_metrics(metrics, metrics_path):
        print(f"Updated metrics saved to {metrics_path}")
    else:
        print("Failed to save metrics")
        return False
    
    print("Backtest report generation completed successfully")
    return True


if __name__ == "__main__":
    # Command line usage
    backtest_dir = sys.argv[1] if len(sys.argv) > 1 else None
    success = generate_backtest_report(backtest_dir)
    sys.exit(0 if success else 1)