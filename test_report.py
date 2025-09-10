#!/usr/bin/env python3
"""
Test script for generate_backtest_report.py functionality
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Create test data and verify the functions work
def create_test_data():
    """Create sample test data for testing."""
    # Create temporary directory
    test_dir = Path("/tmp/test_backtest_results")
    test_dir.mkdir(exist_ok=True)
    
    # Create charts subdirectory
    charts_dir = test_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    # Create sample equity curve data
    equity_data = """date,nav
2023-01-01,10000
2023-01-02,10050
2023-01-03,10020
2023-01-04,10080
2023-01-05,10100
2023-01-08,10150
2023-01-09,10120
2023-01-10,10200
2023-01-11,10180
2023-01-12,10250
2023-01-15,10300
2023-01-16,10280
2023-01-17,10350
2023-01-18,10320
2023-01-19,10400
2023-01-22,10450
2023-01-23,10420
2023-01-24,10500
2023-01-25,10480
2023-01-26,10550
2023-01-29,10600
2023-01-30,10580
2023-01-31,10650
2023-02-01,10620
2023-02-02,10700
2023-02-03,10680
2023-02-06,10750
2023-02-07,10720
2023-02-08,10800
2023-02-09,10780
2023-02-10,10850
2023-02-13,10820
2023-02-14,10900
2023-02-15,10880
2023-02-16,10950
2023-02-17,10920
2023-02-21,11000
2023-02-22,10980
2023-02-23,11050
2023-02-24,11020
2023-02-27,11100
2023-02-28,11080
2023-03-01,11150
2023-03-02,11120
2023-03-03,11200
2023-03-06,11180
2023-03-07,11250
2023-03-08,11220
2023-03-09,11300
2023-03-10,11280
2023-03-13,11350
2023-03-14,11320
2023-03-15,11400
2023-03-16,11380
2023-03-17,11450
2023-03-20,11420
2023-03-21,11500
2023-03-22,11480
2023-03-23,11550
2023-03-24,11520
2023-03-27,11600
2023-03-28,11580
2023-03-29,11650
2023-03-30,11620
2023-03-31,11700
2023-04-03,11680
2023-04-04,11750
2023-04-05,11720
2023-04-06,11800
2023-04-07,11780
2023-04-10,11850
2023-04-11,11820
2023-04-12,11900
2023-04-13,11880
2023-04-14,11950
2023-04-17,11920
2023-04-18,12000
2023-04-19,11980
2023-04-20,12050
2023-04-21,12020
2023-04-24,12100
2023-04-25,12080
2023-04-26,12150
2023-04-27,12120
2023-04-28,12200
2023-05-01,12180
2023-05-02,12250
2023-05-03,12220
2023-05-04,12300
2023-05-05,12280
2023-05-08,12350
2023-05-09,12320
2023-05-10,12400
2023-05-11,12380
2023-05-12,12450
2023-05-15,12420
2023-05-16,12500
2023-05-17,12480
2023-05-18,12550
2023-05-19,12520
2023-05-22,12600
2023-05-23,12580
2023-05-24,12650
2023-05-25,12620
2023-05-26,12700
2023-05-29,12680
2023-05-30,12750
2023-05-31,12720
2023-06-01,12800
2023-06-02,12780
2023-06-05,12850
2023-06-06,12820
2023-06-07,12900
2023-06-08,12880
2023-06-09,12950
2023-06-12,12920
2023-06-13,13000
2023-06-14,12980
2023-06-15,13050
2023-06-16,13020
2023-06-19,13100
2023-06-20,13080
2023-06-21,13150
2023-06-22,13120
2023-06-23,13200
2023-06-26,13180
2023-06-27,13250
2023-06-28,13220
2023-06-29,13300
2023-06-30,13280"""
    
    with open(test_dir / "equity_curve.csv", "w") as f:
        f.write(equity_data)
    
    # Create basic metrics file
    metrics = {
        "performance": {
            "sharpe": 1.45,
            "total_return": 0.328,
            "max_drawdown": -0.045
        },
        "charts": {}
    }
    
    with open(test_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return test_dir


def test_basic_functionality():
    """Test basic math functions without dependencies."""
    
    print("Testing basic functionality...")
    
    # Test data
    nav_series = [10000, 10050, 10020, 10080, 10100, 10150, 10120, 10200]
    
    # Test daily returns calculation (manual)
    daily_returns = []
    for i in range(1, len(nav_series)):
        daily_ret = (nav_series[i] - nav_series[i-1]) / nav_series[i-1]
        daily_returns.append(daily_ret)
    
    print(f"Daily returns: {daily_returns[:5]}...")
    
    # Test Sortino calculation (manual)
    risk_free_rate = 0.0
    daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = [r - daily_rf_rate for r in daily_returns]
    
    negative_returns = [r for r in excess_returns if r < 0]
    print(f"Found {len(negative_returns)} negative returns")
    
    if len(negative_returns) > 0:
        mean_daily_return = sum(excess_returns) / len(excess_returns)
        annualized_return = mean_daily_return * 252
        
        # Downside deviation
        mean_negative = sum(negative_returns) / len(negative_returns)
        variance = sum((r - mean_negative)**2 for r in negative_returns) / len(negative_returns)
        downside_std = variance ** 0.5
        downside_deviation = downside_std * (252 ** 0.5)
        
        if downside_deviation > 0:
            sortino_ratio = annualized_return / downside_deviation
            print(f"Sortino ratio: {sortino_ratio}")
        else:
            print("Sortino ratio: N/A (zero downside deviation)")
    else:
        print("Sortino ratio: N/A (no negative returns)")
    
    print("Basic functionality test completed!")
    return True


if __name__ == "__main__":
    print("Running tests...")
    
    # Test basic math
    test_basic_functionality()
    
    # Create test data
    test_dir = create_test_data()
    print(f"Created test data in: {test_dir}")
    
    # Test if files exist
    equity_file = test_dir / "equity_curve.csv"
    metrics_file = test_dir / "metrics.json"
    
    print(f"Equity curve file exists: {equity_file.exists()}")
    print(f"Metrics file exists: {metrics_file.exists()}")
    
    print("Test setup completed!")