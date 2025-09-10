#!/usr/bin/env python3
"""
Basic test for generate_backtest_report.py functionality.
Tests key functions without requiring external dependencies.
"""

import sys
import os
import tempfile
import shutil

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_backtest_report import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_annualized_return,
    calculate_max_drawdown,
    read_equity_curve_csv
)


def test_calculate_returns():
    """Test daily returns calculation."""
    values = [100, 101, 99, 102]
    returns = calculate_returns(values)
    expected = [0.01, -0.0198, 0.0303]  # Approximate values
    
    assert len(returns) == 3
    assert abs(returns[0] - 0.01) < 1e-6
    assert abs(returns[1] - (-0.019801980198019802)) < 1e-6
    assert abs(returns[2] - 0.030303030303030304) < 1e-6
    print("✓ calculate_returns test passed")


def test_calculate_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    returns = [0.01, -0.005, 0.02, 0.015, -0.01]
    risk_free_rate_daily = 0.0001
    
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate_daily)
    assert sharpe is not None
    assert isinstance(sharpe, float)
    print(f"✓ calculate_sharpe_ratio test passed: {sharpe:.4f}")


def test_calculate_sortino_ratio():
    """Test Sortino ratio calculation."""
    returns = [0.01, -0.005, 0.02, 0.015, -0.01]
    risk_free_rate_daily = 0.0001
    
    sortino = calculate_sortino_ratio(returns, risk_free_rate_daily)
    assert sortino is not None
    assert isinstance(sortino, float)
    print(f"✓ calculate_sortino_ratio test passed: {sortino:.4f}")


def test_calculate_annualized_return():
    """Test annualized return calculation."""
    values = [100, 105, 110, 120]
    
    ann_return = calculate_annualized_return(values)
    assert ann_return is not None
    assert isinstance(ann_return, float)
    assert ann_return > 0  # Should be positive for this increasing series
    print(f"✓ calculate_annualized_return test passed: {ann_return:.4f}")


def test_calculate_max_drawdown():
    """Test maximum drawdown calculation."""
    values = [100, 105, 95, 110, 90, 120]
    
    max_dd = calculate_max_drawdown(values)
    assert max_dd is not None
    assert isinstance(max_dd, float)
    assert max_dd < 0  # Should be negative
    print(f"✓ calculate_max_drawdown test passed: {max_dd:.4f}")


def test_read_equity_curve_csv():
    """Test CSV reading functionality."""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("date,nav\n")
        f.write("2023-01-01,100.0\n")
        f.write("2023-01-02,101.0\n")
        f.write("2023-01-03,99.0\n")
        temp_file = f.name
    
    try:
        result = read_equity_curve_csv(temp_file)
        assert result is not None
        dates, values = result
        assert len(dates) == 3
        assert len(values) == 3
        assert values == [100.0, 101.0, 99.0]
        print("✓ read_equity_curve_csv test passed")
    finally:
        os.unlink(temp_file)


def run_all_tests():
    """Run all tests."""
    print("Running backtest report generation tests...\n")
    
    test_calculate_returns()
    test_calculate_sharpe_ratio()
    test_calculate_sortino_ratio()
    test_calculate_annualized_return()
    test_calculate_max_drawdown()
    test_read_equity_curve_csv()
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    run_all_tests()