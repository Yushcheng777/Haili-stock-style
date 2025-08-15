"""
Test cases for indicator functions in haili_backtest.py
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import haili_backtest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from haili_backtest import detect_rsi_divergence, RSI_VALIDITY_THRESHOLD


def test_rsi_validity_threshold():
    """Test RSI validity threshold behavior with different fractions of non-NaN values."""
    
    # Create test data with known non-NaN fraction
    # 10 data points, window of 5, so we'll test at position 4 (0-indexed)
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    
    # Case 1: RSI with 60% non-NaN values (3 out of 5 values)
    # Should pass with threshold 0.5 (60% > 50%)
    rsi_60_percent = pd.Series([np.nan, np.nan, 30, 40, 50, 60, 70, 80, 90, 85])
    
    result_pass = detect_rsi_divergence(prices, rsi_60_percent, i=4, window=5, 
                                       rsi_validity_threshold=0.5)
    # Should not error and return a boolean (either True or False is acceptable since divergence logic is placeholder)
    assert isinstance(result_pass, bool)
    
    # Case 2: RSI with 40% non-NaN values (2 out of 5 values)  
    # Should fail with threshold 0.5 (40% < 50%)
    rsi_40_percent = pd.Series([np.nan, np.nan, np.nan, 40, 50, 60, 70, 80, 90, 85])
    
    result_fail = detect_rsi_divergence(prices, rsi_40_percent, i=4, window=5, 
                                       rsi_validity_threshold=0.5)
    # Should return False because threshold not met
    assert result_fail == False
    
    # Case 3: Test with higher threshold (0.8) - should fail even with 60% data
    result_high_threshold = detect_rsi_divergence(prices, rsi_60_percent, i=4, window=5, 
                                                 rsi_validity_threshold=0.8)
    # Should return False because 60% < 80%
    assert result_high_threshold == False
    
    # Case 4: Test with lower threshold (0.3) - should pass with 40% data
    result_low_threshold = detect_rsi_divergence(prices, rsi_40_percent, i=4, window=5, 
                                                rsi_validity_threshold=0.3)
    # Should not error and return a boolean (40% > 30%)
    assert isinstance(result_low_threshold, bool)


def test_rsi_validity_threshold_edge_cases():
    """Test edge cases for RSI validity threshold."""
    
    # Empty inputs
    empty_prices = pd.Series([])
    empty_rsi = pd.Series([])
    result = detect_rsi_divergence(empty_prices, empty_rsi, i=0, window=5)
    assert result == False
    
    # Single value
    single_prices = pd.Series([100])
    single_rsi = pd.Series([50])
    result = detect_rsi_divergence(single_prices, single_rsi, i=0, window=5)
    assert isinstance(result, bool)
    
    # All NaN RSI values
    prices = pd.Series([100, 101, 102, 103, 104])
    all_nan_rsi = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    result = detect_rsi_divergence(prices, all_nan_rsi, i=4, window=5)
    assert result == False


def test_default_threshold_constant():
    """Test that the default threshold constant is used correctly."""
    
    # Test that module constant exists and has expected value
    assert RSI_VALIDITY_THRESHOLD == 0.5
    
    # Test that default parameter uses the constant
    prices = pd.Series([100, 101, 102, 103, 104])
    rsi = pd.Series([np.nan, np.nan, 30, 40, 50])  # 60% non-NaN
    
    # Call without explicit threshold should use default
    result_default = detect_rsi_divergence(prices, rsi, i=4, window=5)
    
    # Call with explicit threshold same as default should give same result  
    result_explicit = detect_rsi_divergence(prices, rsi, i=4, window=5, 
                                          rsi_validity_threshold=RSI_VALIDITY_THRESHOLD)
    
    assert result_default == result_explicit


if __name__ == "__main__":
    pytest.main([__file__])