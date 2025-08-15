#!/usr/bin/env python3
"""
Unit tests for RSI divergence detection in haili_backtest.py
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add parent directory to path to import haili_backtest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from haili_backtest import detect_rsi_divergence, RSI_VALIDITY_THRESHOLD


def test_bullish_divergence_detected():
    """Test that bullish divergence is correctly detected when price makes lower low and RSI makes higher low."""
    
    # Create test data where price makes lower low and RSI makes higher low
    # Based on actual extrema detection: price mins at 5, 11 and RSI mins at 5, 10
    prices = pd.Series([
        105, 104, 103, 102, 101, 100,  # First low at index 5: price=100
        101, 102, 103, 102, 101, 99,   # Second low at index 11: price=99 (lower)
        100, 101, 102, 103
    ])
    
    rsi_values = pd.Series([
        40, 38, 36, 34, 32, 30,        # First RSI low at index 5: RSI=30  
        32, 34, 36, 34, 25, 35,        # Second RSI low at index 10: RSI=25 - wait, this should be higher
        37, 39, 41, 43
    ])
    
    # Let me fix this - RSI needs to make higher low for bullish divergence
    rsi_values = pd.Series([
        40, 38, 36, 34, 32, 25,        # First RSI low at index 5: RSI=25  
        27, 29, 31, 29, 30, 35,        # Second RSI low at index 10: RSI=30 (higher than 25)
        37, 39, 41, 43
    ])
    
    # Test with sufficient window size
    result = detect_rsi_divergence(prices, rsi_values, window=14)
    assert result == True, "Should detect bullish divergence (lower price low, higher RSI low)"


def test_bearish_divergence_detected():
    """Test that bearish divergence is correctly detected when price makes higher high and RSI makes lower high."""
    
    # Create test data where price makes higher high and RSI makes lower high
    prices = pd.Series([
        95, 96, 97, 98, 99, 100,       # First high at index 5: price=100
        99, 98, 97, 98, 99, 105,       # Second high at index 11: price=105 (higher)
        104, 103, 102, 101
    ])
    
    rsi_values = pd.Series([
        50, 52, 54, 56, 58, 75,        # First RSI high at index 5: RSI=75  
        73, 71, 69, 71, 73, 70,        # Second RSI high should be lower - let me fix the pattern
        68, 66, 64, 62
    ])
    
    # Let me create a clearer pattern for maxima detection
    rsi_values = pd.Series([
        50, 52, 54, 56, 58, 75,        # First RSI high at index 5: RSI=75
        73, 71, 69, 71, 68, 62,        # Second RSI high at index 9: RSI=71 (lower than 75)
        68, 66, 64, 62
    ])
    
    # Test with sufficient window size
    result = detect_rsi_divergence(prices, rsi_values, window=14)
    assert result == True, "Should detect bearish divergence (higher price high, lower RSI high)"


def test_no_divergence_with_insufficient_rsi():
    """Test that function returns False when RSI validity threshold is not met."""
    
    # Create test data with many NaN values in RSI
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114])
    
    # RSI with too many NaN values (only 30% valid, below default threshold of 70%)
    rsi_values = pd.Series([30, np.nan, np.nan, np.nan, 35, np.nan, np.nan, np.nan, 40, np.nan, np.nan, np.nan, 45, np.nan, np.nan])
    
    # Should return False due to insufficient valid RSI values
    result = detect_rsi_divergence(prices, rsi_values, window=14, rsi_validity_threshold=RSI_VALIDITY_THRESHOLD)
    assert result == False, "Should return False when RSI validity threshold is not met"


def test_no_divergence_normal_case():
    """Test that function returns False when no divergence pattern exists."""
    
    # Create test data with normal price and RSI movement (no divergence)
    prices = pd.Series([95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    rsi_values = pd.Series([25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67])
    
    result = detect_rsi_divergence(prices, rsi_values, window=14)
    assert result == False, "Should return False when no divergence pattern exists"


def test_insufficient_data():
    """Test that function returns False when insufficient data is provided."""
    
    # Test with very short series
    prices = pd.Series([100, 101, 102])
    rsi_values = pd.Series([30, 31, 32])
    
    result = detect_rsi_divergence(prices, rsi_values, window=14)
    assert result == False, "Should return False when insufficient data is provided"


def test_empty_series():
    """Test that function returns False for empty input series."""
    
    prices = pd.Series([])
    rsi_values = pd.Series([])
    
    result = detect_rsi_divergence(prices, rsi_values, window=14)
    assert result == False, "Should return False for empty input series"


if __name__ == "__main__":
    # Run tests manually if script is executed directly
    test_bullish_divergence_detected()
    test_bearish_divergence_detected() 
    test_no_divergence_with_insufficient_rsi()
    test_no_divergence_normal_case()
    test_insufficient_data()
    test_empty_series()
    print("All tests passed!")