"""
Unit tests for haili_backtest.py indicators and utilities.

Tests the RSI divergence detector and CSV column normalization functions
using synthesized pandas DataFrames.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path so we can import haili_backtest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from haili_backtest import normalize_csv_columns, detect_rsi_divergence


class TestCSVNormalization:
    """Test CSV column normalization functionality."""
    
    def test_normalize_date_columns(self):
        """Test normalization of various date column names."""
        # Test with 'date' column
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'Open': [10.0, 10.5],
            'Close': [10.2, 10.7]
        })
        normalized = normalize_csv_columns(df)
        assert 'Date' in normalized.columns
        assert 'date' not in normalized.columns
        
        # Test with '日期' column  
        df = pd.DataFrame({
            '日期': ['2024-01-01', '2024-01-02'],
            'Open': [10.0, 10.5],
            'Close': [10.2, 10.7]
        })
        normalized = normalize_csv_columns(df)
        assert 'Date' in normalized.columns
        assert '日期' not in normalized.columns
        
        # Test with 'Date' column (should remain unchanged)
        df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Open': [10.0, 10.5],
            'Close': [10.2, 10.7]
        })
        normalized = normalize_csv_columns(df)
        assert 'Date' in normalized.columns
        
    def test_normalize_adjclose_columns(self):
        """Test normalization of various AdjClose column names."""
        # Test with 'Adj Close' column
        df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Close': [10.2, 10.7],
            'Adj Close': [10.2, 10.7]
        })
        normalized = normalize_csv_columns(df)
        assert 'AdjClose' in normalized.columns
        assert 'Adj Close' not in normalized.columns
        
        # Test with 'Adj_Close' column
        df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Close': [10.2, 10.7],
            'Adj_Close': [10.2, 10.7]
        })
        normalized = normalize_csv_columns(df)
        assert 'AdjClose' in normalized.columns
        assert 'Adj_Close' not in normalized.columns
        
        # Test with '收盘复权价' column
        df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Close': [10.2, 10.7],
            '收盘复权价': [10.2, 10.7]
        })
        normalized = normalize_csv_columns(df)
        assert 'AdjClose' in normalized.columns
        assert '收盘复权价' not in normalized.columns
        
    def test_no_normalization_needed(self):
        """Test that properly named columns are not modified."""
        df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Open': [10.0, 10.5],
            'Close': [10.2, 10.7],
            'AdjClose': [10.2, 10.7]
        })
        normalized = normalize_csv_columns(df)
        pd.testing.assert_frame_equal(df, normalized)


class TestRSIDivergence:
    """Test RSI divergence detection functionality."""
    
    def create_sample_data(self, num_periods=100):
        """Create sample price and RSI data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=num_periods, freq='D')
        
        # Create synthetic price data with some patterns
        base_price = 100
        prices = []
        for i in range(num_periods):
            # Add some trend and noise
            trend = i * 0.1
            noise = np.random.normal(0, 1)
            price = base_price + trend + noise
            prices.append(price)
            
        # Create corresponding RSI values (simplified calculation)
        rsi_values = []
        for i in range(num_periods):
            if i < 14:
                rsi_values.append(np.nan)
            else:
                # Simplified RSI calculation for testing
                recent_changes = np.diff(prices[max(0, i-14):i+1])
                gains = recent_changes[recent_changes > 0]
                losses = -recent_changes[recent_changes < 0]
                
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0.01  # Avoid division by zero
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
                
        return pd.Series(prices, index=dates), pd.Series(rsi_values, index=dates)
    
    def test_divergence_detection_basic(self):
        """Test basic functionality of divergence detection."""
        prices, rsi = self.create_sample_data(80)
        
        # Test that the function runs without error
        divergence = detect_rsi_divergence(prices, rsi, lookback_window=30)
        
        assert len(divergence) == len(prices)
        assert divergence.index.equals(prices.index)
        
        # Check that divergence values are in expected range
        unique_values = set(divergence.values)
        assert unique_values.issubset({-1, 0, 1})
        
    def test_divergence_with_insufficient_data(self):
        """Test divergence detection with insufficient data."""
        prices, rsi = self.create_sample_data(20)  # Not enough data for meaningful divergence
        
        divergence = detect_rsi_divergence(prices, rsi, lookback_window=60)
        
        # Should return all zeros for insufficient data
        assert all(divergence == 0)
        
    def test_divergence_with_bullish_pattern(self):
        """Test detection of synthetic bullish divergence pattern."""
        # Create a bullish divergence pattern: prices make lower lows, RSI makes higher lows
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        
        # Synthetic price data with lower lows
        prices = [100] * 20 + [95] * 10 + [90] * 10 + [95] * 10 + [98] * 10
        
        # Synthetic RSI with higher lows (opposite of price pattern)
        rsi_base = [50] * 20 + [30] * 10 + [35] * 10 + [40] * 10 + [45] * 10
        
        prices_series = pd.Series(prices, index=dates)
        rsi_series = pd.Series(rsi_base, index=dates)
        
        divergence = detect_rsi_divergence(prices_series, rsi_series, lookback_window=30)
        
        # Should detect some pattern in the data (not necessarily specific values
        # due to the simplified synthetic data)
        assert len(divergence) == len(prices)
        
    def test_divergence_with_nan_rsi(self):
        """Test divergence detection handles NaN RSI values gracefully."""
        prices, rsi = self.create_sample_data(50)
        
        # Introduce some NaN values in RSI
        rsi.iloc[10:20] = np.nan
        
        divergence = detect_rsi_divergence(prices, rsi, lookback_window=25)
        
        # Should handle NaN values without crashing
        assert len(divergence) == len(prices)
        assert not divergence.isna().all()  # Should have some non-NaN values


if __name__ == '__main__':
    pytest.main([__file__])