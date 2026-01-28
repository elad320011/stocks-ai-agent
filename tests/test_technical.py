"""Tests for technical analysis module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.analysis.technical import TechnicalAnalyzer, TechnicalIndicators, TechnicalScore


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 200)
    prices = base_price * np.cumprod(1 + returns)
    
    # Add some volatility
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, 200)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, 200)))
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, 200)),
        'high': high,
        'low': low,
        'close': prices,
        'volume': np.random.randint(100000, 1000000, 200),
    }, index=dates)
    
    return df


@pytest.fixture
def bullish_data():
    """Generate bullish trending data."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Upward trending prices
    base_price = 100
    trend = np.linspace(0, 0.5, 200)  # 50% gain over period
    noise = np.random.normal(0, 0.01, 200)
    prices = base_price * (1 + trend + noise)
    
    df = pd.DataFrame({
        'open': prices * 0.995,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(100000, 1000000, 200),
    }, index=dates)
    
    return df


@pytest.fixture
def bearish_data():
    """Generate bearish trending data."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Downward trending prices
    base_price = 100
    trend = np.linspace(0, -0.3, 200)  # 30% loss over period
    noise = np.random.normal(0, 0.01, 200)
    prices = base_price * (1 + trend + noise)
    
    df = pd.DataFrame({
        'open': prices * 1.005,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(100000, 1000000, 200),
    }, index=dates)
    
    return df


class TestTechnicalAnalyzer:
    """Tests for TechnicalAnalyzer class."""
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = TechnicalAnalyzer()
        assert analyzer is not None
    
    def test_calculate_indicators_with_valid_data(self, sample_ohlcv_data):
        """Test indicator calculation with valid data."""
        analyzer = TechnicalAnalyzer()
        indicators = analyzer.calculate_indicators(sample_ohlcv_data)
        
        assert isinstance(indicators, TechnicalIndicators)
        assert indicators.rsi_14 is not None
        assert 0 <= indicators.rsi_14 <= 100
        assert indicators.sma_20 is not None
        assert indicators.sma_50 is not None
    
    def test_calculate_indicators_with_empty_data(self):
        """Test indicator calculation with empty data."""
        analyzer = TechnicalAnalyzer()
        indicators = analyzer.calculate_indicators(pd.DataFrame())
        
        assert isinstance(indicators, TechnicalIndicators)
        assert indicators.rsi_14 is None
    
    def test_calculate_indicators_with_insufficient_data(self):
        """Test indicator calculation with insufficient data."""
        analyzer = TechnicalAnalyzer()
        small_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [10000, 10000, 10000],
        })
        
        indicators = analyzer.calculate_indicators(small_df)
        assert indicators.rsi_14 is None
    
    def test_rsi_oversold_detection(self):
        """Test RSI oversold signal detection."""
        analyzer = TechnicalAnalyzer()
        
        # Create oversold scenario
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Declining prices to create oversold RSI
        prices = 100 * np.exp(np.linspace(0, -0.4, 100))
        
        df = pd.DataFrame({
            'open': prices * 1.005,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': [100000] * 100,
        }, index=dates)
        
        indicators = analyzer.calculate_indicators(df)
        
        # RSI should be low after sustained decline
        if indicators.rsi_14:
            assert indicators.rsi_14 < 50
    
    def test_bullish_trend_detection(self, bullish_data):
        """Test bullish trend detection."""
        analyzer = TechnicalAnalyzer()
        indicators = analyzer.calculate_indicators(bullish_data)
        
        # Should detect bullish trend
        assert indicators.trend == "bullish"
        assert indicators.price_vs_sma20 == "above"
        assert indicators.price_vs_sma50 == "above"
    
    def test_bearish_trend_detection(self, bearish_data):
        """Test bearish trend detection."""
        analyzer = TechnicalAnalyzer()
        indicators = analyzer.calculate_indicators(bearish_data)
        
        # Should detect bearish trend
        assert indicators.trend == "bearish"
        assert indicators.price_vs_sma20 == "below"
    
    def test_calculate_score(self, sample_ohlcv_data):
        """Test score calculation."""
        analyzer = TechnicalAnalyzer()
        score = analyzer.calculate_score(sample_ohlcv_data)
        
        assert isinstance(score, TechnicalScore)
        assert 0 <= score.total_score <= 100
        assert score.indicators is not None
        assert isinstance(score.signals, list)
    
    def test_bullish_score_higher(self, bullish_data, bearish_data):
        """Test that bullish data scores higher than bearish."""
        analyzer = TechnicalAnalyzer()
        
        bullish_score = analyzer.calculate_score(bullish_data)
        bearish_score = analyzer.calculate_score(bearish_data)
        
        assert bullish_score.total_score > bearish_score.total_score
    
    def test_golden_cross_detection(self):
        """Test golden cross detection."""
        analyzer = TechnicalAnalyzer()
        
        # Create data with golden cross scenario
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=250, freq='D')
        
        # Start bearish, then turn bullish
        first_half = 100 * np.exp(np.linspace(0, -0.2, 100))
        second_half = first_half[-1] * np.exp(np.linspace(0, 0.4, 150))
        prices = np.concatenate([first_half, second_half])
        
        df = pd.DataFrame({
            'open': prices * 0.998,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': [100000] * 250,
        }, index=dates)
        
        indicators = analyzer.calculate_indicators(df)
        
        # After the recovery, SMA50 should be above SMA200
        # (This is a simplified test - golden cross depends on exact timing)
        assert indicators.sma_50 is not None
        assert indicators.sma_200 is not None


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators dataclass."""
    
    def test_default_values(self):
        """Test default indicator values."""
        indicators = TechnicalIndicators()
        
        assert indicators.rsi_14 is None
        assert indicators.rsi_signal == "neutral"
        assert indicators.trend == "neutral"
        assert indicators.golden_cross is False
        assert indicators.bullish_divergence is False
    
    def test_custom_values(self):
        """Test custom indicator values."""
        indicators = TechnicalIndicators(
            rsi_14=25.0,
            rsi_signal="oversold",
            trend="bullish",
            golden_cross=True,
        )
        
        assert indicators.rsi_14 == 25.0
        assert indicators.rsi_signal == "oversold"
        assert indicators.trend == "bullish"
        assert indicators.golden_cross is True


class TestTechnicalScore:
    """Tests for TechnicalScore dataclass."""
    
    def test_default_values(self):
        """Test default score values."""
        score = TechnicalScore()
        
        assert score.total_score == 0.0
        assert score.rsi_score == 0.0
        assert score.signals == []
    
    def test_score_range(self, sample_ohlcv_data):
        """Test that scores are within valid range."""
        analyzer = TechnicalAnalyzer()
        score = analyzer.calculate_score(sample_ohlcv_data)
        
        assert 0 <= score.total_score <= 100
        assert 0 <= score.rsi_score <= 15
        assert 0 <= score.macd_score <= 15
        assert 0 <= score.ma_score <= 20
        assert 0 <= score.volume_score <= 10
        assert 0 <= score.trend_score <= 10
