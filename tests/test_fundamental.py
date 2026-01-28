"""Tests for fundamental analysis module."""

import pytest
from unittest.mock import Mock, patch

from src.analysis.fundamental import (
    FundamentalAnalyzer, 
    FundamentalMetrics, 
    FundamentalScore
)
from src.data.yahoo_client import StockQuote


@pytest.fixture
def mock_quote():
    """Create a mock stock quote."""
    from datetime import datetime
    return StockQuote(
        symbol="TEST.TA",
        name="Test Company Ltd",
        price=150.0,
        change=2.5,
        change_percent=1.7,
        volume=500000,
        market_cap=5_000_000_000,
        pe_ratio=12.5,
        eps=12.0,
        dividend_yield=0.03,
        fifty_two_week_high=180.0,
        fifty_two_week_low=100.0,
        avg_volume=400000,
        timestamp=datetime.now(),
    )


@pytest.fixture
def undervalued_quote():
    """Create an undervalued stock quote."""
    from datetime import datetime
    return StockQuote(
        symbol="CHEAP.TA",
        name="Cheap Company Ltd",
        price=50.0,
        change=1.0,
        change_percent=2.0,
        volume=800000,
        market_cap=15_000_000_000,  # Large cap
        pe_ratio=8.0,  # Low P/E
        eps=6.25,
        dividend_yield=0.04,
        fifty_two_week_high=80.0,
        fifty_two_week_low=45.0,  # Near low
        avg_volume=1_200_000,  # High volume
        timestamp=datetime.now(),
    )


@pytest.fixture
def overvalued_quote():
    """Create an overvalued stock quote."""
    from datetime import datetime
    return StockQuote(
        symbol="EXPENSIVE.TA",
        name="Expensive Company Ltd",
        price=200.0,
        change=-5.0,
        change_percent=-2.5,
        volume=50000,
        market_cap=100_000_000,  # Micro cap
        pe_ratio=45.0,  # High P/E
        eps=4.44,
        dividend_yield=None,
        fifty_two_week_high=210.0,  # Near high
        fifty_two_week_low=100.0,
        avg_volume=30000,  # Low volume
        timestamp=datetime.now(),
    )


class TestFundamentalAnalyzer:
    """Tests for FundamentalAnalyzer class."""
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = FundamentalAnalyzer()
        assert analyzer is not None
        assert analyzer.yahoo is not None
    
    def test_get_sector_pe(self):
        """Test sector P/E lookup."""
        analyzer = FundamentalAnalyzer()
        
        assert analyzer._get_sector_pe("technology") == 28.0
        assert analyzer._get_sector_pe("Technology Sector") == 28.0
        assert analyzer._get_sector_pe("financial") == 11.0
        assert analyzer._get_sector_pe("unknown sector") == 16.0  # Default
    
    def test_categorize_market_cap(self):
        """Test market cap categorization."""
        analyzer = FundamentalAnalyzer()
        
        assert analyzer._categorize_market_cap(15_000_000_000) == "large"
        assert analyzer._categorize_market_cap(5_000_000_000) == "mid"
        assert analyzer._categorize_market_cap(1_000_000_000) == "small"
        assert analyzer._categorize_market_cap(100_000_000) == "micro"
    
    def test_categorize_volume(self):
        """Test volume categorization."""
        analyzer = FundamentalAnalyzer()
        
        assert analyzer._categorize_volume(2_000_000) == "high"
        assert analyzer._categorize_volume(500_000) == "medium"
        assert analyzer._categorize_volume(50_000) == "low"
    
    def test_analyze_with_quote(self, mock_quote):
        """Test analysis with provided quote."""
        analyzer = FundamentalAnalyzer()
        metrics = analyzer.analyze(
            "TEST.TA", 
            quote=mock_quote, 
            sector="financial",
            is_ta35=True,
        )
        
        assert isinstance(metrics, FundamentalMetrics)
        assert metrics.pe_ratio == 12.5
        assert metrics.sector_avg_pe == 11.0
        assert metrics.market_cap_category == "mid"
        assert metrics.is_ta35 is True
    
    def test_pe_vs_sector_undervalued(self, undervalued_quote):
        """Test undervalued P/E detection."""
        analyzer = FundamentalAnalyzer()
        metrics = analyzer.analyze(
            "CHEAP.TA",
            quote=undervalued_quote,
            sector="financial",  # Avg P/E = 11
        )
        
        # P/E of 8 vs sector avg of 11 = undervalued
        assert metrics.pe_vs_sector == "undervalued"
    
    def test_pe_vs_sector_overvalued(self, overvalued_quote):
        """Test overvalued P/E detection."""
        analyzer = FundamentalAnalyzer()
        metrics = analyzer.analyze(
            "EXPENSIVE.TA",
            quote=overvalued_quote,
            sector="technology",  # Avg P/E = 28
        )
        
        # P/E of 45 vs sector avg of 28 = overvalued
        assert metrics.pe_vs_sector == "overvalued"
    
    def test_calculate_score(self, mock_quote):
        """Test score calculation."""
        analyzer = FundamentalAnalyzer()
        score = analyzer.calculate_score(
            "TEST.TA",
            quote=mock_quote,
            sector="financial",
            is_ta35=True,
        )
        
        assert isinstance(score, FundamentalScore)
        assert 0 <= score.total_score <= 100
        assert isinstance(score.signals, list)
        assert score.metrics is not None
    
    def test_undervalued_scores_higher(self, undervalued_quote, overvalued_quote):
        """Test that undervalued stocks score higher."""
        analyzer = FundamentalAnalyzer()
        
        undervalued_score = analyzer.calculate_score(
            "CHEAP.TA",
            quote=undervalued_quote,
            sector="financial",
            is_ta35=True,
        )
        
        overvalued_score = analyzer.calculate_score(
            "EXPENSIVE.TA",
            quote=overvalued_quote,
            sector="technology",
        )
        
        assert undervalued_score.total_score > overvalued_score.total_score
    
    def test_ta35_bonus(self, mock_quote):
        """Test that TA-35 membership adds to quality score."""
        analyzer = FundamentalAnalyzer()
        
        ta35_score = analyzer.calculate_score(
            "TEST.TA",
            quote=mock_quote,
            is_ta35=True,
        )
        
        non_ta35_score = analyzer.calculate_score(
            "TEST.TA",
            quote=mock_quote,
            is_ta35=False,
        )
        
        assert ta35_score.quality_score > non_ta35_score.quality_score
    
    def test_liquidity_scoring(self, undervalued_quote, overvalued_quote):
        """Test liquidity scoring based on volume."""
        analyzer = FundamentalAnalyzer()
        
        high_vol_score = analyzer.calculate_score(
            "CHEAP.TA",
            quote=undervalued_quote,  # High volume
        )
        
        low_vol_score = analyzer.calculate_score(
            "EXPENSIVE.TA",
            quote=overvalued_quote,  # Low volume
        )
        
        assert high_vol_score.liquidity_score > low_vol_score.liquidity_score


class TestFundamentalMetrics:
    """Tests for FundamentalMetrics dataclass."""
    
    def test_default_values(self):
        """Test default metric values."""
        metrics = FundamentalMetrics()
        
        assert metrics.pe_ratio is None
        assert metrics.market_cap == 0.0
        assert metrics.pe_vs_sector == "unknown"
        assert metrics.is_ta35 is False
    
    def test_custom_values(self):
        """Test custom metric values."""
        metrics = FundamentalMetrics(
            pe_ratio=15.0,
            market_cap=10_000_000_000,
            pe_vs_sector="fair",
            is_ta35=True,
        )
        
        assert metrics.pe_ratio == 15.0
        assert metrics.market_cap == 10_000_000_000
        assert metrics.pe_vs_sector == "fair"
        assert metrics.is_ta35 is True


class TestFundamentalScore:
    """Tests for FundamentalScore dataclass."""
    
    def test_default_values(self):
        """Test default score values."""
        score = FundamentalScore()
        
        assert score.total_score == 0.0
        assert score.valuation_score == 0.0
        assert score.liquidity_score == 0.0
        assert score.signals == []
    
    def test_score_components(self, mock_quote):
        """Test that score has all components."""
        analyzer = FundamentalAnalyzer()
        score = analyzer.calculate_score(
            "TEST.TA",
            quote=mock_quote,
            sector="financial",
        )
        
        # All component scores should be set
        assert score.valuation_score >= 0
        assert score.liquidity_score >= 0
        assert score.quality_score >= 0
        assert score.momentum_score >= 0
        
        # Total should be combination of components
        assert score.total_score > 0
